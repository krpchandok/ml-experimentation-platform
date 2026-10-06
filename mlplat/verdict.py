from dataclasses import asdict, dataclass, field
from typing import Optional

RULES_VERSION = 1

GPU_BUSY_PCT = 80.0
GPU_STARVED_PCT = 50.0
WORKER_SATURATED_PCT = 80.0
WORKER_IDLE_PCT = 30.0
MAIN_SATURATED_PCT = 85.0
MAIN_SINGLE_CORE_CEILING_PCT = 150.0
MAIN_WAITING_PCT = 60.0
IO_LOW_CPU_PCT = 50.0
IO_BLKIO_WAIT_PCT = 20.0
IO_READ_HIGH_BYTES_PER_S = 50 * 1024 * 1024
IO_SYSTEM_IOWAIT_PCT = 10.0
MEMORY_NEAR_LIMIT_FRACTION = 0.90
MEMORY_CRITICAL_FRACTION = 0.97
MEMORY_AVAILABLE_LOW_FRACTION = 0.05
MAJFLT_HIGH_PER_S = 50.0
MAJFLT_GROWTH_FACTOR = 2.0
HEALTHY_CPU_CORE_FRACTION = 0.70
DATA_WAIT_HIGH_FRACTION = 0.30
DATA_WAIT_LOW_FRACTION = 0.10
MIN_WINDOW_SAMPLES = 3
OOM_EXIT_CODE = 137

SEVERITY_ORDER = ("critical", "bottleneck", "warning", "ok", "info", "unknown")
RULE_PRIORITY = ("memory_pressure", "gpu_not_used", "preprocessing_bound", "io_bound", "main_process_bound",
                 "healthy", "underutilized", "insufficient_data")


@dataclass
class GpuFeatures:
    device_count: int
    util_mean_pct: float
    util_p50_pct: float
    mem_util_mean_pct: Optional[float] = None
    memory_used_peak_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    tree_memory_peak_mb: Optional[float] = None
    process_info: bool = False
    tree_on_gpu: Optional[bool] = None
    power_mean_w: Optional[float] = None
    util_source: Optional[str] = None


@dataclass
class Features:
    window_samples: int
    window_seconds: float
    usable_cores: float
    tree_cores_mean: float
    main_cpu_mean_pct: float
    worker_count: int
    worker_cpu_mean_pct: Optional[float] = None
    worker_cpu_min_pct: Optional[float] = None
    main_read_bytes_per_s: Optional[float] = None
    worker_read_bytes_per_s: Optional[float] = None
    main_blkio_wait_pct: Optional[float] = None
    worker_blkio_wait_pct: Optional[float] = None
    system_iowait_pct: Optional[float] = None
    rss_peak_kb: Optional[float] = None
    memory_limit_kb: Optional[float] = None
    memory_limit_source: Optional[str] = None
    mem_available_min_fraction: Optional[float] = None
    majflt_early_per_s: Optional[float] = None
    majflt_late_per_s: Optional[float] = None
    oom_kills: Optional[int] = None
    exit_code: Optional[int] = None
    data_wait_fraction: Optional[float] = None
    gpu: Optional[GpuFeatures] = None


@dataclass
class Finding:
    verdict: str
    severity: str
    title: str
    evidence: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    signals: dict = field(default_factory=dict)


def pct(value):
    return f"{value:.0f}%"


def rate(value):
    return f"{value / (1024 * 1024):.1f} MiB/s"


def data_wait_evidence(features):
    if features.data_wait_fraction is None:
        return []
    return [f"The training loop reported spending {features.data_wait_fraction:.0%} of its time waiting for the "
            f"next batch (logged data_time)."]


def waits_for_data(features):
    return features.data_wait_fraction is not None and features.data_wait_fraction >= DATA_WAIT_HIGH_FRACTION


def consumer_starved(features):
    if features.gpu is not None:
        util = features.gpu.util_mean_pct
        if util < GPU_STARVED_PCT:
            return True, [f"Meanwhile GPU utilization averaged {pct(util)} "
                          f"(starved below {pct(GPU_STARVED_PCT)})."] + data_wait_evidence(features)
        if util < GPU_BUSY_PCT and waits_for_data(features):
            return True, [f"Meanwhile GPU utilization averaged only {pct(util)} (busy at {pct(GPU_BUSY_PCT)}) and the "
                          f"training loop reported spending {features.data_wait_fraction:.0%} of its time waiting for "
                          f"the next batch (starved at {DATA_WAIT_HIGH_FRACTION:.0%}, logged data_time)."]
        return False, []
    if features.main_cpu_mean_pct < MAIN_WAITING_PCT:
        return True, [f"Meanwhile the main training process averaged {pct(features.main_cpu_mean_pct)} of a core, "
                      f"so it spent most of its time waiting for batches (waiting below {pct(MAIN_WAITING_PCT)})."
                      ] + data_wait_evidence(features)
    if waits_for_data(features):
        return True, [f"Meanwhile the training loop reported spending {features.data_wait_fraction:.0%} of its time "
                      f"waiting for the next batch (starved at {DATA_WAIT_HIGH_FRACTION:.0%}, logged data_time)."]
    return False, []


def consumer_busy(features):
    if features.gpu is not None:
        return features.gpu.util_mean_pct >= GPU_BUSY_PCT
    return False


def rule_preprocessing_bound(features):
    if features.worker_count < 1 or features.worker_cpu_mean_pct is None:
        return None
    if features.worker_cpu_mean_pct < WORKER_SATURATED_PCT:
        return None
    starved, starved_evidence = consumer_starved(features)
    if not starved:
        return None
    spare_cores = max(0, int(features.usable_cores - features.tree_cores_mean))
    suggestions = [
        f"Increase DataLoader num_workers (currently about {features.worker_count}; "
        f"roughly {spare_cores} cores are unused)." if spare_cores else
        "All cores are busy: reduce per-sample preprocessing cost rather than adding workers.",
        "Move augmentation to the GPU (for example torchvision.transforms.v2 on CUDA tensors) or precompute it.",
        "Cache decoded or preprocessed samples so each epoch does less CPU work.",
    ]
    return Finding(
        verdict="preprocessing_bound",
        severity="bottleneck",
        title="Preprocessing-bound: data-loader workers cannot keep up",
        evidence=[
            f"{features.worker_count} data-loader worker(s) each averaged {pct(features.worker_cpu_mean_pct)} "
            f"of a core (saturated at {pct(WORKER_SATURATED_PCT)}).",
            *starved_evidence,
        ],
        suggestions=suggestions,
        signals={"worker_count": features.worker_count, "worker_cpu_mean_pct": features.worker_cpu_mean_pct,
                 "main_cpu_mean_pct": features.main_cpu_mean_pct,
                 "gpu_util_mean_pct": features.gpu.util_mean_pct if features.gpu else None},
    )


def rule_main_process_bound(features):
    main = features.main_cpu_mean_pct
    inline_loading = features.worker_count == 0 and waits_for_data(features)
    single_thread = MAIN_SATURATED_PCT <= main < MAIN_SINGLE_CORE_CEILING_PCT
    if main < MAIN_SATURATED_PCT or not (single_thread or inline_loading):
        return None
    workers_idle = features.worker_count == 0 or (features.worker_cpu_mean_pct or 0) < WORKER_IDLE_PCT
    pipeline_keeps_up = (features.gpu is not None and features.data_wait_fraction is not None
                         and features.data_wait_fraction <= DATA_WAIT_LOW_FRACTION)
    if not (workers_idle or pipeline_keeps_up) or consumer_busy(features):
        return None
    if single_thread:
        evidence = [f"The main training process averaged {pct(main)} of one core, i.e. a single thread was "
                    f"saturated (saturated at {pct(MAIN_SATURATED_PCT)}, multi-threaded above "
                    f"{pct(MAIN_SINGLE_CORE_CEILING_PCT)})."]
    else:
        evidence = [f"The main training process was busy ({pct(main)} of a core on average) and also produced "
                    f"every batch itself."]
    if features.worker_count == 0:
        evidence.append("There were no data-loader worker processes, so loading and preprocessing ran in the "
                        "main process, serialized with the training step.")
        suggestions = [
            "Set DataLoader num_workers > 0 so preprocessing runs in parallel with the training step.",
            "Profile the loop (py-spy, torch.profiler) to separate data time from step time.",
        ]
    elif not workers_idle:
        evidence.append(f"The input pipeline kept up: the loop waited for data only {features.data_wait_fraction:.0%} "
                        f"of the time (keeping up at or below {DATA_WAIT_LOW_FRACTION:.0%}), so the GPU idles between "
                        f"kernels launched by the saturated Python thread.")
        suggestions = [
            "Remove per-step host-device syncs: .item(), .cpu(), printing or logging tensors every step, and "
            "explicit torch.cuda.synchronize().",
            "Do more work per Python step: a larger batch size, torch.compile or CUDA graphs, mixed precision.",
        ]
    else:
        evidence.append(f"The {features.worker_count} data-loader worker(s) were mostly idle "
                        f"({pct(features.worker_cpu_mean_pct)} of a core each, idle below {pct(WORKER_IDLE_PCT)}).")
        suggestions = [
            "Look for Python-side overhead in the training loop: per-step .item()/.cpu() syncs, logging, "
            "metric computation, or Python loops over tensors.",
            "Use a larger batch size, torch.compile, or mixed precision to do more work per Python step.",
        ]
    if features.gpu is not None:
        evidence.append(f"GPU utilization averaged {pct(features.gpu.util_mean_pct)} "
                        f"(busy at {pct(GPU_BUSY_PCT)}).")
    if workers_idle:
        evidence += data_wait_evidence(features)
    return Finding(
        verdict="main_process_bound",
        severity="bottleneck",
        title="Main-process-bound: the training loop's Python thread is the bottleneck",
        evidence=evidence,
        suggestions=suggestions,
        signals={"main_cpu_mean_pct": main, "worker_count": features.worker_count,
                 "worker_cpu_mean_pct": features.worker_cpu_mean_pct,
                 "gpu_util_mean_pct": features.gpu.util_mean_pct if features.gpu else None},
    )


def rule_io_bound(features):
    if features.worker_count > 0:
        role, cpu = "data-loader workers", features.worker_cpu_mean_pct or 0.0
        read_rate, blkio = features.worker_read_bytes_per_s, features.worker_blkio_wait_pct
    else:
        role, cpu = "main training process", features.main_cpu_mean_pct
        read_rate, blkio = features.main_read_bytes_per_s, features.main_blkio_wait_pct
    if cpu >= IO_LOW_CPU_PCT or consumer_busy(features):
        return None
    if features.gpu is None and features.main_cpu_mean_pct >= MAIN_SATURATED_PCT:
        return None
    triggers = []
    if blkio is not None and blkio >= IO_BLKIO_WAIT_PCT:
        triggers.append(f"the {role} spent {pct(blkio)} of their time blocked on disk I/O "
                        f"(threshold {pct(IO_BLKIO_WAIT_PCT)})")
    if read_rate is not None and read_rate >= IO_READ_HIGH_BYTES_PER_S:
        triggers.append(f"the {role} read from storage at {rate(read_rate)} "
                        f"(high above {rate(IO_READ_HIGH_BYTES_PER_S)})")
    if features.system_iowait_pct is not None and features.system_iowait_pct >= IO_SYSTEM_IOWAIT_PCT:
        triggers.append(f"system-wide iowait averaged {pct(features.system_iowait_pct)} "
                        f"(threshold {pct(IO_SYSTEM_IOWAIT_PCT)})")
    if not triggers:
        return None
    evidence = [f"The {role} used only {pct(cpu)} of a core (low below {pct(IO_LOW_CPU_PCT)})."]
    evidence += [f"At the same time {trigger}." for trigger in triggers]
    evidence += data_wait_evidence(features)
    if features.gpu is not None:
        evidence.append(f"GPU utilization averaged {pct(features.gpu.util_mean_pct)}.")
    return Finding(
        verdict="io_bound",
        severity="bottleneck",
        title="I/O-bound: data loading is waiting on storage",
        evidence=evidence,
        suggestions=[
            "Put the dataset on faster local storage (local NVMe instead of network or /mnt/c on WSL).",
            "Cache the dataset in memory or in a preprocessed, contiguous format (e.g. WebDataset shards, LMDB).",
            "Use larger sequential reads and more workers or prefetch_factor to overlap I/O with compute.",
        ],
        signals={"role": role, "cpu_mean_pct": cpu, "read_bytes_per_s": read_rate, "blkio_wait_pct": blkio,
                 "system_iowait_pct": features.system_iowait_pct},
    )


def rule_memory_pressure(features):
    triggers = []
    critical = False
    limit = features.memory_limit_kb
    if features.rss_peak_kb is not None and limit:
        fraction = features.rss_peak_kb / limit
        if fraction >= MEMORY_NEAR_LIMIT_FRACTION:
            critical = critical or fraction >= MEMORY_CRITICAL_FRACTION
            triggers.append(f"peak resident memory of the run was {features.rss_peak_kb / 1024:.0f} MiB, "
                            f"{fraction:.0%} of the {features.memory_limit_source} limit of {limit / 1024:.0f} MiB "
                            f"(threshold {MEMORY_NEAR_LIMIT_FRACTION:.0%})")
    if (features.mem_available_min_fraction is not None
            and features.mem_available_min_fraction < MEMORY_AVAILABLE_LOW_FRACTION):
        triggers.append(f"system available memory dropped to {features.mem_available_min_fraction:.1%} of total "
                        f"(low below {MEMORY_AVAILABLE_LOW_FRACTION:.0%})")
    early, late = features.majflt_early_per_s, features.majflt_late_per_s
    if late is not None and late >= MAJFLT_HIGH_PER_S and late >= MAJFLT_GROWTH_FACTOR * max(early or 0.0, 1e-9):
        triggers.append(f"major page faults rose from {early or 0:.0f}/s early in the run to {late:.0f}/s late "
                        f"(high above {MAJFLT_HIGH_PER_S:.0f}/s and growing {MAJFLT_GROWTH_FACTOR:.0f}x)")
    if features.oom_kills:
        critical = True
        triggers.append(f"the kernel OOM killer fired {features.oom_kills} time(s) in the run's cgroup")
    if features.exit_code == OOM_EXIT_CODE and triggers:
        critical = True
        triggers.append("the training process was killed by SIGKILL (exit 137), consistent with the OOM killer")
    if not triggers:
        return None
    return Finding(
        verdict="memory_pressure",
        severity="critical" if critical else "warning",
        title="Memory pressure: the run is at or near its memory limit",
        evidence=[f"{trigger[0].upper()}{trigger[1:]}." for trigger in triggers],
        suggestions=[
            "Reduce batch size or the number of data-loader workers (each worker holds its own copy of dataset state).",
            "Avoid holding whole datasets in Python lists in every worker; use memory-mapped arrays instead.",
            "Raise the memory limit if the workload legitimately needs it.",
        ],
        signals={"rss_peak_kb": features.rss_peak_kb, "memory_limit_kb": limit,
                 "memory_limit_source": features.memory_limit_source,
                 "mem_available_min_fraction": features.mem_available_min_fraction,
                 "majflt_early_per_s": early, "majflt_late_per_s": late, "oom_kills": features.oom_kills,
                 "exit_code": features.exit_code},
    )


def rule_gpu_not_used(features):
    gpu = features.gpu
    if gpu is None or not gpu.process_info or gpu.tree_on_gpu is not False:
        return None
    if gpu.util_mean_pct >= GPU_STARVED_PCT:
        return None
    return Finding(
        verdict="gpu_not_used",
        severity="bottleneck",
        title="GPU not used: none of this run's processes are on the GPU",
        evidence=[
            "NVML listed compute processes on the GPU during the run, but none of them belong to this run's "
            "process tree.",
            f"GPU utilization averaged {pct(gpu.util_mean_pct)} while the main process averaged "
            f"{pct(features.main_cpu_mean_pct)} of a core.",
        ],
        suggestions=[
            "Check that the model and each batch are moved to the device (model.to('cuda'), batch.to('cuda')) and "
            "that torch.cuda.is_available() is True inside the run.",
            "Inside a container NVML can report host PIDs that do not match the run's PIDs; confirm with nvidia-smi "
            "before acting on this verdict.",
        ],
        signals={"gpu_util_mean_pct": gpu.util_mean_pct, "tree_on_gpu": gpu.tree_on_gpu,
                 "main_cpu_mean_pct": features.main_cpu_mean_pct},
    )


def rule_healthy(features):
    if features.gpu is not None:
        if features.gpu.util_mean_pct < GPU_BUSY_PCT:
            return None
        evidence = [f"GPU utilization averaged {pct(features.gpu.util_mean_pct)} (busy at {pct(GPU_BUSY_PCT)})."]
    else:
        fraction = features.tree_cores_mean / features.usable_cores if features.usable_cores else 0.0
        pipeline_keeps_up = (features.data_wait_fraction is not None
                             and features.data_wait_fraction <= DATA_WAIT_LOW_FRACTION
                             and features.main_cpu_mean_pct >= MAIN_SATURATED_PCT)
        if fraction >= HEALTHY_CPU_CORE_FRACTION:
            evidence = [f"The run kept {features.tree_cores_mean:.1f} of {features.usable_cores:.0f} usable cores "
                        f"busy ({fraction:.0%}, healthy at {HEALTHY_CPU_CORE_FRACTION:.0%})."]
        elif pipeline_keeps_up:
            evidence = [f"The training loop waited for data only {features.data_wait_fraction:.0%} of the time "
                        f"(healthy at or below {DATA_WAIT_LOW_FRACTION:.0%}, logged data_time) while the main process "
                        f"computed at {pct(features.main_cpu_mean_pct)} of a core: the run is compute-bound.",
                        f"It used {features.tree_cores_mean:.1f} of {features.usable_cores:.0f} usable cores; more "
                        f"compute threads or a GPU would be the next speedup."]
        else:
            return None
    return Finding(
        verdict="healthy",
        severity="ok",
        title="Healthy: the compute device is kept busy",
        evidence=evidence,
        suggestions=["No input-pipeline bottleneck detected; further speedups come from the model itself "
                     "(mixed precision, larger batches, better kernels)."],
        signals={"gpu_util_mean_pct": features.gpu.util_mean_pct if features.gpu else None,
                 "tree_cores_mean": features.tree_cores_mean, "usable_cores": features.usable_cores},
    )


def rule_underutilized(features):
    fraction = features.tree_cores_mean / features.usable_cores if features.usable_cores else 0.0
    evidence = [f"The run used {features.tree_cores_mean:.1f} of {features.usable_cores:.0f} usable cores "
                f"({fraction:.0%}) and no bottleneck rule matched."]
    evidence.append(f"Main process averaged {pct(features.main_cpu_mean_pct)} of a core"
                    + (f"; {features.worker_count} worker(s) averaged {pct(features.worker_cpu_mean_pct)}."
                       if features.worker_count else "; there were no worker processes."))
    if features.gpu is not None:
        evidence.append(f"GPU utilization averaged {pct(features.gpu.util_mean_pct)}.")
    evidence += data_wait_evidence(features)
    return Finding(
        verdict="underutilized",
        severity="info",
        title="Underutilized: neither the CPU nor the accelerator is saturated",
        evidence=evidence,
        suggestions=[
            "Look for waiting that is not CPU or disk: sleeps, locks, network or remote storage, or "
            "host-device synchronization.",
            "A small batch size or a tiny model can also leave hardware idle; try scaling the batch size.",
        ],
        signals={"tree_cores_mean": features.tree_cores_mean, "usable_cores": features.usable_cores,
                 "main_cpu_mean_pct": features.main_cpu_mean_pct,
                 "worker_cpu_mean_pct": features.worker_cpu_mean_pct},
    )


BOTTLENECK_RULES = (rule_memory_pressure, rule_gpu_not_used, rule_preprocessing_bound, rule_io_bound,
                    rule_main_process_bound, rule_healthy)


def insufficient_data(reason):
    return Finding(verdict="insufficient_data", severity="unknown", title="Insufficient data for a verdict",
                   evidence=[reason], suggestions=["Run for longer than a few sampling intervals with the agent "
                                                   "available."])


def sort_key(finding):
    return SEVERITY_ORDER.index(finding.severity), RULE_PRIORITY.index(finding.verdict)


def thresholds():
    return {name: value for name, value in globals().items()
            if name.isupper() and isinstance(value, (int, float)) and name != "RULES_VERSION"}


def decide(features, missing_reason=None):
    if missing_reason is not None:
        findings = [insufficient_data(missing_reason)]
    elif features.window_samples < MIN_WINDOW_SAMPLES:
        findings = [insufficient_data(f"Only {features.window_samples} resource sample(s) fell inside the analysis "
                                      f"window (need {MIN_WINDOW_SAMPLES}).")]
    else:
        findings = [finding for rule in BOTTLENECK_RULES if (finding := rule(features)) is not None]
        if not any(finding.severity in ("bottleneck", "ok") for finding in findings):
            findings.append(rule_underutilized(features))
    findings.sort(key=sort_key)
    primary, secondary = findings[0], findings[1:]
    return {
        "rules_version": RULES_VERSION,
        "primary": primary.verdict,
        "severity": primary.severity,
        "title": primary.title,
        "evidence": primary.evidence,
        "suggestions": primary.suggestions,
        "signals": primary.signals,
        "secondary": [asdict(finding) for finding in secondary],
        "thresholds": thresholds(),
    }

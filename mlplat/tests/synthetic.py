import json

from mlplat.run_store import RunStore

MAIN_PID = 100
RESOURCE_TRACKER_CMDLINE = "python -c from multiprocessing.resource_tracker import main;main(5)"


class SyntheticRun:
    def __init__(self, root, name="synthetic", usable_cpus=8, mem_total_kb=16 * 1024 * 1024, exit_code=0,
                 agent_available=True):
        self.run = RunStore(root).create()
        self.mem_total_kb = mem_total_kb
        self.meta = {
            "run_id": self.run.run_id, "name": name, "status": "completed" if exit_code == 0 else "failed",
            "exit_code": exit_code, "start_mono": 0.0, "start_wall": 1_700_000_000.0, "duration_s": 0.0,
            "pid": MAIN_PID, "config_hash": "abc", "host": {"usable_cpus": usable_cpus, "mem_total_kb": mem_total_kb},
            "agent": {"available": agent_available},
        }
        self.records = [{"type": "header", "root_pid": MAIN_PID, "online_cpus": usable_cpus, "delayacct": True}]
        self.metrics = []
        self.seq = 0
        self.add_process(MAIN_PID, 1, "python train.py", 0.0)

    def add_process(self, pid, ppid, cmdline, t):
        self.records.append({"type": "process", "t_mono": t, "pid": pid, "ppid": ppid, "start_ticks": pid,
                             "comm": cmdline.split()[0], "cmdline": cmdline})

    def exit_process(self, pid, t):
        self.records.append({"type": "exit", "t_mono": t, "pid": pid, "ppid": 1, "start_ticks": pid, "comm": "x",
                             "cpu_s": 0})

    def sample(self, t, procs, dt=1.0, system=None, gpu=None, cgroup=None):
        self.seq += 1
        entries = []
        for pid, values in procs.items():
            entry = {"pid": pid, "ppid": MAIN_PID if pid != MAIN_PID else 1, "comm": "python", "state": "R",
                     "threads": 1, "cpu_pct": 0.0, "rss_kb": 100_000, "cpu_s": 0.0, "read_bytes": 0,
                     "write_bytes": 0, "read_bytes_per_s": 0.0, "write_bytes_per_s": 0.0, "majflt_per_s": 0.0,
                     "blkio_wait_pct": None}
            entry.update(values)
            entries.append(entry)
        record = {
            "type": "sample", "seq": self.seq, "t_mono": t, "t_wall": 1_700_000_000.0 + t, "dt": dt,
            "system": {"cpu_count": self.meta["host"]["usable_cpus"], "cpu_iowait_pct": 0.0,
                       "mem_total_kb": self.mem_total_kb, "mem_available_kb": self.mem_total_kb // 2,
                       **(system or {})},
            "tree": {"nproc": len(entries), "cpu_pct": sum(entry["cpu_pct"] for entry in entries),
                     "rss_kb": sum(entry["rss_kb"] for entry in entries)},
            "procs": entries,
        }
        if gpu is not None:
            record["gpu"] = gpu
        if cgroup is not None:
            record["cgroup"] = cgroup
        self.records.append(record)

    def steady(self, start, seconds, procs, **extra):
        for second in range(seconds):
            self.sample(start + second + 1, procs, **extra)

    def log_steps(self, start, end, step_time, every=1):
        step, t = 0, start
        while t <= end + 1e-9:
            self.metrics.append({"t_mono": t, "t_wall": 1_700_000_000.0 + t, "step": step, "loss": 1.0 / (step + 1)})
            step += every
            t += step_time * every

    def write(self):
        last = max((record["t_mono"] for record in self.records if "t_mono" in record), default=0.0)
        self.meta["duration_s"] = last
        self.meta["end_mono"] = last
        self.records.append({"type": "end", "reason": "root_exited", "samples": self.seq})
        self.run.write_meta(self.meta)
        with open(self.run.resources_path, "w") as handle:
            for record in self.records:
                handle.write(json.dumps(record) + "\n")
        with open(self.run.metrics_path, "w") as handle:
            for record in self.metrics:
                handle.write(json.dumps(record) + "\n")
        return self.run


def gpu_row(util, devices=1, procs=None, mem_used_mb=2000.0):
    return {
        "devices": [{"index": index, "util_pct": util if isinstance(util, (int, float)) else util[index],
                     "mem_util_pct": 30.0, "mem_used_mb": mem_used_mb, "mem_total_mb": 8192.0}
                    for index in range(devices)],
        "procs": procs or [{"pid": MAIN_PID, "device": 0, "mem_used_mb": mem_used_mb}],
    }


def scenario(root, main_cpu, worker_cpus, seconds=20, gpu_util=None, worker_extra=None, main_extra=None,
             system=None, cgroup=None, usable_cpus=8, mem_total_kb=16 * 1024 * 1024, exit_code=0):
    run = SyntheticRun(root, usable_cpus=usable_cpus, mem_total_kb=mem_total_kb, exit_code=exit_code)
    worker_pids = [200 + index for index in range(len(worker_cpus))]
    for pid in worker_pids:
        run.add_process(pid, MAIN_PID, "python train.py", 0.0)
    procs = {MAIN_PID: {"cpu_pct": main_cpu, **(main_extra or {})}}
    for pid, cpu in zip(worker_pids, worker_cpus):
        procs[pid] = {"cpu_pct": cpu, **(worker_extra or {})}
    gpu = gpu_row(gpu_util) if gpu_util is not None else None
    run.steady(0, seconds, procs, system=system, gpu=gpu, cgroup=cgroup)
    run.log_steps(0.5, seconds - 0.5, 0.25)
    return run.write()

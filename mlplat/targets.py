from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS_FILE_NAME = "targets.yaml"
LOCAL_TARGET_ID = "this-machine"
BAKERIES = ("home_kitchen", "community_kitchen", "market_stall", "rented_bakery")
NUMERIC_FIELDS = ("relative_gpu_speed", "relative_cpu_speed", "cpu_cores", "memory_gb", "gpu_memory_gb",
                  "cost_per_hour", "session_hours", "weekly_hours")
REQUIRED_FIELDS = ("relative_gpu_speed", "cpu_cores", "cost_per_hour")
DEFAULTS = {"relative_cpu_speed": 1.0}
KIB_PER_GIB = 1024 * 1024


class TargetError(ValueError):
    pass


@dataclass
class Target:
    id: str
    name: str
    description: str = ""
    bakery: str = "rented_bakery"
    runnable_here: bool = False
    measured: bool = False
    gpu_name: Optional[str] = None
    relative_gpu_speed: float = 1.0
    relative_cpu_speed: float = 1.0
    cpu_cores: float = 1.0
    memory_gb: Optional[float] = None
    gpu_memory_gb: Optional[float] = None
    cost_per_hour: float = 0.0
    session_hours: Optional[float] = None
    weekly_hours: Optional[float] = None
    placeholders: list = field(default_factory=list)
    sources: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


def default_targets_path():
    local = Path.cwd() / TARGETS_FILE_NAME
    return local if local.exists() else REPO_ROOT / TARGETS_FILE_NAME


def unpack(raw, name, entry_id):
    if isinstance(raw, dict):
        if "value" not in raw:
            raise TargetError(f"target {entry_id}: field {name} needs a 'value'")
        return raw["value"], bool(raw.get("placeholder", False)), raw.get("source")
    return raw, False, None


def parse_target(entry):
    if not isinstance(entry, dict) or not entry.get("id") or not entry.get("name"):
        raise TargetError("every target needs an 'id' and a 'name'")
    target_id = str(entry["id"])
    target = Target(id=target_id, name=str(entry["name"]), description=str(entry.get("description", "")),
                    bakery=str(entry.get("bakery", "rented_bakery")),
                    runnable_here=bool(entry.get("runnable_here", False)))
    if target.bakery not in BAKERIES:
        raise TargetError(f"target {target_id}: bakery must be one of {', '.join(BAKERIES)}")
    for name in REQUIRED_FIELDS:
        if name not in entry:
            raise TargetError(f"target {target_id}: missing required field {name}")
    for name in NUMERIC_FIELDS + ("gpu_name",):
        if name not in entry:
            continue
        value, placeholder, source = unpack(entry[name], name, target_id)
        if name in NUMERIC_FIELDS and value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                raise TargetError(f"target {target_id}: {name} must be a number, got {value!r}") from None
            if value < 0 or (name in ("relative_gpu_speed", "relative_cpu_speed", "cpu_cores") and value == 0):
                raise TargetError(f"target {target_id}: {name} must be positive")
        if value is None and name in DEFAULTS:
            value = DEFAULTS[name]
        setattr(target, name, value)
        if placeholder:
            target.placeholders.append(name)
        if source:
            target.sources[name] = source
    return target


def load_targets(path=None):
    path = Path(path) if path else default_targets_path()
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    entries = document.get("targets", []) if isinstance(document, dict) else []
    targets = [parse_target(entry) for entry in entries]
    ids = [target.id for target in targets]
    duplicates = sorted({target_id for target_id in ids if ids.count(target_id) > 1 or target_id == LOCAL_TARGET_ID})
    if duplicates:
        raise TargetError(f"duplicate or reserved target ids: {', '.join(duplicates)}")
    return targets


def local_target(meta, header=None):
    host = meta.get("host") or {}
    gpu_header = (header or {}).get("gpu") or {}
    devices = gpu_header.get("devices") or []
    gpus = host.get("gpus") or []
    gpu_name = devices[0]["name"] if devices else (gpus[0]["name"] if gpus else None)
    gpu_memory_mb = devices[0].get("mem_total_mb") if devices else (gpus[0].get("memory_total_mib") if gpus else None)
    mem_kb = host.get("mem_total_kb")
    label = f"This machine ({gpu_name})" if gpu_name else "This machine (CPU only)"
    return Target(
        id=LOCAL_TARGET_ID,
        name=label,
        description=f"Measured on {host.get('hostname', 'this host')}: {host.get('cpu_model', 'unknown CPU')}.",
        bakery="home_kitchen",
        runnable_here=True,
        measured=True,
        gpu_name=gpu_name,
        relative_gpu_speed=1.0,
        relative_cpu_speed=1.0,
        cpu_cores=float(host.get("usable_cpus") or host.get("logical_cpus") or 1),
        memory_gb=mem_kb / KIB_PER_GIB if mem_kb else None,
        gpu_memory_gb=gpu_memory_mb / 1024.0 if gpu_memory_mb else None,
        cost_per_hour=0.0,
        sources={"all": "measured by mlplat on this machine"},
    )

import hashlib
import json

import yaml

HASH_LENGTH = 16


def load_config(path):
    with open(path, encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if loaded is not None else {}


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(config, command):
    if config is not None:
        source, payload = "config", config
    else:
        source, payload = "command", {"command": list(command)}
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()[:HASH_LENGTH]
    return digest, source


def flatten(value, prefix=""):
    if isinstance(value, dict):
        items = {}
        for key, inner in value.items():
            items.update(flatten(inner, f"{prefix}{key}."))
        return items
    if isinstance(value, list):
        return {prefix.rstrip("."): canonical_json(value)}
    return {prefix.rstrip("."): value}

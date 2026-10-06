import re

import pytest

from mlplat.config import config_hash, flatten
from mlplat.run_store import RunStore, new_run_id


def test_run_ids_are_sortable_and_unique():
    ids = {new_run_id(0) for _ in range(200)}
    assert len(ids) == 200
    assert all(re.fullmatch(r"19700101-000000-[0-9a-f]{6}", run_id) for run_id in ids)
    assert new_run_id(0) < new_run_id(86400)


def test_create_get_and_list(tmp_path):
    store = RunStore(tmp_path)
    first = store.create()
    first.write_meta({"name": "baseline"})
    second = store.create()
    second.write_meta({"name": "tuned"})
    assert [run.run_id for run in store.list()] == sorted([first.run_id, second.run_id])
    assert store.get(first.run_id).path == first.path
    assert store.get("tuned").path == second.path
    assert store.get(second.run_id[:-1]).path == second.path
    with pytest.raises(KeyError, match="no run"):
        store.get("missing")
    with pytest.raises(KeyError, match="ambiguous"):
        store.get(first.run_id[:4])
    assert not list(tmp_path.glob("**/.*.tmp"))


def test_list_ignores_directories_without_meta(tmp_path):
    (tmp_path / "junk").mkdir()
    assert RunStore(tmp_path).list() == []
    assert RunStore(tmp_path / "missing").list() == []


def test_config_hash_is_stable_across_key_order():
    first, source = config_hash({"lr": 0.1, "model": {"depth": 3, "width": 64}}, ["python", "a.py"])
    second, _ = config_hash({"model": {"width": 64, "depth": 3}, "lr": 0.1}, ["python", "b.py"])
    third, _ = config_hash({"lr": 0.2, "model": {"depth": 3, "width": 64}}, ["python", "a.py"])
    assert source == "config"
    assert first == second
    assert first != third
    assert len(first) == 16


def test_config_hash_falls_back_to_command():
    first, source = config_hash(None, ["python", "train.py", "--workers", "4"])
    second, _ = config_hash(None, ["python", "train.py", "--workers", "8"])
    assert source == "command"
    assert first != second


def test_flatten_nested_config():
    assert flatten({"a": 1, "b": {"c": 2, "d": {"e": "x"}}, "f": [1, 2]}) == {
        "a": 1, "b.c": 2, "b.d.e": "x", "f": "[1,2]",
    }

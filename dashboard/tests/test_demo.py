import html
import json
import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import demo, wording
from mlplat.analyzer import analyze_run
from synthetic import MAIN_PID, SyntheticRun, gpu_row

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def timed_run(root, name, workers, step_time, data_time, gpu_util, agent_cpu=0.15):
    run = SyntheticRun(root, name=name)
    run.meta["host"]["gpus"] = [{"name": "NVIDIA GeForce RTX 4060", "memory_total_mib": 8188}]
    run.meta["plan_request"] = {"total_steps": 20000}
    run.records[0]["gpu"] = {"available": True, "devices": [{"index": 0, "name": "NVIDIA GeForce RTX 4060",
                                                              "mem_total_mb": 8188}]}
    pids = list(range(201, 201 + workers))
    for pid in pids:
        run.add_process(pid, MAIN_PID, "python train.py", 0.0)
    procs = {MAIN_PID: {"cpu_pct": 40}, **{pid: {"cpu_pct": 100} for pid in pids}}
    run.steady(0, 30, procs, gpu=gpu_row(gpu_util))
    run.log_steps(2.0, 29.0, step_time)
    for record in run.metrics:
        record.update(data_time=data_time, compute_time=step_time - data_time)
    written = run.write()
    lines = written.resources_path.read_text().splitlines()
    end = json.loads(lines[-1])
    end.update(agent_cpu_pct=agent_cpu, sample_us_mean=2500.0, agent_max_rss_kb=24 * 1024)
    lines[-1] = json.dumps(end)
    written.resources_path.write_text("\n".join(lines) + "\n")
    analyze_run(written)
    return written


@pytest.fixture
def demo_store(runs_dir, tmp_path, monkeypatch):
    timed_run(runs_dir, "diag", 1, 0.040, 0.030, 22.0, agent_cpu=0.17)
    timed_run(runs_dir, "slow", 1, 0.040, 0.028, 33.0, agent_cpu=0.07)
    timed_run(runs_dir, "fast", 6, 0.008, 0.0004, 77.0, agent_cpu=0.20)
    timed_run(runs_dir, "check", 4, 0.011, 0.001, 70.0, agent_cpu=0.14)
    config = tmp_path / "demo.yaml"
    config.write_text("author: Built by Someone\nlinks:\n  github: https://github.com/example/repo\n"
                      "runs:\n  diagnosis: diag\n  before: slow\n  after: fast\n  validation: [check, gone]\n")
    monkeypatch.setenv(demo.CONFIG_ENV, str(config))
    monkeypatch.setenv(demo.DEMO_ENV, "1")
    return config


def test_demo_mode_switch(monkeypatch):
    monkeypatch.setenv(demo.DEMO_ENV, "1")
    assert demo.enabled()
    monkeypatch.setenv(demo.DEMO_ENV, "off")
    assert not demo.enabled()
    monkeypatch.delenv(demo.DEMO_ENV)
    assert not demo.enabled()


def test_facts_come_from_the_run_files(demo_store, runs_dir):
    facts = demo.load_facts(str(runs_dir), demo.load_config())
    assert facts.gpu_name == "NVIDIA GeForce RTX 4060"
    assert facts.missing == ["gone"]
    assert facts.total_steps == 20000
    before, after = facts.runs["before"], facts.runs["after"]
    assert before.steps_per_s == pytest.approx(25.0) and after.steps_per_s == pytest.approx(125.0)
    assert before.workers == 1 and after.workers == 6
    assert before.gpu_busy == pytest.approx(0.33) and after.gpu_busy == pytest.approx(0.77)
    assert facts.runs["diagnosis"].idle_s == pytest.approx(27 * 0.78)
    assert facts.overhead["count"] == 4
    assert facts.overhead["cpu_mean_pct"] == pytest.approx((0.17 + 0.07 + 0.20 + 0.14) / 4)
    assert facts.overhead["cpu_max_pct"] == pytest.approx(0.20)
    assert facts.overhead["sample_ms"] == pytest.approx(2.5)
    row = facts.accuracy[0]
    assert row["run"] == "check" and row["workers"] == 4
    assert row["actual_ms"] == pytest.approx(11.0)
    assert row["predicted_ms"] == pytest.approx(10.0, rel=0.05)
    assert facts.plan["recommendation"]["fix_first"] is not None


def test_missing_demo_runs_are_reported(runs_dir, tmp_path):
    facts = demo.load_facts(str(runs_dir), {"runs": {"diagnosis": "nope", "before": "nope2", "after": "nope3"}})
    assert not facts.runs and set(facts.missing) == {"nope", "nope2", "nope3"}


def bodies(app):
    return html.unescape("\n".join(element.proto.body for element in app.get("html")))


def run_app(page=None):
    app = AppTest.from_file(APP, default_timeout=60)
    app.run()
    if page:
        app.switch_page(page).run()
    assert not app.exception, [item.value for item in app.exception]
    return app


def test_start_page_is_the_demo_landing_page(demo_store):
    app = run_app()
    text = bodies(app)
    assert wording.DEMO_ONE_LINER in text
    words = " ".join(re.sub(r"<[^>]+>", " ", text).split())
    assert "25 125" in words and "5.0× faster" in words and "33% 77%" in words
    assert "real recording from an NVIDIA GeForce RTX 4060" in text
    assert "Built by Someone" in text
    assert [button.label for button in app.button] == [wording.DEMO_TOUR_BUTTON, wording.DEMO_HOW_BUTTON]
    assert any("Demo runs are missing" in item.value and "gone" in item.value for item in app.info)


def test_tour_walks_through_five_steps(demo_store):
    app = run_app("views/tour.py")
    seen = []
    for step in range(1, 6):
        text = bodies(app)
        seen.append(wording.DEMO_TOUR[step - 1][0])
        assert wording.DEMO_STEP_OF.format(step=step, total=5) in text
        assert wording.DEMO_TOUR[step - 1][0] in text
        if step < 5:
            next(button for button in app.button if button.label == wording.DEMO_NEXT).click().run()
            assert not app.exception
    assert "Try it yourself" in bodies(app)
    next(button for button in app.button if button.label == wording.DEMO_BACK).click().run()
    assert wording.DEMO_STEP_OF.format(step=4, total=5) in bodies(app)


def test_tour_numbers_match_the_runs(demo_store):
    app = run_app("views/tour.py")
    assert "sat idle for 21 seconds of 27 seconds" in bodies(app)
    for _ in range(2):
        next(button for button in app.button if button.label == wording.DEMO_NEXT).click().run()
    assert "(1 → 6)" in bodies(app) and "5.0× faster" in bodies(app)


def test_how_it_works_shows_measured_overhead_and_accuracy(demo_store):
    text = bodies(run_app("views/how.py"))
    for name, _, _ in wording.DEMO_HOW_PARTS:
        assert name in text
    for name, _ in wording.DEMO_HOW_TECH:
        assert name in text
    assert "0.15%" in text and "0.20%" in text and "2.5 ms" in text
    assert "4 workers" in text and "% off" in text


def test_demo_pages_are_absent_without_demo_mode(runs_dir, monkeypatch):
    monkeypatch.delenv(demo.DEMO_ENV, raising=False)
    app = run_app()
    assert wording.DEMO_ONE_LINER not in bodies(app)
    assert wording.PAGE_TITLES["home"] in bodies(app)

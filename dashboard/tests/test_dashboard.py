from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from dashboard import charts, data
from mlplat.analyzer import analyze_run
from synthetic import MAIN_PID, SyntheticRun

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_run_rows_include_verdicts_and_final_metrics(populated, runs_dir):
    rows = {row.name: row for row in data.run_rows(str(runs_dir))}
    assert rows["synthetic"].verdict == "preprocessing_bound"
    assert rows["synthetic"].severity == "bottleneck"
    assert rows["fast-gpu"].verdict == "healthy"
    assert rows["fast-gpu"].final_accuracy == pytest.approx(0.9)
    assert rows["fast-gpu"].steps_per_sec == pytest.approx(10.0, rel=0.01)


def test_detail_frames_are_time_aligned(populated):
    view = data.load_detail(populated["fast"])
    assert list(view.process_series["label"].cat.categories)[0] == "main"
    assert set(view.process_series["label"]) == {"main", "worker 201", "worker 202", "worker 203"}
    assert view.tree["t"].min() == pytest.approx(1.0)
    assert view.window == pytest.approx((0.5, 19.5))
    assert view.gpu["util_pct"].mean() == pytest.approx(91.0)
    assert view.metrics["step_time_s"].dropna().median() == pytest.approx(0.1)


def test_many_workers_fold_into_other(runs_dir):
    run = SyntheticRun(runs_dir)
    for pid in range(201, 213):
        run.add_process(pid, MAIN_PID, "python train.py", 0.0)
    run.steady(0, 5, {MAIN_PID: {"cpu_pct": 50}, **{pid: {"cpu_pct": 10} for pid in range(201, 213)}})
    view = data.load_detail(run.write())
    labels = list(view.process_series["label"].cat.categories)
    assert len(labels) == data.MAX_NAMED_PROCESSES + 1
    assert labels[-1] == data.OTHER_PROCESSES
    other = view.process_series[view.process_series["label"] == data.OTHER_PROCESSES]
    assert other["cpu_pct"].iloc[0] == pytest.approx(60.0)


def test_comparison_frame_reports_direction(populated):
    table = data.comparison_frame(populated["slow"].read_summary(), populated["fast"].read_summary())
    step = table[table["metric"] == "Median step time (ms)"].iloc[0]
    assert step["run A"] == pytest.approx(250.0)
    assert step["run B"] == pytest.approx(100.0)
    assert step["change %"] == pytest.approx(-60.0)
    assert step["better"] == "B"


def test_timeline_figure_has_one_row_per_signal_and_stable_colors(populated):
    theme = charts.theme_for("light")
    view = data.load_detail(populated["fast"])
    figure = charts.timeline_figure(view, theme, ["loss", "accuracy"])
    titles = [annotation.text for annotation in figure.layout.annotations]
    assert titles == ["CPU per process (% of one core)", "Memory (MiB resident)", "Storage I/O (MiB/s)",
                      "GPU utilization (%)", "loss", "accuracy", "Step time (ms)"]
    main = next(trace for trace in figure.data if trace.name == "main" and trace.yaxis == "y")
    assert main.line.color == theme["series"][0]
    assert len(figure.layout.shapes) == len(titles)
    dark = charts.timeline_figure(view, charts.theme_for("dark"), [])
    dark_main = next(trace for trace in dark.data if trace.name == "main" and trace.yaxis == "y")
    assert dark_main.line.color == charts.THEMES["dark"]["series"][0]


def test_post_exit_sample_is_not_drawn_as_a_drop_to_zero(runs_dir):
    run = SyntheticRun(runs_dir)
    run.steady(0, 5, {MAIN_PID: {"cpu_pct": 50}})
    run.sample(6, {})
    view = data.load_detail(run.write())
    assert view.tree["t"].max() == pytest.approx(5.0)
    assert view.tree["cpu_cores"].min() == pytest.approx(0.5)


def test_timeline_without_gpu_or_metrics(runs_dir):
    run = SyntheticRun(runs_dir)
    run.steady(0, 5, {MAIN_PID: {"cpu_pct": 50}})
    figure = charts.timeline_figure(data.load_detail(run.write()), charts.theme_for("light"), [])
    titles = [annotation.text for annotation in figure.layout.annotations]
    assert "GPU utilization (%)" not in titles and "Step time (ms)" not in titles


def run_app(page=None, **query):
    app = AppTest.from_file(APP, default_timeout=30)
    for key, value in query.items():
        app.query_params[key] = value
    app.run()
    if page:
        app.switch_page(page).run()
    assert not app.exception, [item.value for item in app.exception]
    return app


def test_runs_page_lists_runs(populated):
    app = run_app()
    frame = app.dataframe[0].value
    assert set(frame["Run"]) == {"synthetic", "fast-gpu"}
    assert "⚠️ preprocessing bound" in set(frame["Efficiency verdict"])
    assert "✅ healthy" in set(frame["Efficiency verdict"])


def test_detail_page_shows_verdict_first(populated):
    app = run_app("pages/run_detail.py", run=populated["slow"].run_id)
    assert app.title[0].value == "synthetic"
    assert "Preprocessing-bound" in app.warning[0].value
    assert "What to try" in app.warning[0].value
    assert [metric.label for metric in app.metric][:2] == ["Median step", "Steps / s"]


def test_detail_page_for_gpu_run_shows_gpu_tile(populated):
    app = run_app("pages/run_detail.py", run=populated["fast"].run_id)
    assert "Healthy" in app.success[0].value
    assert any(metric.label == "GPU util" and metric.value == "91%" for metric in app.metric)


def test_compare_page_renders_table(populated):
    app = run_app("pages/compare.py", a=populated["slow"].run_id, b=populated["fast"].run_id)
    table = app.dataframe[0].value
    assert "Median step time (ms)" in set(table["metric"])


def test_pages_handle_an_empty_store(runs_dir):
    for page in (None, "pages/run_detail.py", "pages/compare.py"):
        app = run_app(page)
        assert app.info, page


def test_unanalyzed_run_still_renders(runs_dir):
    run = SyntheticRun(runs_dir, name="raw")
    run.steady(0, 5, {MAIN_PID: {"cpu_pct": 50}})
    run.write()
    app = run_app("pages/run_detail.py")
    assert any("not been analyzed" in item.value for item in app.info)

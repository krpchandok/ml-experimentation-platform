import html
from pathlib import Path

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from dashboard import art, charts, data, scene, theme, wording
from mlplat.analyzer import analyze_run
from synthetic import MAIN_PID, SyntheticRun, gpu_row

APP = str(Path(__file__).resolve().parents[1] / "app.py")


@pytest.fixture
def fake_art(tmp_path, monkeypatch):
    folder = tmp_path / "vendor"
    folder.mkdir()
    for name in art.KITCHEN + art.FOOD:
        Image.new("RGBA", (512, 512), (200, 120, 40, 255)).save(folder / art.file_for(name).name)
    monkeypatch.setenv(art.ASSETS_ENV, str(folder))
    art.encoded.cache_clear()
    return folder


@pytest.fixture
def no_art(tmp_path, monkeypatch):
    monkeypatch.setenv(art.ASSETS_ENV, str(tmp_path / "missing"))
    art.encoded.cache_clear()


def html_bodies(app):
    return html.unescape("\n".join(element.proto.body for element in app.get("html")))


def run_app(page=None, technical=False, **query):
    app = AppTest.from_file(APP, default_timeout=60)
    for key, value in query.items():
        app.query_params[key] = value
    if technical:
        app.session_state["tech"] = True
    app.run()
    if page:
        app.switch_page(page).run()
    assert not app.exception, [item.value for item in app.exception]
    return app


def test_run_rows_include_verdicts_and_final_metrics(populated, runs_dir):
    rows = {row.name: row for row in data.run_rows(str(runs_dir))}
    assert rows["synthetic"].verdict == "preprocessing_bound"
    assert rows["fast-gpu"].verdict == "healthy"
    assert rows["fast-gpu"].final_accuracy == pytest.approx(0.9)


def test_detail_frames_are_time_aligned(populated):
    view = data.load_detail(populated["fast"])
    assert list(view.process_series["label"].cat.categories)[0] == "main"
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
    assert len(labels) == data.MAX_NAMED_PROCESSES + 1 and labels[-1] == data.OTHER_PROCESSES


def test_comparison_frame_reports_direction(populated):
    table = data.comparison_frame(populated["slow"].read_summary(), populated["fast"].read_summary())
    step = table[table["metric"] == "Median step time (ms)"].iloc[0]
    assert step["change %"] == pytest.approx(-60.0) and step["better"] == "B"


def test_post_exit_sample_is_not_drawn_as_a_drop_to_zero(runs_dir):
    run = SyntheticRun(runs_dir)
    run.steady(0, 5, {MAIN_PID: {"cpu_pct": 50}})
    run.sample(6, {})
    view = data.load_detail(run.write())
    assert view.tree["cpu_cores"].min() == pytest.approx(0.5)


def test_charts_use_plain_titles_unless_technical(populated):
    view = data.load_detail(populated["fast"])
    plain = charts.timeline_figure(view, charts.theme_for("light"), ["loss"], technical=False)
    titles = [annotation.text for annotation in plain.layout.annotations]
    assert titles[:4] == ["How hard each baker worked", "Counter space used", "Ingredients fetched from the pantry",
                          "How busy the oven was"]
    names = {trace.name for trace in plain.data}
    assert {"head baker", "baker 201", "oven 0", "typical tray"} <= names
    technical = charts.timeline_figure(view, charts.theme_for("dark"), [], technical=True)
    assert technical.layout.annotations[0].text == "How hard each baker worked  (CPU per process (% of one core))"
    assert {"main", "GPU 0"} <= {trace.name for trace in technical.data}


def test_wording_helpers():
    assert wording.story_sentence("preprocessing_bound", idle=30, span=51) == (
        "Your oven sat empty for 30 seconds of 51 seconds, waiting for dough.")
    assert wording.story_sentence("preprocessing_bound") == "The oven keeps waiting for the next tray of dough."
    assert wording.slowest_label("preprocessing_bound", 1, 1.01) == "Slowest step: your 1 baker, busy 100% of the time"
    assert wording.busy(1.01, technical=True) == "101%"
    assert wording.bakers(0) == "no bakers" and wording.bakers(3) == "3 bakers"
    assert wording.cost(0) == "Free" and wording.cost(0.004) == "Under $0.01" and wording.cost(2.5) == "$2.50"
    assert set(wording.VERDICT_PLAIN) == set(wording.PLAIN_TRY) == set(wording.STORY_SENTENCES)


def test_art_is_downscaled_and_missing_art_falls_back(fake_art, tmp_path, monkeypatch):
    uri = art.data_uri("oven_closed")
    assert uri.startswith("data:image/png;base64,")
    import base64
    import io
    image = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert image.size == (256, 256)
    assert art.available()
    monkeypatch.setenv(art.ASSETS_ENV, str(tmp_path / "nowhere"))
    assert not art.available() and art.data_uri("oven_closed") is None
    assert 'class="art-missing' in art.img("oven_closed", "oven")
    assert "jimal-art.itch.io" in art.missing_message() and "Jimal" in art.attribution_html()


def test_scene_timing_shows_waiting_or_queueing():
    base = dict(verdict="preprocessing_bound", workers=1, worker_busy=1.0, oven_busy=0.2, steps_per_s=27.0,
                read_mib_s=0.1, slowest_label="Slowest step", slowest_stage="bakers")
    starved = scene.timing(scene.SceneNumbers(compute_s=0.008, prep_s=0.037, **base))
    assert starved["ovenBound"] is False and starved["arrival"] == scene.VISUAL_CYCLE_MS
    assert starved["bake"] == scene.MIN_BAKE_MS and starved["slowdown"] == 59
    busy = scene.timing(scene.SceneNumbers(compute_s=0.006, prep_s=0.005, **base))
    assert busy["ovenBound"] is True and busy["arrival"] < busy["bake"] + 2 * busy["door"]


def test_scene_html_is_static_first_and_respects_reduced_motion(fake_art):
    numbers = scene.SceneNumbers(verdict="preprocessing_bound", workers=8, worker_busy=1.2, compute_s=0.008,
                                 prep_s=0.037, oven_busy=0.22, steps_per_s=27.1, read_mib_s=0.1,
                                 slowest_label="Slowest step: your 8 bakers", slowest_stage="bakers")
    page = scene.scene_html(numbers, theme.TOKENS["light"])
    assert page.count('class="stage') == 4 and 'class="stage slow"' in page
    assert "Slowest step: your 8 bakers" in page and "+2 more" in page and "100% busy" in page
    assert "prefers-reduced-motion: reduce" in page and "matchMedia('(prefers-reduced-motion: reduce)')" in page
    assert "oven_open" not in page and "open oven, empty and waiting" in page
    still = scene.scene_html(numbers, theme.TOKENS["dark"], show_animation=False)
    assert 'class="lane"' not in still


def test_measured_scene_follows_measured_waiting(populated, runs_dir):
    run = planned(runs_dir)
    starved = data.scene_numbers(run, run.read_summary())
    assert starved.prep_s > starved.compute_s and starved.slowest_stage == "bakers"
    assert starved.compute_s == pytest.approx(0.040 * 0.25)
    busy = data.scene_numbers(populated["fast"], populated["fast"].read_summary())
    assert busy.compute_s > busy.prep_s and busy.oven_busy == pytest.approx(0.91)


def test_icons_and_pills_carry_text():
    pill = theme.pill("Finished", "good")
    assert "Finished" in pill and "bh-ic-check" in pill
    assert "mask:" in theme.icon_css() and "bh-ic-arrow" in theme.icon_css()


def planned(runs_dir):
    run = SyntheticRun(runs_dir, name="starved-loader")
    run.add_process(201, MAIN_PID, "python train.py", 0.0)
    run.steady(0, 30, {MAIN_PID: {"cpu_pct": 30}, 201: {"cpu_pct": 100}}, gpu=gpu_row(20.0))
    run.log_steps(2.0, 29.0, 0.040)
    for record in run.metrics:
        record.update(data_time=0.030, compute_time=0.010)
    written = run.write()
    analyze_run(written)
    return written


def test_home_page_puts_fix_first_and_badges_the_best_kitchen(runs_dir, fake_art):
    run = planned(runs_dir)
    app = run_app(run=run.run_id)
    bodies = html_bodies(app)
    assert wording.FIX_TITLE in bodies and "Add bakers (1 → " in bodies
    assert wording.BEST_BADGE in bodies and "This machine" in bodies
    assert "Colab (free)" in bodies and wording.CARD_PLACEHOLDER in bodies
    assert "Cozy Bakery & Food Icons Pack" in bodies
    assert not any("Bakery art not found" in item.value for item in app.info)


def test_home_page_shows_technical_details_only_when_asked(runs_dir, fake_art):
    run = planned(runs_dir)
    assert not any(expander.label == wording.TECH_HEADING for expander in run_app(run=run.run_id).expander)
    technical = run_app(technical=True, run=run.run_id)
    assert any(expander.label == wording.TECH_HEADING for expander in technical.expander)


def test_story_page_leads_with_a_sentence_and_the_bakery(runs_dir, fake_art):
    run = planned(runs_dir)
    app = run_app("views/story.py", run=run.run_id)
    bodies = html_bodies(app)
    assert "Your oven sat empty for 22 seconds of 27 seconds, waiting for dough." in bodies
    assert "Oven sat empty" in bodies and "Counter space used" in bodies
    srcdoc = app.get("iframe")[0].proto.srcdoc
    assert "Slowest step: your 1 baker" in srcdoc
    assert "The bakers can't keep up" in app.warning[0].value
    assert "data-loader worker" in app.warning[0].value


def test_compare_page_shows_before_and_after(populated, fake_art):
    app = run_app("views/compare.py", a=populated["slow"].run_id, b=populated["fast"].run_id)
    bodies = html_bodies(app)
    assert "250 ms" in bodies and "100 ms" in bodies and "2.5× faster" in bodies
    assert "Median step time (ms)" in set(app.dataframe[0].value["metric"])


def test_runs_page_lists_cards_with_plain_verdicts(populated, fake_art):
    bodies = html_bodies(run_app("views/runs.py"))
    assert "The bakers can't keep up" in bodies and "The oven stays busy" in bodies and "Finished" in bodies


def test_pages_work_without_art_and_without_runs(runs_dir, no_art):
    for page in (None, "views/story.py", "views/compare.py", "views/runs.py"):
        app = run_app(page)
        assert "art-missing" in html_bodies(app)
        assert any("Bakery art not found" in item.value for item in app.info)
        assert "Nothing baked yet" in html_bodies(app) or page == "views/story.py"

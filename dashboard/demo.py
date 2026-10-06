import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import streamlit as st
import yaml

from dashboard import art, data, scene, theme, ui, wording
from mlplat.predict import calibrate, extract_profile, predict
from mlplat.run_store import read_jsonl
from mlplat.targets import LOCAL_TARGET_ID, local_target

DEMO_ENV = "MLPLAT_DEMO"
CONFIG_ENV = "MLPLAT_DEMO_CONFIG"
CONFIG_PATH = Path(__file__).resolve().parent / "demo.yaml"
TOUR_STEPS = len(wording.DEMO_TOUR)
ROLES = ("diagnosis", "before", "after", "after_more")
e = theme.escape


def enabled():
    value = os.environ.get(DEMO_ENV)
    if value is None:
        try:
            value = st.secrets.get(DEMO_ENV)
        except Exception:
            value = None
    return str(value).lower() in ("1", "true", "yes", "on")


def config_path():
    return Path(os.environ.get(CONFIG_ENV) or CONFIG_PATH)


def load_config(path=None):
    path = path or config_path()
    if not Path(path).exists():
        return {"links": {}, "runs": {}}
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


@dataclass
class RunFacts:
    run_id: str
    name: str
    steps_per_s: Optional[float]
    step_median_s: Optional[float]
    step_mean_s: Optional[float]
    gpu_busy: Optional[float]
    data_wait: Optional[float]
    workers: int
    worker_busy: Optional[float]
    idle_s: Optional[float]
    span_s: Optional[float]
    verdict: str
    severity: str


@dataclass
class DemoFacts:
    gpu_name: str
    runs: dict
    total_steps: int
    plan: Optional[dict]
    overhead: Optional[dict]
    accuracy: list = field(default_factory=list)
    missing: list = field(default_factory=list)


def run_facts(run):
    summary = run.read_summary() or {}
    totals = summary.get("totals") or {}
    workers = (summary.get("roles") or {}).get("workers") or {}
    gpu = summary.get("gpu") or {}
    idle, span, _ = data.oven_idle_seconds(summary) if summary else (None, None, None)
    verdict = summary.get("verdict") or {}
    return RunFacts(
        run_id=run.run_id,
        name=run.read_meta().get("name", run.run_id),
        steps_per_s=totals.get("steps_per_sec"),
        step_median_s=totals.get("step_time_median_s"),
        step_mean_s=totals.get("step_time_mean_s"),
        gpu_busy=gpu["util_mean_pct"] / 100.0 if gpu.get("util_mean_pct") is not None else None,
        data_wait=(summary.get("features") or {}).get("data_wait_fraction"),
        workers=int(round(workers.get("count_median") or 0)),
        worker_busy=workers["cpu_mean_pct"] / 100.0 if workers.get("cpu_mean_pct") is not None else None,
        idle_s=idle,
        span_s=span,
        verdict=verdict.get("primary", "insufficient_data"),
        severity=verdict.get("severity", "unknown"),
    )


def header_of(run):
    return next((record for record in read_jsonl(run.resources_path) if record.get("type") == "header"), {})


def end_of(run):
    return next((record for record in reversed(read_jsonl(run.resources_path)) if record.get("type") == "end"), None)


def gpu_name_of(run):
    devices = (header_of(run).get("gpu") or {}).get("devices") or []
    if devices:
        return devices[0]["name"]
    gpus = (run.read_meta().get("host") or {}).get("gpus") or []
    return gpus[0]["name"] if gpus else "this machine's CPU"


def overhead_facts(runs):
    ends = [end for end in (end_of(run) for run in runs)
            if end and end.get("agent_cpu_pct") is not None and end.get("sample_us_mean") is not None]
    if not ends:
        return None
    return {
        "count": len(ends),
        "cpu_mean_pct": statistics.fmean(end["agent_cpu_pct"] for end in ends),
        "cpu_max_pct": max(end["agent_cpu_pct"] for end in ends),
        "sample_ms": statistics.fmean(end["sample_us_mean"] for end in ends) / 1000.0,
        "rss_mb": max(end.get("agent_max_rss_kb", 0) for end in ends) / 1024.0,
    }


def accuracy_facts(diagnosis, validations):
    profile = extract_profile(diagnosis)
    local = local_target(diagnosis.read_meta(), header_of(diagnosis))
    calibrate(profile, local)
    rows = []
    for run in validations:
        facts = run_facts(run)
        if not facts.step_mean_s:
            continue
        predicted = predict(profile, local, facts.workers, 1).step_time_s
        rows.append({"run": facts.name, "workers": facts.workers, "predicted_ms": predicted * 1000,
                     "actual_ms": facts.step_mean_s * 1000,
                     "error_pct": (predicted - facts.step_mean_s) / facts.step_mean_s * 100})
    return rows


def resolve(store, name):
    try:
        return store.get(name)
    except KeyError:
        return None


def load_facts(directory, config):
    store = data.store(directory)
    names = config.get("runs") or {}
    runs, missing = {}, []
    for role in ROLES:
        if names.get(role):
            run = resolve(store, names[role])
            if run is None or run.read_summary() is None:
                missing.append(names[role])
            else:
                runs[role] = run
    validations = [run for run in (resolve(store, name) for name in names.get("validation") or []) if run]
    missing += [name for name in names.get("validation") or [] if resolve(store, name) is None]
    if not {"diagnosis", "before", "after"} <= set(runs):
        return DemoFacts(gpu_name="", runs={}, total_steps=0, plan=None, overhead=None, missing=missing or ["demo runs"])
    diagnosis = runs["diagnosis"]
    total_steps = int((diagnosis.read_meta().get("plan_request") or {}).get("total_steps")
                      or data.planned_total_steps(diagnosis))
    return DemoFacts(
        gpu_name=gpu_name_of(runs["before"]),
        runs={role: run_facts(run) for role, run in runs.items()},
        total_steps=total_steps,
        plan=data.plan_in_memory(diagnosis, total_steps),
        overhead=overhead_facts(list({run.run_id: run for run in [*runs.values(), *validations]}.values())),
        accuracy=accuracy_facts(diagnosis, validations),
        missing=missing,
    )


def signature(directory, config):
    store = data.store(directory)
    names = config.get("runs") or {}
    paths = [resolve(store, name) for name in [*(names.get(role) for role in ROLES), *(names.get("validation") or [])]
             if name]
    return tuple(ui.file_signature(run.path) for run in paths if run)


@st.cache_data(show_spinner=False)
def cached_facts(directory, signature_key, config_text):
    return load_facts(directory, yaml.safe_load(config_text) or {})


def facts():
    path = config_path()
    config_text = path.read_text() if path.exists() else ""
    config = yaml.safe_load(config_text) or {}
    return cached_facts(ui.runs_dir(), signature(ui.runs_dir(), config), config_text), config


def favicon():
    path = art.file_for("01_croissant")
    return str(path) if path.exists() else ":material/bakery_dining:"


def note(demo_facts):
    if demo_facts.gpu_name:
        st.html(f'<div class="bh-demo-note">{theme.pill("Demo", "info", "info")}'
                f'<span>{e(wording.DEMO_NOTE.format(gpu=demo_facts.gpu_name))}</span></div>')


def missing_message(demo_facts):
    if demo_facts.missing:
        st.info(wording.DEMO_MISSING.format(names=", ".join(demo_facts.missing)), icon=":material/info:")
    return not demo_facts.runs


def ratio_pill(before, after, lower_is_better=False):
    return ui.change_pill(before, after, lower_is_better=lower_is_better)


def headline_cards(demo_facts):
    before, after = demo_facts.runs["before"], demo_facts.runs["after"]
    cards = [
        ui.before_after(wording.DEMO_HEADLINE_SPEED, f"{before.steps_per_s:.0f}", f"{after.steps_per_s:.0f}",
                        ratio_pill(before.steps_per_s, after.steps_per_s)),
        ui.before_after(wording.DEMO_HEADLINE_STEP, wording.short_duration(before.step_median_s),
                        wording.short_duration(after.step_median_s),
                        ratio_pill(before.step_median_s, after.step_median_s, lower_is_better=True), "baked tray"),
        ui.before_after(wording.DEMO_HEADLINE_GPU, wording.percent(before.gpu_busy), wording.percent(after.gpu_busy),
                        theme.pill(f"+{(after.gpu_busy - before.gpu_busy) * 100:.0f} points", "good", "up")
                        if before.gpu_busy is not None and after.gpu_busy is not None else "", "oven"),
    ]
    st.html(f'<div class="bh-kicker" style="margin-top:14px">{e(wording.DEMO_HEADLINE_KICKER)}</div>'
            f'<div class="bh-ba">{"".join(cards)}</div>'
            f'<div class="bh-muted">{e(wording.DEMO_HEADLINE_SUB.format(gpu=demo_facts.gpu_name, before=before.workers, after=after.workers))}</div>')


def links(config):
    columns = st.columns(len(wording.DEMO_LINKS))
    for column, (key, label) in zip(columns, wording.DEMO_LINKS.items()):
        url = (config.get("links") or {}).get(key)
        if url:
            column.link_button(label, url, width="stretch")


def start_page():
    demo_facts, config = facts()
    theme.page_title(wording.DEMO_PAGE_TITLES["start"], wording.DEMO_ONE_LINER)
    if missing_message(demo_facts):
        return
    headline_cards(demo_facts)
    left, right = st.columns(2)
    if left.button(wording.DEMO_TOUR_BUTTON, type="primary", icon=":material/play_arrow:", width="stretch"):
        st.session_state["tour_step"] = 1
        st.switch_page("views/tour.py")
    if right.button(wording.DEMO_HOW_BUTTON, icon=":material/schema:", width="stretch"):
        st.switch_page("views/how.py")
    st.html(f'<div class="bh-card bh-author">{art.img("01_croissant", "croissant", "bh-food")}'
            f'<div class="bh-title">{e(config.get("author", ""))}</div></div>')
    links(config)


def tour_step():
    step = st.session_state.get("tour_step") or int(st.query_params.get("step", 1) or 1)
    step = max(1, min(TOUR_STEPS, int(step)))
    st.session_state["tour_step"] = step
    st.query_params["step"] = str(step)
    return step


def go(step):
    st.session_state["tour_step"] = step
    st.query_params["step"] = str(step)


def tour_values(demo_facts):
    diagnosis, before, after = demo_facts.runs["diagnosis"], demo_facts.runs["before"], demo_facts.runs["after"]
    return {
        "idle": wording.duration(diagnosis.idle_s), "span": wording.duration(diagnosis.span_s),
        "worker_busy": wording.busy(diagnosis.worker_busy), "gpu_busy": wording.percent(diagnosis.gpu_busy),
        "before_workers": before.workers, "after_workers": after.workers,
        "speedup": (before.step_mean_s or 0) / (after.step_mean_s or 1) if after.step_mean_s else 0,
        "total_steps": f"{demo_facts.total_steps:,}",
    }


def problem_visual(demo_facts):
    diagnosis = demo_facts.runs["diagnosis"]
    st.html('<div class="bh-grid-4">' + "".join([
        ui.card(wording.STORY_CARDS["idle"][0], wording.duration(diagnosis.idle_s),
                f"out of {wording.duration(diagnosis.span_s)}", "oven sat empty"),
        ui.card(wording.STORY_CARDS["busy"][0], wording.percent(diagnosis.gpu_busy), "GPU busy", "oven"),
        ui.card(wording.STORY_CARDS["rate"][0], f"{diagnosis.steps_per_s:.1f}", "training steps per second",
                "baked tray"),
    ]) + "</div>")


def diagnosis_visual(demo_facts):
    run = ui.run_for(demo_facts.runs["diagnosis"].run_id)
    numbers = ui.cached_scene_numbers(ui.runs_dir(), run.run_id, ui.file_signature(run.path))
    st.iframe(scene.scene_html(numbers, theme.tokens()), height="content")


def fix_visual(demo_facts):
    headline_cards(demo_facts)
    more = demo_facts.runs.get("after_more")
    if more and more.steps_per_s:
        st.html(f'<div class="bh-sub">{e(wording.DEMO_MORE_TWEAK.format(rate=more.steps_per_s, gpu_busy=wording.percent(more.gpu_busy)))}</div>')


def train_visual(demo_facts):
    plan = demo_facts.plan
    targets = {target["id"]: target for target in plan["targets"]}
    recommendation = plan["recommendation"]
    if recommendation["fix_first"]:
        ui.fix_card(plan, targets)
    predictions = {entry["target_id"]: entry for entry in plan["predictions"]}
    order = recommendation["ranking"] or list(predictions)
    st.html('<div class="bh-grid">' + "".join(
        ui.target_card(targets[target_id], predictions[target_id], recommendation["scenario"],
                       recommendation["best_target_id"]) for target_id in order) + "</div>")
    st.html(f'<div class="bh-muted">{e(wording.HOME_ESTIMATE_NOTE)} {e(wording.HOME_PLACEHOLDER_NOTE)}</div>')


def sandbox_numbers(plan, scenario):
    entry = next(entry for entry in plan["predictions"] if entry["target_id"] == LOCAL_TARGET_ID)
    prediction = entry[scenario]
    profile = plan["profile"]
    workers = prediction["workers"]
    step = prediction["step_time_s"]
    prep_total = profile["prep_core_s"] or 0.0
    worker_busy = min(1.0, prep_total / workers / step) if workers and step else None
    bottleneck = prediction["bottleneck"]
    verdict = "preprocessing_bound" if bottleneck == "bakers" else "healthy"
    return scene.SceneNumbers(
        verdict=verdict, workers=workers, worker_busy=worker_busy, compute_s=prediction["compute_s"],
        prep_s=prediction["prep_s"], oven_busy=prediction["oven_busy"], steps_per_s=1 / step if step else None,
        read_mib_s=None, slowest_label=wording.slowest_label(verdict, workers, worker_busy),
        slowest_stage=wording.SLOWEST_STAGE[verdict], estimate=True,
    ), prediction


def sandbox_visual(demo_facts):
    choice = st.segmented_control("Kitchen setup", list(wording.DEMO_SANDBOX_CHOICES),
                                  format_func=wording.DEMO_SANDBOX_CHOICES.get, default="as_is", required=True,
                                  key="sandbox_choice")
    numbers, prediction = sandbox_numbers(demo_facts.plan, choice or "as_is")
    st.iframe(scene.scene_html(numbers, theme.tokens()), height="content")
    result = wording.DEMO_SANDBOX_RESULT.format(label=wording.DEMO_SANDBOX_CHOICES[choice or "as_is"],
                                                workers=wording.bakers(prediction["workers"]),
                                                step=wording.short_duration(prediction["step_time_s"]),
                                                busy=wording.percent(prediction["oven_busy"]))
    st.html(f'<div class="bh-summary">{e(result)}</div>')


VISUALS = (problem_visual, diagnosis_visual, fix_visual, train_visual, sandbox_visual)


def tour_page():
    demo_facts, _ = facts()
    if missing_message(demo_facts):
        return
    step = tour_step()
    title, template = wording.DEMO_TOUR[step - 1]
    dots = "".join(f'<span class="bh-dot{" on" if index + 1 <= step else ""}"></span>' for index in range(TOUR_STEPS))
    st.html(f'<div class="bh-row bh-progress" role="img" aria-label="{e(wording.DEMO_STEP_OF.format(step=step, total=TOUR_STEPS))}">'
            f'{dots}<span class="bh-muted">{e(wording.DEMO_STEP_OF.format(step=step, total=TOUR_STEPS))}</span></div>'
            f'<div class="bh-page-title">{e(title)}</div>'
            f'<div class="bh-summary">{e(template.format(**tour_values(demo_facts)))}</div>')
    VISUALS[step - 1](demo_facts)
    back, _, forward = st.columns([1, 2, 1])
    if step > 1 and back.button(wording.DEMO_BACK, icon=":material/arrow_back:", width="stretch"):
        go(step - 1)
        st.rerun()
    if step < TOUR_STEPS:
        if forward.button(wording.DEMO_NEXT, type="primary", icon=":material/arrow_forward:", icon_position="right",
                          width="stretch"):
            go(step + 1)
            st.rerun()
    elif forward.button(wording.DEMO_FINISH, type="primary", width="stretch"):
        go(1)
        st.switch_page("views/home.py")


def architecture():
    boxes = []
    for index, (name, tech, _) in enumerate(wording.DEMO_HOW_PARTS):
        arrow = f'<div class="bh-arch-arrow">{theme.icon("arrow", 24)}</div>' if index else ""
        boxes.append(f'{arrow}<div class="bh-arch-box"><div class="bh-title">{e(name)}</div>'
                     f'<div class="bh-kicker">{e(tech)}</div></div>')
    st.html(f'<div class="bh-arch" role="img" aria-label="Agent, then launcher, then analyzer and planner, then dashboard">'
            f'{"".join(boxes)}</div>')
    st.html('<div class="bh-grid">' + "".join(
        f'<div class="bh-card"><div class="bh-title">{e(name)}</div><div class="bh-sub">{e(text)}</div></div>'
        for name, _, text in wording.DEMO_HOW_PARTS) + "</div>")


def how_page():
    demo_facts, config = facts()
    theme.page_title(wording.DEMO_PAGE_TITLES["how"], wording.DEMO_HOW_INTRO)
    architecture()
    st.html('<div class="bh-grid">' + "".join(
        f'<div class="bh-card"><div class="bh-kicker">How it sees</div><div class="bh-title">{e(name)}</div>'
        f'<div class="bh-sub">{e(text)}</div></div>' for name, text in wording.DEMO_HOW_TECH) + "</div>")
    if demo_facts.overhead:
        overhead = demo_facts.overhead
        values = {"count": overhead["count"], "cpu_mean": "{:.2f}%".format(overhead["cpu_mean_pct"]),
                  "cpu_max": "{:.2f}%".format(overhead["cpu_max_pct"]),
                  "sample_ms": "{:.1f} ms".format(overhead["sample_ms"]), "rss": "{:.0f} MB".format(overhead["rss_mb"])}
        st.html(f'<div class="bh-page-title" style="font-size:1.5rem">{e(wording.DEMO_HOW_OVERHEAD)}</div>'
                f'<div class="bh-grid-4">'
                + ui.card("CPU used by the agent", values["cpu_mean"], "of one core, on average")
                + ui.card("Worst run", values["cpu_max"], "of one core")
                + ui.card("Per measurement", values["sample_ms"], "once a second")
                + ui.card("Memory", values["rss"], "peak") + "</div>"
                f'<div class="bh-muted">{e(wording.DEMO_HOW_OVERHEAD_BODY.format(**values))}</div>')
    if demo_facts.accuracy:
        cards = "".join(
            ui.card(f"{row['workers']} workers", wording.DEMO_ACCURACY_VALUE.format(error=abs(row["error_pct"])),
                    wording.DEMO_ACCURACY_SUB.format(
                        predicted=row["predicted_ms"], actual=row["actual_ms"],
                        direction=wording.DEMO_ACCURACY_SLOW if row["error_pct"] > 0 else wording.DEMO_ACCURACY_FAST))
            for row in demo_facts.accuracy)
        st.html(f'<div class="bh-page-title" style="font-size:1.5rem">{e(wording.DEMO_HOW_ACCURACY)}</div>'
                f'<div class="bh-lead">{e(wording.DEMO_HOW_ACCURACY_BODY)}</div><div class="bh-grid-4">{cards}</div>'
                f'<div class="bh-muted">{e(wording.DEMO_HOW_ACCURACY_NOTE)}</div>')
    url = (config.get("links") or {}).get("github")
    if url:
        st.link_button(wording.DEMO_HOW_CODE, url, icon=":material/code:")

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import art, charts, data, scene, theme, wording
from mlplat.predict import PlanError, format_hours
from mlplat.run_store import RUNS_DIR_ENV, default_runs_dir
from mlplat.targets import TargetError, default_targets_path

REPO_RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
LIVE_REFRESH_S = 2
TABLE_ROW_PX = 35
TARGET_ART = {"home_kitchen": "oven_closed", "community_kitchen": "milk_bottle", "market_stall": "egg_basket",
              "rented_bakery": "baker_peel"}
RUN_FOOD = ("01_croissant", "08_loaf", "09_muffin", "03_pretzel", "11_brioche", "05_donut", "10_waffles", "04_baguette")
e = theme.escape


def runs_dir():
    configured = default_runs_dir()
    if os.environ.get(RUNS_DIR_ENV) or configured.exists():
        return str(configured)
    return str(REPO_RUNS_DIR)


def chart_theme():
    return charts.theme_for(theme.mode())


def file_signature(run_path):
    signature = []
    for name in ("meta.json", "summary.json", "resources.jsonl", "metrics.jsonl", "plan.json"):
        path = Path(run_path) / name
        stat = path.stat() if path.exists() else None
        signature.append((name, stat.st_mtime_ns if stat else 0, stat.st_size if stat else 0))
    return tuple(signature)


def targets_signature():
    path = default_targets_path()
    return (str(path), path.stat().st_mtime_ns if path.exists() else 0)


@st.cache_data(show_spinner=False)
def cached_rows(directory, signature):
    return data.run_rows(directory)


@st.cache_data(show_spinner=False)
def cached_detail(directory, run_id, signature):
    return data.load_detail(data.store(directory).get(run_id))


@st.cache_data(show_spinner=False)
def cached_plan(directory, run_id, total_steps, prefer, signature, targets):
    return data.plan_in_memory(data.store(directory).get(run_id), total_steps, prefer)


@st.cache_data(show_spinner=False)
def cached_scene_numbers(directory, run_id, signature):
    run = data.store(directory).get(run_id)
    return data.scene_numbers(run, run.read_summary() or {})


def rows():
    directory = runs_dir()
    store = data.store(directory)
    return cached_rows(directory, tuple(file_signature(run.path) for run in store.list()))


def run_for(run_id):
    return data.store(runs_dir()).get(run_id)


def detail(run_id):
    return cached_detail(runs_dir(), run_id, file_signature(run_for(run_id).path))


def started_label(row):
    return row.started.strftime("%b %d, %H:%M") if isinstance(row.started, datetime) else ""


def pick_run(label, key, all_rows, default_index=0):
    ids = [row.run_id for row in all_rows]
    labels = {row.run_id: f"{row.name}  ·  {started_label(row)}" for row in all_rows}
    wanted = st.query_params.get(key)
    index = ids.index(wanted) if wanted in ids else min(default_index, len(ids) - 1)
    chosen = st.selectbox(label, ids, index=index, format_func=labels.get, key=f"select_{key}")
    st.query_params[key] = chosen
    return chosen


def empty_state(message, art_name="oven_open"):
    st.html(f'<div class="bh-card bh-empty">{art.img(art_name, "empty oven")}'
            f'<div class="bh-title">Nothing baked yet</div><div class="bh-sub">{e(message)}</div></div>')


def card(kicker, value, sub="", term=None, tone_html=""):
    help_html = theme.tip(term) if term else ""
    return (f'<div class="bh-card"><div class="bh-kicker">{e(kicker)}{help_html}</div>'
            f'<div class="bh-big-sm">{e(value)}</div><div class="bh-muted">{e(sub)}</div>{tone_html}</div>')


def coin_html(cost_value):
    name, alt = ("golden_coin", "gold coin") if cost_value else ("silver_coin", "silver coin")
    return art.img(name, alt, "bh-coin")


def fix_card(plan, targets):
    fix = plan["recommendation"]["fix_first"]
    upgrade = fix.get("upgrade")
    now = format_hours(fix["hours_now"])
    arrow = f'<span class="bh-arrow">{theme.icon("arrow", 26)}</span>'
    fix_label = wording.FIX_OPTION_FIX.format(before=fix["current_workers"], after=fix["suggested_workers"])
    fix_gain = theme.pill("{:.0f}% faster".format(fix["gain"] * 100), "good", "up")
    fix_col = (f'<div class="bh-fix-col"><div class="bh-kicker">{e(fix_label)}{theme.tip("baker")}</div>'
               f'<div class="bh-big">{e(now)}{arrow}{e(format_hours(fix["hours_fixed"]))}</div>'
               f'<div class="bh-row">{coin_html(0)}<b>{wording.cost(0)}</b>{fix_gain}</div></div>')
    if upgrade:
        upgrade_gain = upgrade["gain"] * 100
        tone = "good" if upgrade_gain >= 15 else "warn"
        gain_pill = theme.pill("{:.0f}% faster".format(upgrade_gain), tone, "up" if upgrade_gain >= 1 else "alert")
        upgrade_label = wording.FIX_OPTION_UPGRADE.format(name=upgrade["name"])
        upgrade_col = (f'<div class="bh-fix-col"><div class="bh-kicker">{e(upgrade_label)}{theme.tip("oven")}</div>'
                       f'<div class="bh-big">{e(now)}{arrow}{e(format_hours(upgrade["total_hours"]))}</div>'
                       f'<div class="bh-row">{coin_html(upgrade["cost"])}<b>{wording.cost(upgrade["cost"])}</b>'
                       f'{gain_pill}</div></div>')
    else:
        upgrade_col = '<div class="bh-fix-col"><div class="bh-sub">No faster oven is listed in targets.yaml.</div></div>'
    still = f'<div class="bh-sub" style="margin-top:10px">{e(wording.FIX_STILL_BOUND)}</div>' if fix["still_data_bound"] else ""
    st.html(f'<div class="bh-fix"><div class="bh-row">{theme.pill(wording.FIX_TITLE, "warn", "alert")}</div>'
            f'<div class="bh-title" style="margin-top:8px">{e(wording.FIX_BODY)}</div>'
            f'<div class="bh-fix-cols">{fix_col}{upgrade_col}</div>{still}</div>')


def fit_pill(prediction, target):
    if prediction["fits_memory"] is False:
        return theme.pill(wording.CARD_NO_ROOM, "bad")
    if prediction["fits_gpu_memory"] is False:
        return theme.pill(wording.CARD_NO_OVEN_ROOM, "bad")
    if prediction["sessions"] > 1:
        return theme.pill(wording.CARD_FIT_MANY.format(sessions=prediction["sessions"]), "warn")
    if prediction["fits_weekly"] is False:
        return theme.pill(wording.CARD_FIT_WEEK.format(weeks=prediction["weeks"]), "warn")
    return theme.pill(wording.CARD_FIT_ONE, "good")


def target_card(target, entry, scenario, best_id):
    prediction = entry[scenario]
    winner = target["id"] == best_id
    badge = f'<span class="bh-best">{theme.icon("spark", 14)}{wording.BEST_BADGE}</span>' if winner else ""
    limit = wording.CARD_LIMIT_BAKERS if prediction["bottleneck"] == "bakers" else wording.CARD_LIMIT_OVEN
    before = ""
    if scenario == "fixed" and entry["as_is"]["workers"] != prediction["workers"]:
        before = f'<div class="bh-muted">Without the fix: {e(format_hours(entry["as_is"]["total_hours"]))}</div>'
    placeholder = theme.pill(wording.CARD_PLACEHOLDER, "info") if target["placeholders"] else ""
    return (f'<div class="bh-card{" bh-winner" if winner else ""}">'
            f'<div class="bh-target-art"><div><div class="bh-kicker">{e(wording.BAKERIES.get(target["bakery"], ""))}'
            f'{theme.tip("kitchen")}</div><div class="bh-title">{e(target["name"])}</div>{badge}</div>'
            f'{art.img(TARGET_ART.get(target["bakery"], "oven_closed"), "kitchen")}</div>'
            f'<div class="bh-kicker" style="margin-top:12px">{wording.CARD_TIME}</div>'
            f'<div class="bh-big">{e(format_hours(prediction["total_hours"]))}</div>'
            f'<div class="bh-muted">{e(wording.CARD_BAKERS.format(bakers=wording.bakers(prediction["workers"])))} · {e(limit)}</div>{before}'
            f'<div class="bh-row" style="margin-top:10px">{coin_html(prediction["cost"])}'
            f'<b>{e(wording.cost(prediction["cost"]))}</b></div>'
            f'<div class="bh-row" style="margin-top:10px">{fit_pill(prediction, target)}{placeholder}</div></div>')


def plan_table(plan):
    names = {target["id"]: target["name"] for target in plan["targets"]}
    rows = []
    for entry in plan["predictions"]:
        for scenario in ("as_is", "fixed"):
            prediction = entry[scenario]
            rows.append({"Kitchen": names[entry["target_id"]], "Scenario": "as it is" if scenario == "as_is" else "fixed",
                         "Bakers": prediction["workers"], "Time per tray (ms)": prediction["step_time_s"] * 1000,
                         "Full bake (h)": prediction["total_hours"], "Cost ($)": prediction["cost"],
                         "Sessions": prediction["sessions"], "Oven busy (%)": prediction["oven_busy"] * 100,
                         "Slowest part": prediction["bottleneck"], "Fits memory": prediction["fits_memory"]})
    return pd.DataFrame(rows)


def home_page():
    theme.page_title(wording.PAGE_TITLES["home"], wording.HOME_INTRO)
    all_rows = [row for row in rows() if row.verdict and row.status in ("completed", "profiled", "interrupted")]
    if not all_rows:
        empty_state(wording.HOME_EMPTY)
        return
    planned = [row for row in all_rows if data.has_plan(run_for(row.run_id))]
    choices = planned + [row for row in all_rows if row not in planned]
    left, middle, right = st.columns([1.4, 1, 1.4])
    with left:
        run_id = pick_run(wording.HOME_PICK_RUN, "run", choices)
    run = run_for(run_id)
    with middle:
        total_steps = st.number_input(wording.HOME_TOTAL_STEPS, min_value=1, step=1000,
                                      value=data.planned_total_steps(run), help=wording.HOME_TOTAL_STEPS_HELP,
                                      key=f"steps_{run_id}")
    with right:
        prefer = st.segmented_control(wording.HOME_PREFER, list(wording.HOME_PREFER_OPTIONS),
                                      format_func=wording.HOME_PREFER_OPTIONS.get, default="balanced",
                                      required=True, key="prefer")
    try:
        plan = cached_plan(runs_dir(), run_id, int(total_steps), prefer or "balanced", file_signature(run.path),
                           targets_signature())
    except (PlanError, TargetError) as error:
        st.info(wording.HOME_NO_PLAN.format(reason=error.args[0]), icon=":material/info:")
        return

    recommendation = plan["recommendation"]
    targets = {target["id"]: target for target in plan["targets"]}
    if recommendation["fix_first"]:
        fix_card(plan, targets)
    elif recommendation["best_target_id"]:
        st.html(f'<div class="bh-summary">{e(recommendation["best_reason"])}</div>')
    scenario = recommendation["scenario"]
    predictions = {entry["target_id"]: entry for entry in plan["predictions"]}
    order = recommendation["ranking"] or list(predictions)
    cards = "".join(target_card(targets[target_id], predictions[target_id], scenario, recommendation["best_target_id"])
                    for target_id in order)
    st.html(f'<div class="bh-grid">{cards}</div>')
    notes = [wording.HOME_ESTIMATE_NOTE]
    if plan["placeholders_used"]:
        notes.append(wording.HOME_PLACEHOLDER_NOTE)
    st.html(f'<div class="bh-muted">{" ".join(e(note) for note in notes)}</div>')
    if theme.technical():
        with st.expander(wording.TECH_HEADING, icon=":material/build:", expanded=True):
            profile = plan["profile"]
            st.markdown(f"**Headline:** {recommendation['headline']}")
            st.markdown(f"Measured mean step {profile['step_time_s'] * 1000:.1f} ms = compute "
                        f"{profile['compute_s'] * 1000:.1f} ms + data wait {profile['data_wait_s'] * 1000:.1f} ms "
                        f"(split from {profile['split_method']}); data prep "
                        f"{(profile['prep_core_s'] or 0) * 1000:.1f} CPU-ms per batch; overhead "
                        f"{profile['overhead_s'] * 1000:.1f} ms; startup {profile['startup_s']:.1f} s.")
            st.markdown("**Model assumptions**\n" + "\n".join(f"- {line}" for line in plan["assumptions"]))
            if plan["placeholders_used"]:
                st.markdown("**Placeholder values:** " + ", ".join(plan["placeholders_used"]))
    with st.expander(wording.HOME_TABLE, icon=":material/table:"):
        st.dataframe(plan_table(plan), hide_index=True, width="stretch",
                     column_config={column: st.column_config.NumberColumn(format="%.2f")
                                    for column in ("Time per tray (ms)", "Full bake (h)", "Cost ($)", "Oven busy (%)")})


def verdict_box(summary):
    verdict = summary["verdict"]
    title, body = wording.verdict_plain(verdict["primary"])
    label, icon, kind = wording.SEVERITY.get(verdict["severity"], wording.SEVERITY["unknown"])
    lines = [f"**{label}: {title}.** {body}", "", f"**{wording.WHAT_TO_TRY}**"]
    lines += [f"- {item}" for item in wording.PLAIN_TRY.get(verdict["primary"], [])]
    getattr(st, kind)("\n".join(lines), icon=icon)
    if theme.technical():
        technical = [f"**{verdict['title']}**"] + [f"- {line}" for line in verdict["evidence"]]
        technical += ["", "Technical suggestions:"] + [f"- {line}" for line in verdict["suggestions"]]
        for finding in verdict.get("secondary", []):
            technical.append(f"- Also: {finding['title']}")
        st.markdown("\n".join(technical))


def story_cards(summary, idle, span, busy):
    totals = summary.get("totals") or {}
    workers = (summary.get("roles") or {}).get("workers") or {}
    worker_busy = workers.get("cpu_mean_pct")
    cards = [
        card(wording.STORY_CARDS["idle"][0], wording.duration(idle) if idle is not None else "-",
             f"out of {wording.duration(span)} of baking" if span else "", "oven sat empty"),
        card(wording.STORY_CARDS["busy"][0], wording.busy(busy), wording.STORY_CARDS["busy"][1], "oven"),
        card(wording.STORY_CARDS["tray_time"][0], wording.short_duration(totals.get("step_time_median_s")),
             wording.STORY_CARDS["tray_time"][1], "baked tray"),
        card(wording.STORY_CARDS["rate"][0], f"{totals.get('steps_per_sec') or 0:.1f}", wording.STORY_CARDS["rate"][1],
             "baked tray"),
        card(wording.STORY_CARDS["bakers"][0], wording.busy(None if worker_busy is None else worker_busy / 100, theme.technical()),
             f"{wording.bakers(int(workers.get('count_median') or 0))}, each", "baker"),
        card(wording.STORY_CARDS["counter"][0], f"{(totals.get('peak_rss_mb') or 0) / 1024:.1f} GB",
             wording.STORY_CARDS["counter"][1], "counter space"),
    ]
    st.html(f'<div class="bh-grid-4">{"".join(cards)}</div>')


def story_page():
    all_rows = rows()
    if not all_rows:
        theme.page_title(wording.PAGE_TITLES["story"])
        empty_state(wording.RUNS_EMPTY)
        return
    run_id = pick_run("Which bake?", "run", all_rows)
    status = next(row.status for row in all_rows if row.run_id == run_id)
    if status == "running":
        live = st.toggle("Live refresh", value=True)
        st.fragment(run_every=LIVE_REFRESH_S if live else None)(render_story)(run_id)
    else:
        render_story(run_id)


def render_story(run_id):
    view = detail(run_id)
    meta, summary = view.meta, view.summary
    st.html(f'<div class="bh-page-title">{e(meta.get("name", run_id))}</div>'
            f'<div class="bh-row">{theme.status_pill(meta.get("status"))}'
            + (theme.severity_pill(summary["verdict"]["severity"], wording.verdict_plain(summary["verdict"]["primary"])[0])
               if summary else "") + "</div>")
    if not summary:
        st.info("This bake hasn't been looked at yet. Run: mlplat analyze " + run_id, icon=":material/info:")
        return
    idle, span, busy = data.oven_idle_seconds(summary)
    totals = summary.get("totals") or {}
    sentence = wording.story_sentence(summary["verdict"]["primary"], idle=idle, span=span, busy=busy,
                                      peak=f"{(totals.get('peak_rss_mb') or 0) / 1024:.1f} GB",
                                      limit=f"{(summary['features'].get('memory_limit_kb') or 0) / 1024 / 1024:.1f} GB")
    st.html(f'<div class="bh-summary">{e(sentence)}</div>')
    numbers = cached_scene_numbers(runs_dir(), run_id, file_signature(run_for(run_id).path))
    st.iframe(scene.scene_html(numbers, theme.tokens()), height="content")
    story_cards(summary, idle, span, busy)
    verdict_box(summary)

    names = data.metric_names(view.metrics)
    default = [name for name in ("loss",) if name in names]
    chosen = st.multiselect("Also show training scores", names, default=default)
    figure = charts.timeline_figure(view, chart_theme(), chosen, technical=theme.technical())
    if figure is not None:
        if view.window:
            st.caption("The shaded band is the part we measured. Warm-up before it is left out.")
        st.plotly_chart(figure, width="stretch", theme=None)
        with st.expander("See the numbers behind the charts", icon=":material/table:"):
            st.dataframe(view.process_series, hide_index=True, width="stretch")
            st.dataframe(view.tree, hide_index=True, width="stretch")
            if not view.gpu.empty:
                st.dataframe(view.gpu, hide_index=True, width="stretch")
            st.dataframe(view.metrics, hide_index=True, width="stretch")
    if theme.technical():
        with st.expander(wording.TECH_HEADING, icon=":material/build:"):
            if not view.processes.empty:
                st.dataframe(view.processes, hide_index=True, width="stretch")
            st.code(meta.get("command_line", ""), language="bash")
            st.json({key: meta.get(key) for key in ("config", "config_hash", "git", "host", "agent")}, expanded=False)
            st.json({"window": summary.get("window"), "data_quality": summary.get("data_quality"),
                     "thresholds": summary["verdict"].get("thresholds")}, expanded=False)


def change_pill(before, after, lower_is_better=True):
    if not before or not after:
        return ""
    ratio = before / after if lower_is_better else after / before
    if abs(ratio - 1) < 0.05:
        return theme.pill(wording.COMPARE_SAME, "info", "info")
    if ratio > 1:
        return theme.pill(wording.COMPARE_FASTER.format(factor=ratio), "good", "up")
    return theme.pill(wording.COMPARE_SLOWER.format(factor=1 / ratio), "bad", "down")


def before_after(label, before_text, after_text, pill_html, term=None):
    return (f'<div class="bh-card"><div class="bh-kicker">{e(label)}{theme.tip(term) if term else ""}</div>'
            f'<div class="bh-big">{e(before_text)}<span class="bh-arrow">{theme.icon("arrow", 26)}</span>{e(after_text)}</div>'
            f'<div class="bh-row" style="margin-top:8px">{pill_html}</div></div>')


def compare_page():
    theme.page_title(wording.PAGE_TITLES["compare"], wording.COMPARE_INTRO)
    all_rows = [row for row in rows() if row.verdict]
    if len(all_rows) < 2:
        empty_state("Compare needs two bakes. Time another one after you change something.")
        return
    left, right = st.columns(2)
    with left:
        run_a = pick_run(wording.COMPARE_BEFORE, "a", all_rows, 1)
    with right:
        run_b = pick_run(wording.COMPARE_AFTER, "b", all_rows, 0)
    view_a, view_b = detail(run_a), detail(run_b)
    sa, sb = view_a.summary, view_b.summary
    ta, tb = sa.get("totals") or {}, sb.get("totals") or {}
    _, _, busy_a = data.oven_idle_seconds(sa)
    _, _, busy_b = data.oven_idle_seconds(sb)
    cards = [
        before_after("Time per tray", wording.short_duration(ta.get("step_time_median_s")),
                     wording.short_duration(tb.get("step_time_median_s")),
                     change_pill(ta.get("step_time_median_s"), tb.get("step_time_median_s")), "baked tray"),
        before_after("Trays per second", f"{ta.get('steps_per_sec') or 0:.1f}", f"{tb.get('steps_per_sec') or 0:.1f}",
                     change_pill(ta.get("steps_per_sec"), tb.get("steps_per_sec"), lower_is_better=False)),
        before_after("Oven busy", wording.percent(busy_a), wording.percent(busy_b),
                     change_pill(busy_a, busy_b, lower_is_better=False) if busy_a and busy_b else "", "oven"),
    ]
    st.html(f'<div class="bh-ba">{"".join(cards)}</div>')
    verdicts = ""
    for label, summary in ((wording.COMPARE_BEFORE, sa), (wording.COMPARE_AFTER, sb)):
        title, _ = wording.verdict_plain(summary["verdict"]["primary"])
        verdicts += (f'<div class="bh-card"><div class="bh-kicker">{e(label)}</div>'
                     f'<div class="bh-row" style="margin-top:6px">{theme.severity_pill(summary["verdict"]["severity"], title)}</div>'
                     f'<div class="bh-sub" style="margin-top:8px">{e(wording.verdict_plain(summary["verdict"]["primary"])[1])}</div></div>')
    st.html(f'<div class="bh-ba">{verdicts}</div>')

    st.subheader(wording.COMPARE_DETAILS)
    name_a = f"Before: {view_a.meta.get('name', run_a)}"
    name_b = f"After: {view_b.meta.get('name', run_b)}"
    table = data.comparison_frame(sa, sb)
    st.dataframe(table, hide_index=True, width="stretch", height=TABLE_ROW_PX * (len(table) + 1) + 3,
                 column_config={"run A": st.column_config.NumberColumn(name_a, format="%.4g"),
                                "run B": st.column_config.NumberColumn(name_b, format="%.4g"),
                                "change %": st.column_config.NumberColumn("Change", format="%+.1f%%"),
                                "better": st.column_config.TextColumn("Better")})
    st.plotly_chart(charts.comparison_figure(view_a, view_b, name_a, name_b, chart_theme(), technical=theme.technical()),
                    width="stretch", theme=None)
    with st.expander("See the numbers behind the charts", icon=":material/table:"):
        for name, view in ((name_a, view_a), (name_b, view_b)):
            st.markdown(f"**{name}**")
            columns = [column for column in ("step", "t", "step_time_s", "step_time_rolling_s") if column in view.metrics]
            st.dataframe(view.metrics[columns], hide_index=True, width="stretch")


def runs_page():
    theme.page_title(wording.PAGE_TITLES["runs"], wording.RUNS_INTRO)
    all_rows = rows()
    if not all_rows:
        empty_state(wording.RUNS_EMPTY)
        return
    columns = st.columns(3)
    for index, row in enumerate(all_rows):
        with columns[index % 3]:
            with st.container(border=False):
                title, _ = wording.verdict_plain(row.verdict) if row.verdict else ("Not looked at yet", "")
                verdict_pill = theme.severity_pill(row.severity or "unknown", title)
                rate = f"{row.steps_per_sec:.1f} trays/s" if row.steps_per_sec else ""
                st.html(f'<div class="bh-card"><div class="bh-target-art"><div>'
                        f'<div class="bh-kicker">{e(started_label(row))} · {e(wording.short_duration(row.duration_s))}</div>'
                        f'<div class="bh-title">{e(row.name)}</div></div>'
                        f'{art.img(RUN_FOOD[index % len(RUN_FOOD)], "baked goods", "bh-food")}</div>'
                        f'<div class="bh-row">{theme.status_pill(row.status)}{verdict_pill}</div>'
                        f'<div class="bh-muted" style="margin-top:8px">{e(rate)}</div></div>')
                st.page_link("views/story.py", label=wording.RUNS_OPEN, icon=":material/arrow_forward:",
                             query_params={"run": row.run_id})
    with st.expander("See all bakes as a table", icon=":material/table:"):
        st.dataframe(pd.DataFrame({
            "Bake": [row.name for row in all_rows],
            "Started": [row.started for row in all_rows],
            "Status": [wording.STATUS.get(row.status, (row.status, None))[0] for row in all_rows],
            "What happened": [wording.verdict_plain(row.verdict)[0] if row.verdict else "-" for row in all_rows],
            "Trays per second": [row.steps_per_sec for row in all_rows],
            "Run ID": [row.run_id for row in all_rows],
        }), hide_index=True, width="stretch")

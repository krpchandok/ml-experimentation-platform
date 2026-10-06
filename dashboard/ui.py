from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard import charts, data
from mlplat.run_store import default_runs_dir

LIVE_REFRESH_S = 2
TABLE_ROW_PX = 35
SEVERITY = {
    "critical": ("error", ":material/dangerous:", "Critical", "⛔"),
    "bottleneck": ("warning", ":material/speed:", "Bottleneck", "⚠️"),
    "warning": ("warning", ":material/warning:", "Warning", "⚠️"),
    "ok": ("success", ":material/check_circle:", "Healthy", "✅"),
    "info": ("info", ":material/info:", "Info", "ℹ️"),
    "unknown": ("info", ":material/help:", "Unknown", "❔"),
}


def runs_dir():
    return str(default_runs_dir())


def theme():
    kind = getattr(st.context.theme, "type", None)
    return charts.theme_for(kind)


def file_signature(run_path):
    names = ("meta.json", "summary.json", "resources.jsonl", "metrics.jsonl")
    signature = []
    for name in names:
        path = Path(run_path) / name
        stat = path.stat() if path.exists() else None
        signature.append((name, stat.st_mtime_ns if stat else 0, stat.st_size if stat else 0))
    return tuple(signature)


@st.cache_data(show_spinner=False)
def cached_rows(directory, signature):
    return data.run_rows(directory)


@st.cache_data(show_spinner=False)
def cached_detail(directory, run_id, signature):
    return data.load_detail(data.store(directory).get(run_id))


def rows():
    directory = runs_dir()
    store = data.store(directory)
    signature = tuple(file_signature(run.path) for run in store.list())
    return cached_rows(directory, signature)


def detail(run_id):
    directory = runs_dir()
    run = data.store(directory).get(run_id)
    return cached_detail(directory, run_id, file_signature(run.path))


def verdict_label(verdict, severity):
    if not verdict:
        return "not analyzed"
    icon = SEVERITY.get(severity, SEVERITY["unknown"])[3]
    return f"{icon} {verdict.replace('_', ' ')}"


def fmt_duration(seconds):
    if seconds is None or pd.isna(seconds):
        return "-"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def run_choice_label(row):
    return f"{row.name} · {row.run_id}"


def pick_run(label, key, default_index, all_rows):
    ids = [row.run_id for row in all_rows]
    labels = {row.run_id: run_choice_label(row) for row in all_rows}
    wanted = st.query_params.get(key)
    index = ids.index(wanted) if wanted in ids else min(default_index, len(ids) - 1)
    chosen = st.selectbox(label, ids, index=index, format_func=labels.get, key=f"select_{key}")
    st.query_params[key] = chosen
    return chosen


def labeled_title(label, title):
    return title if title.lower().startswith(label.lower()) else f"{label}: {title}"


def render_verdict(verdict, compact=False):
    kind, icon, label, _ = SEVERITY.get(verdict.get("severity"), SEVERITY["unknown"])
    body = [f"**{labeled_title(label, verdict['title'])}**", ""]
    body += [f"- {line}" for line in verdict.get("evidence", [])]
    if verdict.get("suggestions") and not compact:
        body += ["", "**What to try**"]
        body += [f"- {line}" for line in verdict["suggestions"]]
    getattr(st, kind)("\n".join(body), icon=icon)


def render_secondary(verdict):
    for finding in verdict.get("secondary", []):
        kind, icon, label, _ = SEVERITY.get(finding["severity"], SEVERITY["unknown"])
        with st.expander(f"Also: {labeled_title(label, finding['title'])}", icon=icon):
            for line in finding["evidence"]:
                st.markdown(f"- {line}")


def stat_tiles(summary):
    totals = summary.get("totals", {})
    gpu = summary.get("gpu")
    wait = (summary.get("features") or {}).get("data_wait_fraction")
    tiles = [
        ("Median step", f"{totals['step_time_median_s'] * 1000:.1f} ms" if totals.get("step_time_median_s") else "-"),
        ("Steps / s", f"{totals['steps_per_sec']:.2f}" if totals.get("steps_per_sec") else "-"),
        ("Avg cores (window)", f"{totals.get('avg_cores_window') or 0:.2f}"),
        ("Peak RSS", f"{totals.get('peak_rss_mb') or 0:,.0f} MiB"),
        ("CPU-s / step", f"{totals['cpu_seconds_per_step']:.3f}" if totals.get("cpu_seconds_per_step") else "-"),
        ("GPU util", f"{gpu['util_mean_pct']:.0f}%") if gpu else
        ("Data wait", f"{wait:.0%}" if wait is not None else "-"),
    ]
    for column, (label, value) in zip(st.columns(len(tiles)), tiles):
        column.metric(label, value)


def runs_page():
    st.title("Runs")
    all_rows = rows()
    if not all_rows:
        st.info(f"No runs yet in `{runs_dir()}`. Start one with `mlplat run --name exp -- python train.py`.",
                icon=":material/info:")
        return
    frame = pd.DataFrame({
        "Run": [row.name for row in all_rows],
        "Started": [row.started for row in all_rows],
        "Duration": [fmt_duration(row.duration_s) for row in all_rows],
        "Status": [row.status for row in all_rows],
        "Final loss": [row.final_loss for row in all_rows],
        "Final accuracy": [row.final_accuracy for row in all_rows],
        "Steps/s": [row.steps_per_sec for row in all_rows],
        "Median step (ms)": [row.step_time_median_s * 1000 if row.step_time_median_s else None for row in all_rows],
        "Efficiency verdict": [verdict_label(row.verdict, row.severity) for row in all_rows],
        "Config hash": [row.config_hash for row in all_rows],
        "Run ID": [row.run_id for row in all_rows],
    })
    st.caption(f"{len(all_rows)} runs in `{runs_dir()}` · select a row to open it")
    event = st.dataframe(
        frame, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
        column_config={
            "Started": st.column_config.DatetimeColumn(format="YYYY-MM-DD HH:mm"),
            "Final loss": st.column_config.NumberColumn(format="%.4f"),
            "Final accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Steps/s": st.column_config.NumberColumn(format="%.2f"),
            "Median step (ms)": st.column_config.NumberColumn(format="%.1f"),
        },
    )
    selected = event.selection.rows if event and event.selection else []
    if selected:
        st.switch_page("pages/run_detail.py", query_params={"run": all_rows[selected[0]].run_id})


def detail_page():
    all_rows = rows()
    if not all_rows:
        st.info("No runs yet.", icon=":material/info:")
        return
    run_id = pick_run("Run", "run", 0, all_rows)
    meta_status = next(row.status for row in all_rows if row.run_id == run_id)
    if meta_status == "running":
        live = st.toggle("Live refresh", value=True, help=f"Re-read the run files every {LIVE_REFRESH_S}s")
        st.fragment(run_every=LIVE_REFRESH_S if live else None)(render_detail)(run_id)
    else:
        render_detail(run_id)


def render_detail(run_id):
    view = detail(run_id)
    meta, summary = view.meta, view.summary
    st.title(meta.get("name", run_id))
    host = meta.get("host") or {}
    git = meta.get("git") or {}
    gpus = ", ".join(gpu["name"] for gpu in host.get("gpus", [])) or "no GPU"
    st.caption(" · ".join(filter(None, [
        run_id, meta.get("status"), f"exit {meta.get('exit_code')}", fmt_duration(meta.get("duration_s")),
        f"{host.get('cpu_model', '?')} ({host.get('usable_cpus', '?')} CPUs), {gpus}",
        f"git {git['commit'][:8]}{' (dirty)' if git.get('dirty') else ''}" if git.get("commit") else None,
    ])))

    if summary:
        render_verdict(summary["verdict"])
        render_secondary(summary["verdict"])
        stat_tiles(summary)
    else:
        st.info("This run has not been analyzed yet. Run `mlplat analyze " + run_id + "`.", icon=":material/info:")

    names = data.metric_names(view.metrics)
    default = [name for name in ("loss",) if name in names]
    chosen = st.multiselect("Training metrics on the time axis", names, default=default)
    figure = charts.timeline_figure(view, theme(), chosen)
    if figure is None:
        st.info("No resource samples or metrics were recorded for this run.", icon=":material/info:")
    else:
        if view.window:
            st.caption("Shaded band: analysis window (first to last logged step); warm-up before it is excluded "
                       "from the verdict.")
        st.plotly_chart(figure, width="stretch", theme=None)
        with st.expander("Data behind the charts", icon=":material/table:"):
            st.markdown("**Per-process samples**")
            st.dataframe(view.process_series, hide_index=True, width="stretch")
            st.markdown("**Whole-run samples**")
            st.dataframe(view.tree, hide_index=True, width="stretch")
            if not view.gpu.empty:
                st.markdown("**GPU samples**")
                st.dataframe(view.gpu, hide_index=True, width="stretch")
            st.markdown("**Training metrics**")
            st.dataframe(view.metrics, hide_index=True, width="stretch")

    if not view.processes.empty:
        st.subheader("Processes")
        columns = ["pid", "role", "cmdline", "lifetime_s", "cpu_seconds", "cpu_mean_pct", "cpu_p95_pct",
                   "peak_rss_mb", "read_bytes", "write_bytes", "majflt_total", "threads_max"]
        st.dataframe(view.processes[[column for column in columns if column in view.processes]], hide_index=True,
                     width="stretch",
                     column_config={"cpu_mean_pct": st.column_config.NumberColumn("CPU mean %", format="%.0f"),
                                    "cpu_p95_pct": st.column_config.NumberColumn("CPU p95 %", format="%.0f"),
                                    "cpu_seconds": st.column_config.NumberColumn("CPU-s", format="%.1f"),
                                    "peak_rss_mb": st.column_config.NumberColumn("Peak RSS MiB", format="%.0f"),
                                    "lifetime_s": st.column_config.NumberColumn("Lifetime s", format="%.1f")})

    with st.expander("Run metadata", icon=":material/description:"):
        st.code(meta.get("command_line", ""), language="bash")
        st.json({key: meta.get(key) for key in ("config", "config_hash", "git", "host", "agent", "mlflow")},
                expanded=False)
        if summary:
            st.json({"window": summary.get("window"), "data_quality": summary.get("data_quality"),
                     "thresholds": summary["verdict"].get("thresholds")}, expanded=False)


def compare_page():
    st.title("Compare runs")
    all_rows = rows()
    if len(all_rows) < 2:
        st.info("Comparison needs at least two runs.", icon=":material/info:")
        return
    left, right = st.columns(2)
    with left:
        run_a = pick_run("Run A (baseline)", "a", 1, all_rows)
    with right:
        run_b = pick_run("Run B (candidate)", "b", 0, all_rows)
    view_a, view_b = detail(run_a), detail(run_b)
    name_a = f"A: {view_a.meta.get('name', run_a)}"
    name_b = f"B: {view_b.meta.get('name', run_b)}"

    for column, view, name in ((left, view_a, name_a), (right, view_b, name_b)):
        with column:
            st.subheader(name)
            if view.summary:
                render_verdict(view.summary["verdict"], compact=True)
            else:
                st.info("Not analyzed.", icon=":material/info:")

    if view_a.summary and view_b.summary:
        table = data.comparison_frame(view_a.summary, view_b.summary)
        st.dataframe(table, hide_index=True, width="stretch", height=TABLE_ROW_PX * (len(table) + 1) + 3,
                     column_config={"run A": st.column_config.NumberColumn(name_a, format="%.4g"),
                                    "run B": st.column_config.NumberColumn(name_b, format="%.4g"),
                                    "change %": st.column_config.NumberColumn("B vs A", format="%+.1f%%"),
                                    "better": st.column_config.TextColumn("Better")})
        if view_a.meta.get("config_hash") == view_b.meta.get("config_hash"):
            st.caption("Both runs share a config hash, so differences come from code, data or environment.")
    st.plotly_chart(charts.comparison_figure(view_a, view_b, name_a, name_b, theme()), width="stretch",
                    theme=None)
    with st.expander("Data behind the charts", icon=":material/table:"):
        for name, view in ((name_a, view_a), (name_b, view_b)):
            st.markdown(f"**{name}**")
            columns = [column for column in ("step", "t", "step_time_s", "step_time_rolling_s") if column in view.metrics]
            st.dataframe(view.metrics[columns], hide_index=True, width="stretch")

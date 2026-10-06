import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.data import OTHER_PROCESSES, ROLLING_STEPS
from dashboard.wording import CHART_TITLES, PROCESS_LABELS

FONT = 'Nunito, system-ui, -apple-system, "Segoe UI", sans-serif'
ROW_HEIGHT_PX = 190

THEMES = {
    "light": {
        "surface": "#fcfcfb", "text": "#0b0b0b", "secondary": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "window": "rgba(137,135,129,0.12)",
        "series": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    },
    "dark": {
        "surface": "#1a1a19", "text": "#ffffff", "secondary": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "window": "rgba(195,194,183,0.10)",
        "series": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
    },
}


def theme_for(kind):
    return THEMES["dark" if kind == "dark" else "light"]


def series_colors(labels, theme):
    colors = {}
    slot = 0
    for label in labels:
        if label == OTHER_PROCESSES:
            colors[label] = theme["muted"]
        else:
            colors[label] = theme["series"][slot % len(theme["series"])]
            slot += 1
    return colors


def style(figure, theme, height):
    figure.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": theme["secondary"], "size": 12},
        margin={"l": 64, "r": 190, "t": 28, "b": 40},
        hovermode="x unified",
        hoverlabel={"font": {"family": FONT}},
    )
    figure.update_xaxes(showgrid=False, linecolor=theme["axis"], ticks="outside", tickcolor=theme["axis"],
                        zeroline=False, color=theme["muted"])
    figure.update_yaxes(gridcolor=theme["grid"], gridwidth=1, zeroline=False, linecolor=theme["axis"],
                        color=theme["muted"], rangemode="tozero")
    figure.update_annotations(font={"family": FONT, "color": theme["text"], "size": 13}, xanchor="left", x=0)
    return figure


def line(x, y, name, color, legend, show_legend=True, width=2, hover_format=".1f"):
    return go.Scatter(x=x, y=y, name=name, mode="lines", line={"color": color, "width": width},
                      legend=legend, showlegend=show_legend,
                      hovertemplate=f"%{{y:{hover_format}}}<extra>{name}</extra>")


def legend_layout(index, rows, theme):
    gap = 1.0 / rows
    top = 1.0 - index * gap
    return {"x": 1.01, "y": top, "yanchor": "top", "xanchor": "left", "bgcolor": "rgba(0,0,0,0)",
            "font": {"color": theme["secondary"], "size": 11}, "tracegroupgap": 2}


def chart_title(kind, technical):
    plain, tech = CHART_TITLES[kind]
    return f"{plain}  ({tech})" if technical else plain


def process_name(label, technical):
    if technical or label == OTHER_PROCESSES:
        return label
    if label == "main":
        return PROCESS_LABELS["main"]
    return label.replace("worker", PROCESS_LABELS["worker"])


def device_name(device, technical):
    return device if technical else device.replace("GPU", "oven")


def timeline_figure(detail, theme, metric_keys, technical=True):
    panels = []
    series = detail.process_series
    if not series.empty:
        panels.append(("cpu", chart_title("cpu", technical)))
        panels.append(("memory", chart_title("memory", technical)))
    if not detail.tree.empty:
        panels.append(("io", chart_title("io", technical)))
    if not detail.gpu.empty:
        panels.append(("gpu", chart_title("gpu", technical)))
        if detail.gpu["mem_used_mib"].notna().any():
            panels.append(("gpu_memory", chart_title("gpu_memory", technical)))
    for key in metric_keys:
        if key in detail.metrics:
            panels.append((f"metric:{key}", key))
    if "step_time_s" in detail.metrics and detail.metrics["step_time_s"].notna().any():
        panels.append(("step_time", chart_title("step_time", technical)))
    if not panels:
        return None

    rows = len(panels)
    figure = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.32 / rows,
                           subplot_titles=[title for _, title in panels])
    legends = {}
    labels = list(series["label"].cat.categories) if not series.empty else []
    process_colors = series_colors(labels, theme)

    for row, (kind, _) in enumerate(panels, start=1):
        legend = "legend" if row == 1 else f"legend{row}"
        legends[legend] = legend_layout(row - 1, rows, theme)
        if kind == "cpu":
            for label in labels:
                part = series[series["label"] == label]
                if not part.empty:
                    figure.add_trace(line(part["t"], part["cpu_pct"], process_name(label, technical), process_colors[label], legend,
                                          hover_format=".0f"), row=row, col=1)
        elif kind == "memory":
            figure.add_trace(line(detail.tree["t"], detail.tree["rss_mib"], "all processes (RSS sum)" if technical else "everyone together",
                                  theme["secondary"], legend, hover_format=".0f"), row=row, col=1)
            main = series[series["label"] == "main"]
            if not main.empty:
                figure.add_trace(line(main["t"], main["rss_mib"], process_name("main", technical), process_colors["main"], legend,
                                      hover_format=".0f"), row=row, col=1)
        elif kind == "io":
            figure.add_trace(line(detail.tree["t"], detail.tree["read_mib_s"], "read", theme["series"][0], legend),
                             row=row, col=1)
            figure.add_trace(line(detail.tree["t"], detail.tree["write_mib_s"], "write", theme["series"][1],
                                  legend), row=row, col=1)
        elif kind == "gpu":
            for index, (device, part) in enumerate(detail.gpu.groupby("device", sort=True)):
                figure.add_trace(line(part["t"], part["util_pct"], device_name(device, technical), theme["series"][index % 8], legend,
                                      hover_format=".0f"), row=row, col=1)
            figure.update_yaxes(range=[0, 100], row=row, col=1)
        elif kind == "gpu_memory":
            for index, (device, part) in enumerate(detail.gpu.groupby("device", sort=True)):
                figure.add_trace(line(part["t"], part["mem_used_mib"], device_name(device, technical), theme["series"][index % 8], legend,
                                      hover_format=".0f"), row=row, col=1)
        elif kind.startswith("metric:"):
            key = kind.split(":", 1)[1]
            part = detail.metrics[["t", key]].dropna()
            figure.add_trace(line(part["t"], part[key], key, theme["series"][0], legend, show_legend=False,
                                  hover_format=".4g"), row=row, col=1)
        elif kind == "step_time":
            frame = detail.metrics
            figure.add_trace(go.Scatter(x=frame["t"], y=frame["step_time_s"] * 1000, name="per logged step" if technical else "each tray",
                                        mode="markers", marker={"color": theme["series"][0], "size": 4,
                                                                "opacity": 0.35},
                                        legend=legend, hovertemplate="%{y:.1f} ms<extra>step</extra>"),
                             row=row, col=1)
            figure.add_trace(line(frame["t"], frame["step_time_rolling_s"] * 1000,
                                  f"rolling median ({ROLLING_STEPS})" if technical else "typical tray", theme["series"][0], legend,
                                  hover_format=".1f"), row=row, col=1)

    if detail.window is not None:
        for row in range(1, rows + 1):
            figure.add_vrect(x0=detail.window[0], x1=detail.window[1], fillcolor=theme["window"], line_width=0,
                             layer="below", row=row, col=1)
    figure.update_xaxes(title_text="seconds since run start" if technical else "seconds since the bake started", row=rows, col=1)
    figure.update_layout(**legends)
    return style(figure, theme, ROW_HEIGHT_PX * rows + 60)


def comparison_figure(detail_a, detail_b, name_a, name_b, theme, technical=True):
    titles = (["Step time (ms, rolling median) by step",
               "CPU used by the whole run (cores), seconds since first logged step"] if technical else
              ["Time per tray, in milliseconds", "Work stations in use, from the first tray onward"])
    figure = make_subplots(rows=2, cols=1, vertical_spacing=0.18, subplot_titles=titles)
    for detail, name, color in ((detail_a, name_a, theme["series"][0]), (detail_b, name_b, theme["series"][1])):
        metrics = detail.metrics
        if "step_time_rolling_s" in metrics and "step" in metrics:
            figure.add_trace(line(metrics["step"], metrics["step_time_rolling_s"] * 1000, name, color, "legend"),
                             row=1, col=1)
        start = detail.window[0] if detail.window else 0.0
        tree = detail.tree
        figure.add_trace(line(tree["t"] - start, tree["cpu_cores"], name, color, "legend", show_legend=False,
                              hover_format=".2f"), row=2, col=1)
    figure.update_xaxes(title_text="step" if technical else "tray number", row=1, col=1)
    figure.update_xaxes(title_text="seconds since first logged step" if technical else "seconds", row=2, col=1)
    figure.update_layout(legend={"orientation": "h", "x": 0, "y": 1.08, "yanchor": "bottom",
                                 "bgcolor": "rgba(0,0,0,0)"})
    style(figure, theme, 620)
    figure.update_layout(margin={"r": 24, "t": 70})
    return figure

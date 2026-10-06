import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from mlplat.analyzer import analyze_run, format_report
from mlplat.launcher import launch
from mlplat.run_store import RUNS_DIR_ENV, RunStore, default_runs_dir


def build_parser():
    parser = argparse.ArgumentParser(prog="mlplat")
    commands = parser.add_subparsers(dest="command_name", required=True)

    run = commands.add_parser("run", help="run a training command with resource profiling")
    run.add_argument("--name", required=True)
    run.add_argument("--config")
    run.add_argument("--runs-dir")
    run.add_argument("--agent", dest="agent_path")
    run.add_argument("--interval-ms", type=int, default=1000)
    run.add_argument("--mlflow", choices=["auto", "on", "off"], default="auto")
    run.add_argument("command", nargs=argparse.REMAINDER)

    run.add_argument("--no-analyze", action="store_true")

    listing = commands.add_parser("list", help="list runs in the run store")
    listing.add_argument("--runs-dir")

    analyze = commands.add_parser("analyze", help="summarize a run and diagnose bottlenecks")
    analyze.add_argument("run_id", nargs="?", default="latest")
    analyze.add_argument("--runs-dir")
    analyze.add_argument("--json", action="store_true")

    dashboard = commands.add_parser("dashboard", help="open the Streamlit efficiency dashboard")
    dashboard.add_argument("--runs-dir")
    dashboard.add_argument("--port", type=int, default=8501)
    return parser


def dashboard_command(runs_dir, port):
    app = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"
    if not app.exists():
        print(f"mlplat dashboard: {app} not found (the dashboard ships with the repository checkout)",
              file=sys.stderr)
        return 1
    env = dict(os.environ)
    env[RUNS_DIR_ENV] = str(Path(runs_dir).resolve()) if runs_dir else str(default_runs_dir().resolve())
    command = [sys.executable, "-m", "streamlit", "run", str(app), "--server.port", str(port),
               "--server.headless", "true", "--browser.gatherUsageStats", "false"]
    try:
        return subprocess.call(command, env=env)
    except KeyboardInterrupt:
        return 130


def format_duration(seconds):
    if seconds is None:
        return "-"
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"


def list_runs(runs_dir):
    rows = [("RUN ID", "NAME", "STATUS", "DURATION", "EXIT", "CONFIG HASH", "VERDICT")]
    for run in RunStore(runs_dir).list():
        meta = run.read_meta()
        summary = run.read_summary() or {}
        rows.append((run.run_id, meta.get("name", ""), meta.get("status", ""), format_duration(meta.get("duration_s")),
                     str(meta.get("exit_code", "")), meta.get("config_hash", ""),
                     summary.get("verdict", {}).get("primary", "-")))
    widths = [max(len(row[column]) for row in rows) for column in range(len(rows[0]))]
    for row in rows:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip())
    return 0


def analyze_command(run_id, runs_dir, as_json):
    store = RunStore(runs_dir)
    try:
        runs = store.list()
        run = runs[-1] if run_id == "latest" and runs else store.get(run_id)
    except KeyError as error:
        print(f"mlplat analyze: {error.args[0]}", file=sys.stderr)
        return 1
    summary = analyze_run(run)
    print(json.dumps(summary, indent=2, default=str) if as_json else format_report(summary))
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command_name == "list":
        return list_runs(args.runs_dir)
    if args.command_name == "analyze":
        return analyze_command(args.run_id, args.runs_dir, args.json)
    if args.command_name == "dashboard":
        return dashboard_command(args.runs_dir, args.port)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        print("mlplat run: missing training command after --", file=sys.stderr)
        return 2
    if args.interval_ms < 10:
        print("mlplat run: --interval-ms must be at least 10", file=sys.stderr)
        return 2
    return launch(args.name, command, config_path=args.config, runs_dir=args.runs_dir, agent_path=args.agent_path,
                  interval_ms=args.interval_ms, mlflow_mode=args.mlflow, analyze=not args.no_analyze)


if __name__ == "__main__":
    sys.exit(main())

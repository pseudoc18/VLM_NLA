#!/usr/bin/env python3
"""Initialize a traceable VLM-NLA experiment run directory.

The script intentionally stays lightweight: it creates the run folder, copies a
config and run-record template, snapshots the current Git state, and records the
local software/hardware environment. Training and evaluation commands should be
appended to command_log.txt by the person or agent running the experiment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import platform
import socket
import subprocess
import sys
from pathlib import Path


TRACKED_PACKAGES = [
    "numpy",
    "pillow",
    "pyarrow",
    "pyyaml",
    "torch",
    "transformers",
    "peft",
    "matplotlib",
]


def run_command(args: list[str], cwd: Path) -> dict[str, object]:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {"returncode": None, "stdout": "", "stderr": str(exc)}


def command_stdout(args: list[str], cwd: Path) -> str:
    result = run_command(args, cwd)
    return str(result["stdout"]) if result["returncode"] == 0 else ""


def package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def replace_yaml_scalar(text: str, key: str, value: str) -> str:
    lines = []
    replaced = False
    for line in text.splitlines():
        if line.startswith(f"{key}:"):
            lines.append(f'{key}: "{value}"')
            replaced = True
        else:
            lines.append(line)
    if not replaced:
        lines.insert(0, f'{key}: "{value}"')
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--config", default="experiments/templates/config_template.yaml")
    parser.add_argument("--template", default="experiments/templates/run_record_template.md")
    parser.add_argument("--run-root", default="experiments/runs")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--study", default="")
    parser.add_argument("--status", default="planned")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo = Path(command_stdout(["git", "rev-parse", "--show-toplevel"], Path.cwd()) or Path.cwd()).resolve()
    run_dir = (repo / args.run_root / args.run_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.force:
        raise SystemExit(f"{run_dir} already exists; use --force to refresh template files")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "qualitative_panels").mkdir(exist_ok=True)

    config_src = (repo / args.config).resolve()
    template_src = (repo / args.template).resolve()
    if not config_src.exists():
        raise SystemExit(f"missing config template: {config_src}")
    if not template_src.exists():
        raise SystemExit(f"missing run record template: {template_src}")

    config_text = config_src.read_text(encoding="utf-8")
    replacements = {
        "run_id": args.run_id,
        "status": args.status,
    }
    if args.hypothesis:
        replacements["hypothesis"] = args.hypothesis
    if args.study:
        replacements["study"] = args.study
    for key, value in replacements.items():
        config_text = replace_yaml_scalar(config_text, key, value)
    config_text = config_text.replace("${run_id}", args.run_id)
    (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")

    run_record = template_src.read_text(encoding="utf-8").replace("{run_id}", args.run_id)
    (run_dir / "run_record.md").write_text(run_record, encoding="utf-8")
    (run_dir / "command_log.txt").write_text(
        f"# Commands for {args.run_id}\n# Append exact shell commands and important stdout/stderr paths here.\n",
        encoding="utf-8",
    )

    git_diff = run_command(["git", "diff", "--binary"], repo)
    (run_dir / "git_diff.patch").write_text(str(git_diff["stdout"]) + "\n", encoding="utf-8")

    environment = {
        "run_id": args.run_id,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "repo": str(repo),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "executable": sys.executable,
            "version": sys.version,
        },
        "git": {
            "commit": command_stdout(["git", "rev-parse", "HEAD"], repo),
            "branch": command_stdout(["git", "branch", "--show-current"], repo),
            "status_short": command_stdout(["git", "status", "--short"], repo),
            "remote": command_stdout(["git", "remote", "-v"], repo),
        },
        "packages": package_versions(),
        "cuda": {
            "nvidia_smi": run_command(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,driver_version",
                    "--format=csv,noheader",
                ],
                repo,
            ),
        },
    }
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    for name in [
        "train_summary.json",
        "sensitivity.json",
        "ranking.json",
        "semantic_eval.json",
        "failure_cases.json",
    ]:
        path = run_dir / name
        if not path.exists():
            path.write_text("{}\n", encoding="utf-8")

    print(json.dumps({"run_id": args.run_id, "run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()

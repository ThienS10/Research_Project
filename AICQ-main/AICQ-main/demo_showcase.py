#!/usr/bin/env python3
"""
Automated end-to-end demo runner for AIQA.

Execute this script to:
  1. Run the LLM-driven moisture classification + physics/GPR pipeline for both options.
  2. Surface the key artifacts (metrics, classifications, predictions, explainability log).
  3. Provide concise console output that is easy to capture in a short demo video.

Usage:
    python demo_showcase.py
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple

from aws_bedrock_agent.pipeline_runner import run_from_csv_with_text
from config import get_config


PROJECT_ROOT = Path(__file__).resolve().parent


def _print_banner(text: str) -> None:
    line = "=" * len(text)
    print(f"\n{line}\n{text}\n{line}")


def _print_subheader(text: str) -> None:
    print(f"\n{text}\n" + "-" * len(text))


def _read_text_file(path: Path, limit: int | None = None) -> Iterable[str]:
    try:
        with path.open("r") as handle:
            if limit is None:
                yield from (line.rstrip("\n") for line in handle)
            else:
                for idx, line in enumerate(handle):
                    if idx >= limit:
                        break
                    yield line.rstrip("\n")
    except FileNotFoundError:
        yield f"[missing] {path}"


def _preview_csv_rows(path: Path, limit: int = 5) -> Tuple[int, list[Tuple[str, ...]]]:
    rows: list[Tuple[str, ...]] = []
    total = 0
    try:
        with path.open("r", newline="") as handle:
            reader = csv.reader(handle)
            for idx, row in enumerate(reader):
                total += 1
                if idx <= limit:
                    rows.append(tuple(row))
    except FileNotFoundError:
        return 0, []
    return total, rows


def _load_xai_steps(path: Path, limit: int = 5) -> list[str]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        return [f"[missing] {path}"]
    steps = payload.get("steps", [])
    titles = [f"{idx + 1}. {step.get('title', 'Untitled')}" for idx, step in enumerate(steps)]
    return titles[:limit]


def _summarize_predictions(path: Path) -> Dict[str, float]:
    try:
        with path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            diffs = []
            for row in reader:
                try:
                    actual = float(row["Actual_NDG_Density"])
                    predicted = float(row["Predicted_Density"])
                    diffs.append(predicted - actual)
                except (KeyError, ValueError):
                    continue
    except FileNotFoundError:
        return {}

    if not diffs:
        return {}
    count = len(diffs)
    mae = sum(abs(d) for d in diffs) / count
    bias = sum(diffs) / count
    return {"count": count, "mae": mae, "bias": bias}


def _ensure_env_defaults() -> None:
    os.environ.setdefault("AWS_REGION", "us-east-1")


def _run_option(option_name: str, csv_path: Path, json_path: Path) -> Dict[str, Path]:
    _print_subheader(f"Running pipeline: {option_name}")
    start = time.time()
    run_from_csv_with_text(
        option_name=option_name,
        csv_in_path=str(csv_path),
        csv_copy_out_path=str(csv_path),  # in-place copy for demo simplicity
        json_out_path=str(json_path),
    )
    elapsed = time.time() - start
    print(f"Completed {option_name} in {elapsed:0.1f} seconds.")

    artifacts = {
        "classification_log": PROJECT_ROOT / "data" / f"moisture_classification_{option_name}.csv",
        "final_metrics": PROJECT_ROOT / f"final_results_{option_name}.txt",
        "prediction_vs_ndg": PROJECT_ROOT / f"prediction_vs_ndg_{option_name}.csv",
        "xai_report": PROJECT_ROOT / "data" / f"xai_report_{option_name}.json",
        "xai_report_md": PROJECT_ROOT / "data" / f"xai_report_{option_name}.md",
    }
    return artifacts


def _show_artifacts(option_name: str, artifacts: Dict[str, Path]) -> None:
    _print_subheader(f"Key outputs for {option_name}")

    final_metrics = artifacts["final_metrics"]
    print("Holdout metrics:")
    for line in _read_text_file(final_metrics):
        print(f"  {line}")

    pred_summary = _summarize_predictions(artifacts["prediction_vs_ndg"])
    if pred_summary:
        print(
            "Prediction summary: "
            f"{pred_summary['count']} rows, "
            f"MAE={pred_summary['mae']:.4f}, "
            f"bias={pred_summary['bias']:.4f}"
        )
    else:
        print("Prediction summary: [unavailable]")

    log_total, log_rows = _preview_csv_rows(artifacts["classification_log"])
    if log_rows:
        header, *samples = log_rows
        print(f"Classification log preview (total rows: {log_total - 1}):")
        print("  " + ", ".join(header))
        for sample in samples[:5]:
            print("  " + ", ".join(sample))
    else:
        print("Classification log preview: [missing]")

    xai_titles = _load_xai_steps(artifacts["xai_report"])
    print("Explainability steps (first few):")
    for title in xai_titles:
        print(f"  {title}")

    print("Artifact locations:")
    for label, path in artifacts.items():
        print(f"  {label}: {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    _ensure_env_defaults()

    cfg = get_config()
    options = [
        ("option1_llm", Path(cfg["data"]["csv_option1"]), Path(cfg["data"]["json_option1"])),
        ("option2_llm", Path(cfg["data"]["csv_option2"]), Path(cfg["data"]["json_option2"])),
    ]

    _print_banner("AIQA Automated Showcase")
    print("This scripted run performs both pipeline options and highlights the generated artifacts.\n")

    all_artifacts: Dict[str, Dict[str, Path]] = {}
    for option_name, csv_path, json_path in options:
        artifacts = _run_option(option_name, csv_path, json_path)
        all_artifacts[option_name] = artifacts

    for option_name, artifacts in all_artifacts.items():
        _show_artifacts(option_name, artifacts)

    _print_banner("Demo complete")
    print("Next step: capture this terminal output alongside the generated plots for your submission video.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nDemo interrupted by user.")

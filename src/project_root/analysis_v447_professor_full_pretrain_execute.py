"""V447 executed professor full/medium external pretraining.

This runner launches the professor-requested V447 configs through the V431
sequence pretraining primitive. It exports representation/probability artifacts
only; no submission CSVs are produced.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_v431_external_sequence_model_zoo import FORBIDDEN_COLUMNS, ModelConfig, train_one_pretrain_model
from analysis_v441_full_external_pretrain_runner import load_professor_pretrain_sequences


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v447_professor_full_pretrain_execute"
SMOKE_MAX_WINDOWS = 2048
SMOKE_MODELS = ("gru_small_exec", "lstm_small_exec")
SMALL_MODELS = ("gru_small_exec", "lstm_small_exec", "transformer_small_exec")
MEDIUM_MODELS = ("gru_medium_exec", "lstm_medium_exec")
MODE_DEFAULTS = {
    "smoke": SMOKE_MODELS,
    "small": SMALL_MODELS,
    "medium": MEDIUM_MODELS,
    "all": SMALL_MODELS + MEDIUM_MODELS,
}


@dataclass(frozen=True)
class FullPretrainConfig:
    name: str
    base_model: str
    max_windows: int
    epochs: int
    dropout: float
    mask_probability: float
    embedding_dim: int
    hidden_dim: int
    layers: int = 1
    batch_size: int = 32
    context_events: int = 8
    learning_rate: float = 1e-3
    seed: int = 447
    device: str = "cpu"


def build_v447_execution_grid() -> dict[str, FullPretrainConfig]:
    """Return the executable V447 small and medium external-pretrain grid."""

    return {
        "gru_small_exec": FullPretrainConfig("gru_small_exec", "gru", 12_000, 2, 0.15, 0.20, 32, 64),
        "lstm_small_exec": FullPretrainConfig("lstm_small_exec", "lstm", 12_000, 2, 0.15, 0.20, 32, 64),
        "transformer_small_exec": FullPretrainConfig(
            "transformer_small_exec",
            "transformer",
            12_000,
            2,
            0.15,
            0.20,
            32,
            64,
            layers=2,
        ),
        "gru_medium_exec": FullPretrainConfig("gru_medium_exec", "gru", 30_000, 3, 0.20, 0.25, 64, 128, layers=2),
        "lstm_medium_exec": FullPretrainConfig(
            "lstm_medium_exec",
            "lstm",
            30_000,
            3,
            0.20,
            0.25,
            64,
            128,
            layers=2,
        ),
    }


def _parse_models(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def select_v447_configs(
    grid: dict[str, FullPretrainConfig],
    *,
    mode: str,
    models: list[str] | None = None,
) -> list[FullPretrainConfig]:
    """Select V447 configs for smoke/small/medium/all modes."""

    if mode not in MODE_DEFAULTS:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(MODE_DEFAULTS)}")
    names = models or list(MODE_DEFAULTS[mode])
    unknown = [name for name in names if name not in grid]
    if unknown:
        raise ValueError(f"Unknown V447 model names: {unknown}")

    selected = [grid[name] for name in names]
    if mode == "smoke":
        selected = [
            replace(
                cfg,
                max_windows=min(cfg.max_windows, SMOKE_MAX_WINDOWS),
                epochs=1,
                batch_size=min(cfg.batch_size, 32),
                seed=447,
            )
            for cfg in selected
        ]
    return selected


def _to_v431_model_config(config: FullPretrainConfig) -> ModelConfig:
    return ModelConfig(
        name=config.name,
        model_type=config.base_model,
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        layers=config.layers,
        dropout=config.dropout,
        epochs=config.epochs,
        max_windows=config.max_windows,
        batch_size=config.batch_size,
        context_events=config.context_events,
        mask_probability=config.mask_probability,
        learning_rate=config.learning_rate,
        seed=config.seed,
        device=config.device,
    )


def _cap_sequences_for_mode(seq: pd.DataFrame, *, mode: str, max_windows: int) -> pd.DataFrame:
    if mode != "smoke" or seq.empty:
        return seq.reset_index(drop=True)

    ordered = seq.sort_values(["source_dataset", "sequence_id", "event_index"]).reset_index(drop=True)
    selected: list[pd.DataFrame] = []
    estimated_windows = 0
    for _, group in ordered.groupby(["source_dataset", "sequence_id"], sort=True):
        if len(group) < 2:
            continue
        selected.append(group)
        estimated_windows += max(len(group) - 1, 0)
        if estimated_windows >= max_windows:
            break
    if not selected:
        return ordered.head(max_windows + 1).reset_index(drop=True)
    return pd.concat(selected, ignore_index=True)


def _report_row_from_error(config: FullPretrainConfig, message: str) -> dict[str, Any]:
    return {
        "model_name": config.name,
        "model_type": config.base_model,
        "embedding_dim": config.embedding_dim,
        "hidden_dim": config.hidden_dim,
        "layers": config.layers,
        "dropout": config.dropout,
        "mask_probability": config.mask_probability,
        "epochs": config.epochs,
        "max_windows": config.max_windows,
        "train_windows": None,
        "vocab_size": None,
        "final_loss": None,
        "status": "error",
        "message": message,
    }


def _report_row_from_result(report: dict[str, Any]) -> dict[str, Any]:
    final_epoch = report["epoch_reports"][-1] if report.get("epoch_reports") else {}
    config = report["config"]
    return {
        "model_name": report["model_name"],
        "model_type": report["model_type"],
        "embedding_dim": config["embedding_dim"],
        "hidden_dim": config["hidden_dim"],
        "layers": config["layers"],
        "dropout": report["dropout"],
        "mask_probability": report["mask_probability"],
        "epochs": config["epochs"],
        "max_windows": config["max_windows"],
        "train_windows": report["train_windows"],
        "vocab_size": report["vocab_size"],
        "final_loss": final_epoch.get("loss"),
        "status": "trained",
        "message": "",
    }


def _csv_forbidden_column_hits(outdir: Path) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {}
    for path in sorted(outdir.glob("*.csv")):
        columns = pd.read_csv(path, nrows=0).columns
        overlap = sorted(FORBIDDEN_COLUMNS & set(columns))
        if overlap:
            hits[path.name] = overlap
    return hits


def _submission_exports(outdir: Path) -> list[str]:
    return sorted(path.name for path in outdir.glob("submission*.csv"))


def run_v447_pretraining(
    *,
    mode: str,
    models: list[str] | None = None,
    root: Path | str = ROOT,
    outdir: Path | str = OUTDIR,
) -> dict[str, Any]:
    """Execute selected V447 pretraining configs and write reports."""

    grid = build_v447_execution_grid()
    configs = select_v447_configs(grid, mode=mode, models=models)
    max_selected_windows = max((config.max_windows for config in configs), default=0)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    seq, input_report = load_professor_pretrain_sequences(root=root)
    train_seq = _cap_sequences_for_mode(seq, mode=mode, max_windows=max_selected_windows)

    reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    if input_report["status"] == "no_input":
        errors.extend(_report_row_from_error(config, input_report["message"]) for config in configs)
    else:
        for config in configs:
            try:
                result = train_one_pretrain_model(train_seq, config=_to_v431_model_config(config), outdir=outdir)
                reports.append(result["report"])
            except Exception as exc:  # pragma: no cover - protects long sweeps from aborting all configs
                errors.append(_report_row_from_error(config, str(exc)))

    report_rows = [_report_row_from_result(report) for report in reports] + errors
    model_report_path = outdir / "v447_model_reports.csv"
    pd.DataFrame(report_rows).to_csv(model_report_path, index=False)

    forbidden_hits = _csv_forbidden_column_hits(outdir)
    submissions = _submission_exports(outdir)
    status = "complete" if reports and not errors and not forbidden_hits and not submissions else ("partial" if reports else "error")
    summary = {
        "version": "V447",
        "status": status,
        "mode": mode,
        "input_report": input_report,
        "training_rows_used": int(len(train_seq)),
        "source_counts_used": train_seq.groupby("source_dataset").size().to_dict() if not train_seq.empty else {},
        "models_run": [report["model_name"] for report in reports],
        "models_planned": [config.name for config in configs],
        "model_configs": [asdict(config) for config in configs],
        "model_configs_run": [report["config"] for report in reports],
        "errors": errors,
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "forbidden_exact_column_exports": forbidden_hits,
        "submission_exports": len(submissions),
        "submission_export_files": submissions,
        "outputs": {
            "model_reports": str(model_report_path),
            "pretrain_summary": str(outdir / "v447_pretrain_summary.json"),
        },
    }
    summary_path = outdir / "v447_pretrain_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute V447 professor external pretraining configs.")
    parser.add_argument("--mode", choices=sorted(MODE_DEFAULTS), default="smoke")
    parser.add_argument("--models", default=None, help="Comma-separated V447 model names.")
    parser.add_argument("--root", default=str(ROOT), help="Workspace root containing V440/V430 input directories.")
    parser.add_argument("--outdir", default=str(OUTDIR), help="Output directory.")
    args = parser.parse_args()

    summary = run_v447_pretraining(
        mode=args.mode,
        models=_parse_models(args.models),
        root=Path(args.root),
        outdir=Path(args.outdir),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

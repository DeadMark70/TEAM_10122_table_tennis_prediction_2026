"""V441 full external pretraining runner.

Coordinates professor-requested external pretraining configurations while
delegating the actual bounded sequence modeling to V431 primitives. Quick mode
trains a small capped run; medium/full modes record runnable configs by default
so large jobs are not launched accidentally.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_v431_external_sequence_model_zoo import (
    FORBIDDEN_COLUMNS,
    ModelConfig,
    _coerce_pretrain_schema,
    run_pipeline,
    train_one_pretrain_model,
)


ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT / "v441_full_external_pretrain_runner"
V440_INPUT = ROOT / "v440_professor_corpus_weighting" / "v440_weighted_external_events.csv"
V430_INPUT = ROOT / "v430_external_audit_canonical_expander" / "canonical_expanded_events.csv"

MODE_DEFAULTS = {
    "quick": ["gru_small_full", "lstm_small_full", "transformer_small_full"],
    "full": ["gru_small_full", "lstm_small_full", "transformer_small_full"],
    "medium": ["gru_medium_full", "lstm_medium_full", "transformer_medium_full"],
}
QUICK_ROW_CAP = 900
QUICK_MAX_WINDOWS = 256


@dataclass(frozen=True)
class PretrainRunConfig:
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
    seed: int = 441
    device: str = "cpu"


def build_professor_model_grid() -> dict[str, PretrainRunConfig]:
    """Return runnable V441 professor pretraining configs."""

    return {
        "gru_small_full": PretrainRunConfig("gru_small_full", "gru", 50_000, 3, 0.10, 0.15, 32, 64),
        "lstm_small_full": PretrainRunConfig("lstm_small_full", "lstm", 50_000, 3, 0.10, 0.15, 32, 64),
        "transformer_small_full": PretrainRunConfig(
            "transformer_small_full", "transformer", 50_000, 3, 0.10, 0.15, 32, 64, layers=2
        ),
        "gru_medium_full": PretrainRunConfig("gru_medium_full", "gru", 100_000, 4, 0.15, 0.20, 64, 128, layers=2),
        "lstm_medium_full": PretrainRunConfig("lstm_medium_full", "lstm", 100_000, 4, 0.15, 0.20, 64, 128, layers=2),
        "transformer_medium_full": PretrainRunConfig(
            "transformer_medium_full", "transformer", 100_000, 4, 0.15, 0.20, 64, 128, layers=3
        ),
    }


def _parse_models(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def select_configs_for_mode(
    grid: dict[str, PretrainRunConfig],
    *,
    mode: str,
    models: list[str] | None = None,
) -> list[PretrainRunConfig]:
    """Select configs for quick/full/medium mode.

    Quick mode keeps the small professor architecture names but caps windows and
    epochs so the script is suitable for a smoke run.
    """

    if mode not in MODE_DEFAULTS:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {sorted(MODE_DEFAULTS)}")
    names = models or MODE_DEFAULTS[mode]
    unknown = [name for name in names if name not in grid]
    if unknown:
        raise ValueError(f"Unknown V441 model names: {unknown}")
    selected = [grid[name] for name in names]
    if mode == "quick":
        return [
            replace(
                cfg,
                epochs=1,
                max_windows=min(cfg.max_windows, QUICK_MAX_WINDOWS),
                batch_size=min(cfg.batch_size, 32),
                seed=441,
            )
            for cfg in selected
        ]
    return selected


def load_professor_pretrain_sequences(*, root: Path | str = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load V440 weighted events first, then V430 canonical events.

    The returned frame is coerced through the V431 pretraining schema, which
    strips exact AICUP labels before any training code sees the data.
    """

    root = Path(root)
    candidates = [
        (root / "v440_professor_corpus_weighting" / "v440_weighted_external_events.csv", "v440_weighted_external_events"),
        (root / "v430_external_audit_canonical_expander" / "canonical_expanded_events.csv", "v430_canonical_expanded_events"),
    ]
    checked_paths: list[str] = []
    for path, input_kind in candidates:
        checked_paths.append(str(path))
        if not path.exists():
            continue
        raw = pd.read_csv(path, low_memory=False)
        clean = _coerce_pretrain_schema(raw)
        return clean, {
            "status": "loaded",
            "input_path": str(path),
            "input_kind": input_kind,
            "input_rows": int(len(raw)),
            "usable_rows": int(len(clean)),
            "stripped_forbidden_columns": sorted(FORBIDDEN_COLUMNS & set(raw.columns)),
        }
    empty = _coerce_pretrain_schema(pd.DataFrame())
    return empty, {
        "status": "no_input",
        "message": "No V440 weighted external events or V430 canonical expanded events input was found.",
        "checked_paths": checked_paths,
    }


def _cap_quick_sequences(seq: pd.DataFrame, *, row_cap: int = QUICK_ROW_CAP) -> pd.DataFrame:
    if len(seq) <= row_cap:
        return seq.reset_index(drop=True)
    ordered = seq.sort_values(["source_dataset", "sequence_id", "event_index"]).reset_index(drop=True)
    groups_by_source: dict[str, list[pd.DataFrame]] = {}
    for (source, _), group in ordered.groupby(["source_dataset", "sequence_id"], sort=True):
        if len(group) >= 2:
            groups_by_source.setdefault(str(source), []).append(group)
    if not groups_by_source:
        return ordered.head(row_cap).reset_index(drop=True)

    selected: list[pd.DataFrame] = []
    selected_rows = 0
    source_names = sorted(groups_by_source)
    positions = {source: 0 for source in source_names}
    while selected_rows < row_cap:
        added = False
        for source in source_names:
            source_groups = groups_by_source[source]
            position = positions[source]
            if position >= len(source_groups):
                continue
            group = source_groups[position]
            if selected_rows + len(group) > row_cap:
                continue
            selected.append(group)
            selected_rows += len(group)
            positions[source] += 1
            added = True
            if selected_rows >= row_cap:
                break
        if not added:
            break
    if not selected:
        return ordered.head(row_cap).reset_index(drop=True)
    return pd.concat(selected, ignore_index=True)


def _to_v431_model_config(config: PretrainRunConfig) -> ModelConfig:
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


def _report_row_from_config(config: PretrainRunConfig, *, status: str, message: str = "") -> dict[str, Any]:
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
        "status": status,
        "message": message,
    }


def write_planned_run_outputs(
    configs: list[PretrainRunConfig],
    *,
    outdir: Path | str = OUTDIR,
    mode: str,
    input_report: dict[str, Any],
    reason: str,
) -> dict[str, Any]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = [_report_row_from_config(config, status="planned", message=reason) for config in configs]
    pd.DataFrame(rows).to_csv(outdir / "model_reports.csv", index=False)
    summary = {
        "version": "V441",
        "status": "planned",
        "mode": mode,
        "launch_policy": "non_quick_modes_are_planned_by_default",
        "reason": reason,
        "input_report": input_report,
        "models_run": [],
        "models_planned": [config.name for config in configs],
        "model_configs": [asdict(config) for config in configs],
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "outputs": {
            "model_reports": str(outdir / "model_reports.csv"),
            "pretrain_run_summary": str(outdir / "pretrain_run_summary.json"),
        },
    }
    (outdir / "pretrain_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary


def run_v441_pretraining(
    *,
    mode: str,
    models: list[str] | None = None,
    root: Path | str = ROOT,
    outdir: Path | str = OUTDIR,
    execute_non_quick: bool = False,
) -> dict[str, Any]:
    grid = build_professor_model_grid()
    configs = select_configs_for_mode(grid, mode=mode, models=models)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seq, input_report = load_professor_pretrain_sequences(root=root)

    if input_report["status"] == "no_input":
        return write_planned_run_outputs(
            configs,
            outdir=outdir,
            mode=mode,
            input_report=input_report,
            reason=input_report["message"],
        )

    if mode != "quick" and not execute_non_quick:
        return write_planned_run_outputs(
            configs,
            outdir=outdir,
            mode=mode,
            input_report=input_report,
            reason="Use --execute-non-quick to launch medium/full jobs.",
        )

    train_seq = _cap_quick_sequences(seq) if mode == "quick" else seq
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for config in configs:
        try:
            result = train_one_pretrain_model(train_seq, config=_to_v431_model_config(config), outdir=outdir)
            reports.append(result["report"])
        except Exception as exc:  # pragma: no cover - script resilience for long sweeps
            errors.append({**_report_row_from_config(config, status="error", message=str(exc))})

    rows: list[dict[str, Any]] = []
    for report in reports:
        final_epoch = report["epoch_reports"][-1] if report["epoch_reports"] else {}
        rows.append(
            {
                "model_name": report["model_name"],
                "model_type": report["model_type"],
                "embedding_dim": report["config"]["embedding_dim"],
                "hidden_dim": report["config"]["hidden_dim"],
                "layers": report["config"]["layers"],
                "dropout": report["dropout"],
                "mask_probability": report["mask_probability"],
                "epochs": report["config"]["epochs"],
                "max_windows": report["config"]["max_windows"],
                "train_windows": report["train_windows"],
                "vocab_size": report["vocab_size"],
                "final_loss": final_epoch.get("loss"),
                "status": "trained",
                "message": "",
            }
        )
    rows.extend(errors)
    pd.DataFrame(rows).to_csv(outdir / "model_reports.csv", index=False)

    status = "complete" if reports and not errors else ("partial" if reports else "error")
    summary = {
        "version": "V441",
        "status": status,
        "mode": mode,
        "launch_policy": "quick_trains_bounded_configs",
        "input_report": input_report,
        "training_rows_used": int(len(train_seq)),
        "source_counts_used": train_seq.groupby("source_dataset").size().to_dict() if not train_seq.empty else {},
        "models_run": [report["model_name"] for report in reports],
        "models_planned": [config.name for config in configs],
        "model_configs": [asdict(config) for config in configs],
        "model_configs_run": [report["config"] for report in reports],
        "errors": errors,
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "outputs": {
            "model_reports": str(outdir / "model_reports.csv"),
            "pretrain_run_summary": str(outdir / "pretrain_run_summary.json"),
        },
        "v431_run_pipeline_available": callable(run_pipeline),
    }
    (outdir / "pretrain_run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V441 full external pretraining configs through V431 primitives.")
    parser.add_argument("--mode", choices=sorted(MODE_DEFAULTS), default="quick")
    parser.add_argument("--models", default=None, help="Comma-separated V441 model names.")
    parser.add_argument("--root", default=str(ROOT), help="Workspace root containing V440/V430 input directories.")
    parser.add_argument("--outdir", default=str(OUTDIR), help="Output directory.")
    parser.add_argument(
        "--execute-non-quick",
        action="store_true",
        help="Actually launch medium/full configs. Without this flag they are only recorded.",
    )
    args = parser.parse_args()
    summary = run_v441_pretraining(
        mode=args.mode,
        models=_parse_models(args.models),
        root=Path(args.root),
        outdir=Path(args.outdir),
        execute_non_quick=bool(args.execute_non_quick),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

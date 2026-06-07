"""V431 multi-model external sequence pretraining zoo.

Trains bounded GRU/LSTM/Transformer encoders on coarse external sequence
streams only. Exact AICUP labels are stripped before feature construction and
are never exported.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
V430_INPUT = ROOT / "v430_external_audit_canonical_expander" / "canonical_expanded_events.csv"
V414_FALLBACK_INPUT = ROOT / "v414_masked_pretraining_inputs" / "pretrain_sequences.csv"
OUTDIR = ROOT / "v431_external_sequence_model_zoo"

FORBIDDEN_COLUMNS = {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}
SPECIAL_TOKENS = ["<PAD>", "<MASK>", "<UNK>"]
TOKEN_COLUMNS = {
    "family": "token_family",
    "phase": "phase",
    "terminal": "terminal_label",
    "depth": "landing_depth_bin",
    "side": "landing_side_bin",
    "speed": "speed_bin",
    "spin": "spin_bin",
}
OBJECTIVES = ["family", "depth", "side", "speed", "spin", "terminal"]


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_type: str
    embedding_dim: int
    hidden_dim: int
    layers: int
    dropout: float
    epochs: int
    max_windows: int
    batch_size: int = 256
    context_events: int = 8
    mask_probability: float = 0.15
    learning_rate: float = 1e-3
    min_count: int = 1
    seed: int = 431
    device: str = "cpu"


def build_model_registry(*, include_large: bool = False) -> dict[str, ModelConfig]:
    """Return V431 model configs; large variants are opt-in."""

    registry = {
        "gru_small": ModelConfig("gru_small", "gru", 32, 64, 1, 0.10, 2, 80_000),
        "gru_medium": ModelConfig("gru_medium", "gru", 64, 128, 2, 0.15, 3, 120_000),
        "lstm_small": ModelConfig("lstm_small", "lstm", 32, 64, 1, 0.10, 2, 80_000),
        "lstm_medium": ModelConfig("lstm_medium", "lstm", 64, 128, 2, 0.15, 3, 120_000),
        "transformer_small": ModelConfig("transformer_small", "transformer", 32, 64, 2, 0.10, 2, 80_000),
        "transformer_medium": ModelConfig("transformer_medium", "transformer", 64, 128, 3, 0.15, 3, 120_000),
    }
    if include_large:
        registry.update(
            {
                "gru_large": ModelConfig("gru_large", "gru", 96, 192, 3, 0.20, 4, 180_000),
                "lstm_large": ModelConfig("lstm_large", "lstm", 96, 192, 3, 0.20, 4, 180_000),
                "transformer_large": ModelConfig("transformer_large", "transformer", 96, 192, 4, 0.20, 4, 180_000),
            }
        )
    return registry


def _normalize_text(value: Any, default: str = "unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return default
    return "_".join(text.split())


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _axis_bins(values: pd.Series, labels: tuple[str, str, str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    out = pd.Series(["unknown"] * len(values), index=values.index, dtype=object)
    valid = numeric.dropna()
    if len(valid) < 3 or valid.nunique() < 3:
        return out
    q1, q2 = valid.quantile([1 / 3, 2 / 3]).tolist()
    out.loc[numeric <= q1] = labels[0]
    out.loc[(numeric > q1) & (numeric <= q2)] = labels[1]
    out.loc[numeric > q2] = labels[2]
    return out


def _value_bins_by_source(frame: pd.DataFrame, column: str, labels: tuple[str, str, str]) -> pd.Series:
    out = pd.Series(["unknown"] * len(frame), index=frame.index, dtype=object)
    if column not in frame.columns:
        return out
    sources = frame.get("source_dataset", pd.Series(["unknown"] * len(frame), index=frame.index)).map(_normalize_text)
    for _, idx in sources.groupby(sources, dropna=False).groups.items():
        out.loc[idx] = _axis_bins(frame.loc[idx, column], labels)
    return out


def _strip_forbidden(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[col for col in FORBIDDEN_COLUMNS if col in frame.columns], errors="ignore").copy()


def _coerce_pretrain_schema(frame: pd.DataFrame) -> pd.DataFrame:
    clean = _strip_forbidden(frame)
    if "token_family" not in clean.columns and "coarse_family" in clean.columns:
        clean["token_family"] = clean["coarse_family"]
    if "terminal_label" not in clean.columns and "target_terminal" in clean.columns:
        clean["terminal_label"] = clean["target_terminal"]
    if "landing_depth_bin" not in clean.columns and "target_depth_bin" in clean.columns:
        clean["landing_depth_bin"] = clean["target_depth_bin"]
    if "landing_side_bin" not in clean.columns and "target_side_bin" in clean.columns:
        clean["landing_side_bin"] = clean["target_side_bin"]
    if "speed_bin" not in clean.columns:
        clean["speed_bin"] = _value_bins_by_source(clean, "speed_norm", ("low", "medium", "high"))
    if "spin_bin" not in clean.columns:
        clean["spin_bin"] = _value_bins_by_source(clean, "spin_norm", ("low", "medium", "high"))

    required_defaults = {
        "source_dataset": "unknown_source",
        "sequence_id": "unknown_sequence",
        "event_index": 0,
        "token_family": "unknown",
        "phase": "unknown",
        "terminal_label": "unknown",
        "landing_depth_bin": "unknown",
        "landing_side_bin": "unknown",
        "speed_bin": "unknown",
        "spin_bin": "unknown",
    }
    for column, default in required_defaults.items():
        if column not in clean.columns:
            clean[column] = default

    clean = clean[list(required_defaults)].copy()
    clean["event_index"] = pd.to_numeric(clean["event_index"], errors="coerce").fillna(0).astype(int)
    for column in clean.columns:
        if column != "event_index":
            clean[column] = clean[column].map(_normalize_text)
    return clean.sort_values(["source_dataset", "sequence_id", "event_index"]).reset_index(drop=True)


def load_pretrain_sequences(*, root: Path | str = ROOT) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load V430 canonical events first, then V414 fallback, stripping exact labels."""

    root = Path(root)
    candidates = [
        (root / "v430_external_audit_canonical_expander" / "canonical_expanded_events.csv", "v430_canonical_expanded_events"),
        (root / "v414_masked_pretraining_inputs" / "pretrain_sequences.csv", "v414_pretrain_sequences"),
    ]
    for path, kind in candidates:
        if path.exists():
            raw = pd.read_csv(path, low_memory=False)
            seq = _coerce_pretrain_schema(raw)
            return seq, {
                "status": "loaded",
                "input_path": str(path),
                "input_kind": kind,
                "input_rows": int(len(raw)),
                "usable_rows": int(len(seq)),
                "stripped_forbidden_columns": sorted(FORBIDDEN_COLUMNS & set(raw.columns)),
            }
    return pd.DataFrame(columns=list(_coerce_pretrain_schema(pd.DataFrame()).columns)), {
        "status": "no_input",
        "message": "No V430 canonical_expanded_events.csv or V414 pretrain_sequences.csv input was found.",
        "checked_paths": [str(path) for path, _ in candidates],
    }


def _event_tokens(row: pd.Series) -> list[str]:
    return [f"{stream}={_normalize_text(row[column])}" for stream, column in TOKEN_COLUMNS.items()]


def _build_vocabulary(seq: pd.DataFrame, min_count: int) -> tuple[dict[str, int], dict[str, int]]:
    counts: dict[str, int] = {}
    for _, row in seq.iterrows():
        for token in _event_tokens(row):
            counts[token] = counts.get(token, 0) + 1
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token, count in sorted(counts.items()):
        if count >= min_count and token not in vocab:
            vocab[token] = len(vocab)
    return vocab, counts


def _label_maps(seq: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for objective in OBJECTIVES:
        column = TOKEN_COLUMNS[objective]
        values = sorted(seq[column].map(_normalize_text).dropna().unique().tolist()) or ["unknown"]
        maps[objective] = {label: idx for idx, label in enumerate(values)}
    return maps


def _event_token_ids(row: pd.Series, vocab: dict[str, int]) -> list[int]:
    unk = vocab["<UNK>"]
    return [vocab.get(token, unk) for token in _event_tokens(row)]


def _build_windows(
    seq: pd.DataFrame,
    vocab: dict[str, int],
    labels: dict[str, dict[str, int]],
    config: ModelConfig,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for (source, sequence_id), group in seq.groupby(["source_dataset", "sequence_id"], sort=True):
        group = group.reset_index(drop=True)
        if len(group) < 2:
            continue
        token_ids = [_event_token_ids(row, vocab) for _, row in group.iterrows()]
        for pos in range(1, len(group)):
            start = max(0, pos - config.context_events)
            target = group.iloc[pos]
            windows.append(
                {
                    "source_dataset": source,
                    "sequence_id": sequence_id,
                    "target_event_index": int(target["event_index"]),
                    "input_ids": token_ids[start:pos],
                    "targets": {
                        objective: labels[objective][_normalize_text(target[TOKEN_COLUMNS[objective]])]
                        for objective in OBJECTIVES
                    },
                }
            )
            if len(windows) >= config.max_windows:
                return windows
    return windows


class _WindowDataset(Dataset):
    def __init__(self, windows: list[dict[str, Any]]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.windows[index]


def _collate_windows(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    stream_count = len(TOKEN_COLUMNS)
    max_len = max(len(item["input_ids"]) for item in batch)
    ids = torch.zeros((len(batch), max_len, stream_count), dtype=torch.long)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    targets = {objective: torch.empty(len(batch), dtype=torch.long) for objective in OBJECTIVES}
    for row_idx, item in enumerate(batch):
        values = torch.tensor(item["input_ids"], dtype=torch.long)
        ids[row_idx, : len(item["input_ids"]), :] = values
        mask[row_idx, : len(item["input_ids"])] = True
        for objective in OBJECTIVES:
            targets[objective][row_idx] = int(item["targets"][objective])
    return ids, mask, targets


class _ZooSequenceModel(nn.Module):
    def __init__(self, vocab_size: int, label_sizes: dict[str, int], config: ModelConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(config.dropout)
        recurrent_dropout = config.dropout if config.layers > 1 else 0.0
        if config.model_type == "gru":
            self.encoder = nn.GRU(
                config.embedding_dim,
                config.hidden_dim,
                num_layers=config.layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.input_projection = None
            self.positional = None
        elif config.model_type == "lstm":
            self.encoder = nn.LSTM(
                config.embedding_dim,
                config.hidden_dim,
                num_layers=config.layers,
                batch_first=True,
                dropout=recurrent_dropout,
            )
            self.input_projection = None
            self.positional = None
        elif config.model_type == "transformer":
            nhead = 4 if config.hidden_dim % 4 == 0 else 2
            self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
            self.positional = nn.Embedding(max(config.context_events + 2, 16), config.hidden_dim)
            layer = nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=nhead,
                dim_feedforward=max(config.hidden_dim * 2, 32),
                dropout=config.dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
        else:
            raise ValueError(f"Unsupported model_type: {config.model_type}")
        self.projection = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.heads = nn.ModuleDict({objective: nn.Linear(config.hidden_dim, size) for objective, size in label_sizes.items()})

    def encode(self, token_ids: torch.Tensor, event_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        event_vectors = self.embedding(token_ids).mean(dim=2)
        event_vectors = self.input_dropout(event_vectors)
        if self.config.model_type in {"gru", "lstm"}:
            outputs, _ = self.encoder(event_vectors)
        else:
            projected = self.input_projection(event_vectors)
            positions = torch.arange(projected.shape[1], device=projected.device).clamp(max=self.positional.num_embeddings - 1)
            projected = projected + self.positional(positions).unsqueeze(0)
            outputs = self.encoder(projected, src_key_padding_mask=~event_mask)
        lengths = event_mask.sum(dim=1).clamp(min=1) - 1
        batch_index = torch.arange(token_ids.shape[0], device=token_ids.device)
        hidden = outputs[batch_index, lengths]
        return hidden, self.projection(hidden)

    def forward(self, token_ids: torch.Tensor, event_mask: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        hidden, projected = self.encode(token_ids, event_mask)
        return {objective: head(hidden) for objective, head in self.heads.items()}, projected


def _mask_inputs(token_ids: torch.Tensor, event_mask: torch.Tensor, config: ModelConfig) -> torch.Tensor:
    if config.mask_probability <= 0:
        return token_ids
    random_values = torch.rand(token_ids.shape, device=token_ids.device)
    mask = (random_values < config.mask_probability) & event_mask.unsqueeze(-1) & (token_ids > 2)
    masked = token_ids.clone()
    masked[mask] = 1
    return masked


def _embedding_columns(dim: int) -> list[str]:
    return [f"emb_{idx:02d}" for idx in range(dim)]


def _token_embedding_frame(model: _ZooSequenceModel, vocab: dict[str, int], counts: dict[str, int]) -> pd.DataFrame:
    values = model.embedding.weight.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for token, token_id in sorted(vocab.items(), key=lambda item: item[1]):
        row: dict[str, Any] = {"token": token, "token_id": int(token_id), "token_count": int(counts.get(token, 0))}
        for idx, column in enumerate(_embedding_columns(values.shape[1])):
            row[column] = float(values[token_id, idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _sequence_embedding_frame(
    model: _ZooSequenceModel,
    seq: pd.DataFrame,
    vocab: dict[str, int],
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for (source, sequence_id), group in seq.groupby(["source_dataset", "sequence_id"], sort=True):
            group = group.reset_index(drop=True)
            ids = torch.tensor([_event_token_ids(row, vocab) for _, row in group.iterrows()], dtype=torch.long, device=device).unsqueeze(0)
            event_mask = torch.ones((1, ids.shape[1]), dtype=torch.bool, device=device)
            _, projected = model.encode(ids, event_mask)
            emb = projected.squeeze(0).detach().cpu().numpy()
            row: dict[str, Any] = {"source_dataset": source, "sequence_id": sequence_id, "event_count": int(len(group))}
            for idx, column in enumerate(_embedding_columns(len(emb))):
                row[column] = float(emb[idx])
            rows.append(row)
    return pd.DataFrame(rows)


def _probability_frame(
    model: _ZooSequenceModel,
    windows: list[dict[str, Any]],
    labels: dict[str, dict[str, int]],
    config: ModelConfig,
    device: torch.device,
) -> pd.DataFrame:
    inverse = {objective: {idx: label for label, idx in mapping.items()} for objective, mapping in labels.items()}
    rows: list[dict[str, Any]] = []
    loader = DataLoader(_WindowDataset(windows), batch_size=config.batch_size, shuffle=False, collate_fn=_collate_windows, num_workers=0)
    offset = 0
    model.eval()
    with torch.no_grad():
        for ids, event_mask, _ in loader:
            ids = ids.to(device)
            event_mask = event_mask.to(device)
            logits, _ = model(ids, event_mask)
            batch_size = ids.shape[0]
            batch_probs = {objective: torch.softmax(logit, dim=1).detach().cpu().numpy() for objective, logit in logits.items()}
            for batch_idx in range(batch_size):
                item = windows[offset + batch_idx]
                row: dict[str, Any] = {
                    "source_dataset": item["source_dataset"],
                    "sequence_id": item["sequence_id"],
                    "target_event_index": int(item["target_event_index"]),
                }
                for objective in OBJECTIVES:
                    probs = batch_probs[objective][batch_idx]
                    pred_idx = int(np.argmax(probs))
                    row[f"pred_{objective}"] = inverse[objective][pred_idx]
                    row[f"prob_{objective}"] = float(probs[pred_idx])
                rows.append(row)
            offset += batch_size
    return pd.DataFrame(rows)


def _assert_no_forbidden(outputs: dict[str, pd.DataFrame]) -> None:
    for name, frame in outputs.items():
        overlap = FORBIDDEN_COLUMNS & set(frame.columns)
        if overlap:
            raise ValueError(f"{name} contains forbidden exact AICUP columns: {sorted(overlap)}")


def train_one_pretrain_model(seq: pd.DataFrame, *, config: ModelConfig, outdir: Path | str) -> dict[str, Any]:
    """Train one tiny/bounded model and export embeddings, probabilities, report, checkpoint."""

    _set_seed(config.seed)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    model_dir = outdir / config.name
    model_dir.mkdir(parents=True, exist_ok=True)

    clean = _coerce_pretrain_schema(seq)
    vocab, token_counts = _build_vocabulary(clean, config.min_count)
    label_maps = _label_maps(clean)
    windows = _build_windows(clean, vocab, label_maps, config)
    if not windows:
        raise ValueError("No trainable sequence windows were built; need at least one two-event sequence")

    device_name = config.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    label_sizes = {objective: len(mapping) for objective, mapping in label_maps.items()}
    model = _ZooSequenceModel(len(vocab), label_sizes, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    loader = DataLoader(
        _WindowDataset(windows),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=_collate_windows,
        num_workers=0,
    )

    epoch_reports: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0.0
        total_rows = 0
        total_by_objective = {objective: 0.0 for objective in OBJECTIVES}
        for ids, event_mask, targets in loader:
            ids = ids.to(device)
            event_mask = event_mask.to(device)
            targets = {objective: target.to(device) for objective, target in targets.items()}
            masked_ids = _mask_inputs(ids, event_mask, config)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(masked_ids, event_mask)
            losses = {objective: loss_fn(logits[objective], targets[objective]) for objective in OBJECTIVES}
            loss = sum(losses.values())
            loss.backward()
            optimizer.step()
            rows = ids.shape[0]
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows
            for objective, objective_loss in losses.items():
                total_by_objective[objective] += float(objective_loss.detach().cpu()) * rows
        epoch_report = {"epoch": float(epoch + 1), "loss": float(total_loss / max(total_rows, 1))}
        for objective, objective_loss in total_by_objective.items():
            epoch_report[f"{objective}_loss"] = float(objective_loss / max(total_rows, 1))
        epoch_reports.append(epoch_report)

    token_embeddings = _token_embedding_frame(model, vocab, token_counts)
    sequence_embeddings = _sequence_embedding_frame(model, clean, vocab, device)
    probabilities = _probability_frame(model, windows, label_maps, config, device)
    outputs = {
        "token_embeddings": token_embeddings,
        "sequence_embeddings": sequence_embeddings,
        "probabilities": probabilities,
    }
    _assert_no_forbidden(outputs)

    checkpoint_path = model_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "vocab": vocab,
            "label_maps": label_maps,
            "objectives": OBJECTIVES,
        },
        checkpoint_path,
    )

    token_path = outdir / f"token_embeddings_{config.name}.csv"
    sequence_path = outdir / f"sequence_embeddings_{config.name}.csv"
    probability_path = outdir / f"probabilities_{config.name}.csv"
    report_path = model_dir / "pretrain_report.json"
    token_embeddings.to_csv(token_path, index=False)
    sequence_embeddings.to_csv(sequence_path, index=False)
    probabilities.to_csv(probability_path, index=False)
    token_embeddings.to_csv(model_dir / "token_embeddings.csv", index=False)
    sequence_embeddings.to_csv(model_dir / "sequence_embeddings.csv", index=False)
    probabilities.to_csv(model_dir / "probabilities.csv", index=False)

    report: dict[str, Any] = {
        "version": "V431",
        "model_name": config.name,
        "model_type": config.model_type,
        "objectives": list(OBJECTIVES),
        "input_rows": int(len(seq)),
        "clean_rows": int(len(clean)),
        "train_windows": int(len(windows)),
        "source_counts": clean.groupby("source_dataset").size().to_dict() if not clean.empty else {},
        "label_sizes": {objective: int(size) for objective, size in label_sizes.items()},
        "vocab_size": int(len(vocab)),
        "epoch_reports": epoch_reports,
        "config": asdict(config),
        "dropout": float(config.dropout),
        "mask_probability": float(config.mask_probability),
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "outputs": {
            "token_embeddings": str(token_path),
            "sequence_embeddings": str(sequence_path),
            "probabilities": str(probability_path),
            "checkpoint": str(checkpoint_path),
            "pretrain_report": str(report_path),
        },
        "output_rows": {name: int(len(frame)) for name, frame in outputs.items()},
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return {**outputs, "report": report}


def _quick_config(config: ModelConfig) -> ModelConfig:
    return replace(config, epochs=1, max_windows=min(config.max_windows, 64), batch_size=min(config.batch_size, 32), seed=431)


def _selected_model_names(
    *,
    registry: dict[str, ModelConfig],
    models: list[str] | None,
    run_medium: bool,
    include_large: bool,
) -> list[str]:
    if models:
        unknown = [name for name in models if name not in registry]
        if unknown:
            raise ValueError(f"Unknown model names: {unknown}")
        return models
    names = ["gru_small", "lstm_small", "transformer_small"]
    if run_medium:
        names.extend(["gru_medium", "lstm_medium", "transformer_medium"])
    if include_large:
        names.extend(["gru_large", "lstm_large", "transformer_large"])
    return names


def run_pipeline(
    *,
    root: Path | str = ROOT,
    outdir: Path | str | None = None,
    models: list[str] | None = None,
    quick: bool = False,
    run_medium: bool = False,
    include_large: bool = False,
) -> dict[str, Any]:
    root = Path(root)
    outdir = Path(outdir) if outdir is not None else root / "v431_external_sequence_model_zoo"
    outdir.mkdir(parents=True, exist_ok=True)
    seq, input_report = load_pretrain_sequences(root=root)
    if input_report["status"] == "no_input":
        summary = {
            "version": "V431",
            "status": "no_input",
            "message": input_report["message"],
            "input_report": input_report,
            "models_requested": models or [],
            "model_reports": [],
        }
        (outdir / "model_reports.csv").write_text("model_name,status,message\n", encoding="utf-8")
        (outdir / "model_zoo_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    registry = build_model_registry(include_large=include_large)
    selected = _selected_model_names(registry=registry, models=models, run_medium=run_medium, include_large=include_large)
    reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for name in selected:
        config = registry[name]
        if quick:
            config = _quick_config(config)
        try:
            result = train_one_pretrain_model(seq, config=config, outdir=outdir)
            report = result["report"]
            reports.append(report)
        except Exception as exc:  # pragma: no cover - retained for script resilience
            errors.append({"model_name": name, "status": "error", "message": str(exc)})

    report_rows = []
    for report in reports:
        final_epoch = report["epoch_reports"][-1] if report["epoch_reports"] else {}
        report_rows.append(
            {
                "model_name": report["model_name"],
                "model_type": report["model_type"],
                "embedding_dim": report["config"]["embedding_dim"],
                "hidden_dim": report["config"]["hidden_dim"],
                "layers": report["config"]["layers"],
                "dropout": report["dropout"],
                "mask_probability": report["mask_probability"],
                "epochs": report["config"]["epochs"],
                "train_windows": report["train_windows"],
                "vocab_size": report["vocab_size"],
                "final_loss": final_epoch.get("loss"),
                "status": "trained",
            }
        )
    report_rows.extend(errors)
    model_reports = pd.DataFrame(report_rows)
    model_reports.to_csv(outdir / "model_reports.csv", index=False)
    model_reports.to_csv(outdir / "model_zoo_summary.csv", index=False)

    summary = {
        "version": "V431",
        "status": "complete" if reports and not errors else ("partial" if reports else "error"),
        "input_report": input_report,
        "models_run": [report["model_name"] for report in reports],
        "model_configs_run": [report["config"] for report in reports],
        "errors": errors,
        "outputs": {
            "model_reports": str(outdir / "model_reports.csv"),
            "model_zoo_summary_csv": str(outdir / "model_zoo_summary.csv"),
            "model_zoo_summary_json": str(outdir / "model_zoo_summary.json"),
        },
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
    }
    (outdir / "model_zoo_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return summary


def _parse_models(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train V431 external sequence pretraining model zoo.")
    parser.add_argument("--models", default=None, help="Comma-separated model names to run.")
    parser.add_argument("--quick", action="store_true", help="Run bounded smoke training with one epoch and capped windows.")
    parser.add_argument("--run-medium", action="store_true", help="Include medium configs in the default selection.")
    parser.add_argument("--include-large", action="store_true", help="Register and allow large configs.")
    parser.add_argument("--root", default=str(ROOT), help="Workspace root containing V430/V414 input directories.")
    parser.add_argument("--outdir", default=None, help="Output directory. Defaults to <root>/v431_external_sequence_model_zoo.")
    args = parser.parse_args()
    summary = run_pipeline(
        root=Path(args.root),
        outdir=Path(args.outdir) if args.outdir else None,
        models=_parse_models(args.models),
        quick=bool(args.quick),
        run_medium=bool(args.run_medium),
        include_large=bool(args.include_large),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()

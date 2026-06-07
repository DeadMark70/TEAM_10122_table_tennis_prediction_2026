"""V418 clean external sequence pretraining.

Trains a compact GRU over V414 coarse external sequence tokens and exports
learned token/sequence embeddings without exact AICUP label columns.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "v414_masked_pretraining_inputs" / "pretrain_sequences.csv"
OUTDIR = ROOT / "v418_clean_external_sequence_pretrain"

FORBIDDEN_COLUMNS = {"actionId", "pointId", "serverGetPoint", "spinId", "strengthId", "positionId"}
EXCLUDED_SOURCES = {"TTMATCH", "TT-MatchDynamics", "sonytabletennis"}
SPECIAL_TOKENS = ["<PAD>", "<MASK>", "<UNK>"]
TOKEN_COLUMNS = {
    "fam": "token_family",
    "phase": "phase",
    "terminal": "terminal_label",
    "depth": "landing_depth_bin",
    "side": "landing_side_bin",
    "speed": "speed_bin",
    "spin": "spin_bin",
}
TARGET_STREAMS = ["fam", "depth", "side", "speed", "spin"]


@dataclass(frozen=True)
class TrainConfig:
    max_windows: int = 80_000
    epochs: int = 2
    embedding_dim: int = 32
    hidden_dim: int = 64
    batch_size: int = 256
    context_events: int = 8
    min_count: int = 1
    dropout: float = 0.10
    mask_probability: float = 0.15
    learning_rate: float = 1e-3
    seed: int = 418
    device: str = "cpu"


@dataclass
class TrainResult:
    token_embeddings: pd.DataFrame
    sequence_embeddings: pd.DataFrame
    report: dict[str, Any]


def _normalize_text(value: Any, default: str = "unknown") -> str:
    if pd.isna(value):
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "<na>"}:
        return default
    return "_".join(text.split())


def _set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _clean_source_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
    source = frame.get("source_dataset", pd.Series(["unknown"] * len(frame), index=frame.index)).map(_normalize_text)
    mask = ~source.isin(EXCLUDED_SOURCES)
    mask &= ~source.str.contains("ttmatch", case=False, na=False)
    mask &= ~source.str.contains("sony", case=False, na=False)
    clean = frame.loc[mask].copy()
    clean["source_dataset"] = source.loc[mask].values
    return clean.reset_index(drop=True), int((~mask).sum())


def _prepare_sequences(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["source_dataset", "sequence_id", "event_index", *TOKEN_COLUMNS.values()]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"pretrain_sequences is missing required columns: {missing}")

    seq = frame.copy()
    seq["source_dataset"] = seq["source_dataset"].map(_normalize_text)
    seq["sequence_id"] = seq["sequence_id"].map(_normalize_text)
    seq["_event_index"] = pd.to_numeric(seq["event_index"], errors="coerce").fillna(0)
    for column in TOKEN_COLUMNS.values():
        seq[column] = seq[column].map(_normalize_text)
    return seq.sort_values(["source_dataset", "sequence_id", "_event_index"]).reset_index(drop=True)


def _event_tokens(row: pd.Series) -> list[str]:
    return [f"{prefix}={_normalize_text(row[column])}" for prefix, column in TOKEN_COLUMNS.items()]


def build_vocabulary(pretrain_sequences: pd.DataFrame, min_count: int = 1) -> dict[str, int]:
    """Build a clean token vocabulary from coarse V414 streams only."""

    counts: dict[str, int] = {}
    seq = _prepare_sequences(pretrain_sequences)
    for _, row in seq.iterrows():
        for token in _event_tokens(row):
            counts[token] = counts.get(token, 0) + 1

    tokens = [token for token, count in counts.items() if count >= min_count]
    vocab = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
    for token in sorted(tokens):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def _label_maps(seq: pd.DataFrame) -> dict[str, dict[str, int]]:
    maps: dict[str, dict[str, int]] = {}
    for stream in TARGET_STREAMS:
        column = TOKEN_COLUMNS[stream]
        labels = sorted(seq[column].map(_normalize_text).dropna().unique().tolist())
        if not labels:
            labels = ["unknown"]
        maps[stream] = {label: idx for idx, label in enumerate(labels)}
    return maps


def _event_token_ids(row: pd.Series, vocab: dict[str, int]) -> list[int]:
    unk = vocab["<UNK>"]
    return [vocab.get(token, unk) for token in _event_tokens(row)]


def _build_windows(
    seq: pd.DataFrame,
    vocab: dict[str, int],
    label_maps: dict[str, dict[str, int]],
    config: TrainConfig,
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
                    "start_event_index": int(group.iloc[start]["_event_index"]),
                    "target_event_index": int(target["_event_index"]),
                    "input_ids": token_ids[start:pos],
                    "targets": {stream: label_maps[stream][_normalize_text(target[TOKEN_COLUMNS[stream]])] for stream in TARGET_STREAMS},
                }
            )
    if len(windows) > config.max_windows:
        rng = np.random.default_rng(config.seed)
        keep = np.sort(rng.choice(len(windows), size=config.max_windows, replace=False))
        windows = [windows[int(idx)] for idx in keep]
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
    targets = {stream: torch.empty(len(batch), dtype=torch.long) for stream in TARGET_STREAMS}
    for row_idx, item in enumerate(batch):
        values = torch.tensor(item["input_ids"], dtype=torch.long)
        ids[row_idx, : len(item["input_ids"]), :] = values
        mask[row_idx, : len(item["input_ids"])] = True
        for stream in TARGET_STREAMS:
            targets[stream][row_idx] = int(item["targets"][stream])
    return ids, mask, targets


class _SequencePretrainModel(nn.Module):
    def __init__(self, vocab_size: int, label_sizes: dict[str, int], config: TrainConfig):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=0)
        self.input_dropout = nn.Dropout(config.dropout)
        self.gru = nn.GRU(config.embedding_dim, config.hidden_dim, batch_first=True)
        self.projection = nn.Linear(config.hidden_dim, config.embedding_dim)
        self.heads = nn.ModuleDict({stream: nn.Linear(config.hidden_dim, size) for stream, size in label_sizes.items()})

    def encode(self, token_ids: torch.Tensor, event_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        event_vectors = self.embedding(token_ids).mean(dim=2)
        event_vectors = self.input_dropout(event_vectors)
        outputs, _ = self.gru(event_vectors)
        lengths = event_mask.sum(dim=1).clamp(min=1) - 1
        batch_index = torch.arange(token_ids.shape[0], device=token_ids.device)
        hidden = outputs[batch_index, lengths]
        return hidden, self.projection(hidden)

    def forward(self, token_ids: torch.Tensor, event_mask: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        hidden, projected = self.encode(token_ids, event_mask)
        logits = {stream: head(hidden) for stream, head in self.heads.items()}
        return logits, projected


def _mask_inputs(token_ids: torch.Tensor, event_mask: torch.Tensor, config: TrainConfig) -> torch.Tensor:
    if config.mask_probability <= 0:
        return token_ids
    keep_special = token_ids <= 2
    random_values = torch.rand(token_ids.shape, device=token_ids.device)
    mask = (random_values < config.mask_probability) & event_mask.unsqueeze(-1) & ~keep_special
    masked = token_ids.clone()
    masked[mask] = 1
    return masked


def _embedding_columns(dim: int) -> list[str]:
    return [f"emb_{idx:02d}" for idx in range(dim)]


def _token_embedding_frame(model: _SequencePretrainModel, vocab: dict[str, int], token_counts: dict[str, int]) -> pd.DataFrame:
    emb = model.embedding.weight.detach().cpu().numpy()
    rows: list[dict[str, Any]] = []
    for token, idx in sorted(vocab.items(), key=lambda item: item[0]):
        row: dict[str, Any] = {"token": token, "token_id": int(idx), "token_count": int(token_counts.get(token, 0))}
        for col_idx, col in enumerate(_embedding_columns(emb.shape[1])):
            row[col] = float(emb[idx, col_idx])
        rows.append(row)
    return pd.DataFrame(rows)


def _sequence_embedding_frame(
    model: _SequencePretrainModel,
    seq: pd.DataFrame,
    vocab: dict[str, int],
    device: torch.device,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for (source, sequence_id), group in seq.groupby(["source_dataset", "sequence_id"], sort=True):
            group = group.reset_index(drop=True)
            if group.empty:
                continue
            ids = torch.tensor([_event_token_ids(row, vocab) for _, row in group.iterrows()], dtype=torch.long, device=device).unsqueeze(0)
            mask = torch.ones((1, ids.shape[1]), dtype=torch.bool, device=device)
            _, projected = model.encode(ids, mask)
            values = projected.squeeze(0).detach().cpu().numpy()
            row: dict[str, Any] = {"source_dataset": source, "sequence_id": sequence_id, "event_count": int(len(group))}
            for col_idx, col in enumerate(_embedding_columns(len(values))):
                row[col] = float(values[col_idx])
            rows.append(row)
    return pd.DataFrame(rows)


def _token_counts(seq: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for _, row in seq.iterrows():
        for token in _event_tokens(row):
            counts[token] = counts.get(token, 0) + 1
    return counts


def _assert_no_forbidden_columns(outputs: dict[str, pd.DataFrame]) -> None:
    for name, frame in outputs.items():
        overlap = FORBIDDEN_COLUMNS & set(frame.columns)
        if overlap:
            raise ValueError(f"{name} contains forbidden exact AICUP columns: {sorted(overlap)}")


def train_sequence_model(
    pretrain_sequences: pd.DataFrame,
    *,
    config: TrainConfig | None = None,
    outdir: Path | str | None = None,
) -> TrainResult:
    config = config or TrainConfig()
    _set_deterministic_seed(config.seed)

    clean, excluded_rows = _clean_source_rows(pretrain_sequences)
    seq = _prepare_sequences(clean)
    vocab = build_vocabulary(seq, min_count=config.min_count)
    label_maps = _label_maps(seq)
    windows = _build_windows(seq, vocab, label_maps, config)
    if not windows:
        raise ValueError("No trainable sequence windows were built; need at least one two-event sequence")

    device_name = config.device
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    label_sizes = {stream: len(mapping) for stream, mapping in label_maps.items()}
    model = _SequencePretrainModel(len(vocab), label_sizes, config).to(device)
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
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0.0
        total_rows = 0
        for ids, event_mask, targets in loader:
            ids = ids.to(device)
            event_mask = event_mask.to(device)
            targets = {stream: values.to(device) for stream, values in targets.items()}
            masked_ids = _mask_inputs(ids, event_mask, config)

            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(masked_ids, event_mask)
            loss = sum(loss_fn(logits[stream], targets[stream]) for stream in TARGET_STREAMS)
            loss.backward()
            optimizer.step()

            rows = ids.shape[0]
            total_loss += float(loss.detach().cpu()) * rows
            total_rows += rows
        epoch_reports.append({"epoch": float(epoch + 1), "loss": float(total_loss / max(total_rows, 1))})

    token_embeddings = _token_embedding_frame(model, vocab, _token_counts(seq))
    sequence_embeddings = _sequence_embedding_frame(model, seq, vocab, device)
    outputs = {"token_embeddings": token_embeddings, "sequence_embeddings": sequence_embeddings}
    _assert_no_forbidden_columns(outputs)

    report: dict[str, Any] = {
        "version": "V418",
        "seed": int(config.seed),
        "device": str(device),
        "input_rows": int(len(pretrain_sequences)),
        "clean_rows": int(len(seq)),
        "excluded_rows": int(excluded_rows),
        "train_windows": int(len(windows)),
        "max_windows": int(config.max_windows),
        "epochs": int(config.epochs),
        "embedding_dim": int(config.embedding_dim),
        "hidden_dim": int(config.hidden_dim),
        "batch_size": int(config.batch_size),
        "context_events": int(config.context_events),
        "vocab_size": int(len(vocab)),
        "label_sizes": {stream: int(size) for stream, size in label_sizes.items()},
        "epoch_reports": epoch_reports,
        "source_counts": seq.groupby("source_dataset").size().to_dict() if not seq.empty else {},
        "forbidden_columns": sorted(FORBIDDEN_COLUMNS),
        "output_rows": {name: int(len(frame)) for name, frame in outputs.items()},
    }

    if outdir is not None:
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        token_path = outdir / "token_embeddings.csv"
        sequence_path = outdir / "sequence_embeddings.csv"
        report_path = outdir / "pretraining_report.json"
        token_embeddings.to_csv(token_path, index=False)
        sequence_embeddings.to_csv(sequence_path, index=False)
        report["outdir"] = str(outdir)
        report["outputs"] = {
            "token_embeddings": str(token_path),
            "sequence_embeddings": str(sequence_path),
            "pretraining_report": str(report_path),
        }
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    return TrainResult(token_embeddings=token_embeddings, sequence_embeddings=sequence_embeddings, report=report)


def run_pipeline(
    *,
    input_path: Path | str = INPUT_PATH,
    outdir: Path | str = OUTDIR,
    config: TrainConfig | None = None,
) -> dict[str, Any]:
    input_path = Path(input_path)
    outdir = Path(outdir)
    pretrain_sequences = pd.read_csv(input_path, low_memory=False)
    result = train_sequence_model(pretrain_sequences, config=config or TrainConfig(), outdir=outdir)
    print(json.dumps(result.report, indent=2, ensure_ascii=False, default=str))
    return result.report


def main() -> None:
    run_pipeline()


if __name__ == "__main__":
    main()

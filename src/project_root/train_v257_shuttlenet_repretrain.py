from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from analysis_v257_coachai_schema_helpers import sequence_pad


ROOT = Path(".")
CORPUS_DIR = ROOT / "v257_shuttlenet_corpus"
OUTDIR = ROOT / "v257_shuttlenet_repretrain"
MAX_LEN = 48
RNG = 257


def load_corpus() -> pd.DataFrame:
    parquet = CORPUS_DIR / "v257_canonical_sequences.parquet"
    csv = CORPUS_DIR / "v257_canonical_sequences.csv"
    if parquet.exists():
        return pd.read_parquet(parquet)
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Missing V257 corpus under {CORPUS_DIR}")


def build_vocab(values: pd.Series) -> dict[str, int]:
    uniq = sorted(values.astype(str).fillna("unknown").unique())
    return {value: idx + 1 for idx, value in enumerate(uniq)}


def encode_sequences(corpus: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict]:
    corpus = corpus.sort_values(["global_rally_uid", "stroke_index"], kind="mergesort").reset_index(drop=True)
    vocabs = {
        "shot": build_vocab(corpus["shot_type_raw"]),
        "family": build_vocab(corpus["action_family"]),
        "player": build_vocab(corpus["player_id"]),
        "phase": build_vocab(corpus["phase"]),
    }
    seqs = []
    for _, g in corpus.groupby("global_rally_uid", sort=False):
        if len(g) < 2:
            continue
        seqs.append(
            {
                "shot": [vocabs["shot"][str(v)] for v in g["shot_type_raw"]],
                "family": [vocabs["family"][str(v)] for v in g["action_family"]],
                "player": [vocabs["player"][str(v)] for v in g["player_id"]],
                "phase": [vocabs["phase"][str(v)] for v in g["phase"]],
                "xy": g[["x_norm", "y_norm"]].astype(float).to_numpy(),
                "terminal": g["terminal_like"].astype(float).to_numpy(),
            }
        )
    if not seqs:
        raise RuntimeError("V257 corpus has no sequences with at least two strokes.")

    rows = {k: [] for k in ["shot", "family", "player", "phase", "terminal"]}
    xy_rows = []
    masks = []
    y_family = []
    y_phase = []
    y_terminal = []
    for seq in seqs:
        limit = min(len(seq["shot"]) - 1, MAX_LEN)
        rows["shot"].append(sequence_pad(seq["shot"][:limit], MAX_LEN))
        rows["family"].append(sequence_pad(seq["family"][:limit], MAX_LEN))
        rows["player"].append(sequence_pad(seq["player"][:limit], MAX_LEN))
        rows["phase"].append(sequence_pad(seq["phase"][:limit], MAX_LEN))
        rows["terminal"].append(sequence_pad(seq["terminal"][:limit], MAX_LEN, pad_value=0))
        xy = np.zeros((MAX_LEN, 2), dtype=np.float32)
        xy[:limit] = seq["xy"][:limit]
        xy_rows.append(xy)
        masks.append(sequence_pad([1] * limit, MAX_LEN, pad_value=0))
        y_family.append(sequence_pad(seq["family"][1 : limit + 1], MAX_LEN))
        y_phase.append(sequence_pad(seq["phase"][1 : limit + 1], MAX_LEN))
        y_terminal.append(sequence_pad(seq["terminal"][1 : limit + 1], MAX_LEN, pad_value=0))

    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays["xy"] = np.asarray(xy_rows, dtype=np.float32)
    arrays["mask"] = np.asarray(masks, dtype=np.float32)
    arrays["y_family"] = np.asarray(y_family, dtype=np.int64)
    arrays["y_phase"] = np.asarray(y_phase, dtype=np.int64)
    arrays["y_terminal"] = np.asarray(y_terminal, dtype=np.float32)
    config = {"max_len": MAX_LEN, "vocabs": vocabs}
    return arrays, config


def train_torch(arrays: dict[str, np.ndarray], config: dict) -> dict:
    import torch
    import torch.nn as nn

    class V257SequenceEncoder(nn.Module):
        def __init__(self, n_shot: int, n_family: int, n_player: int, hidden: int = 96):
            super().__init__()
            self.shot_emb = nn.Embedding(n_shot, 32, padding_idx=0)
            self.family_emb = nn.Embedding(n_family, 16, padding_idx=0)
            self.player_emb = nn.Embedding(n_player, 16, padding_idx=0)
            self.xy_proj = nn.Linear(2, 16)
            self.gru = nn.GRU(80, hidden, batch_first=True)
            self.family_head = nn.Linear(hidden, n_family)
            self.phase_head = nn.Linear(hidden, max(len(config["vocabs"]["phase"]) + 1, 6))
            self.terminal_head = nn.Linear(hidden, 1)

        def forward(self, shot, family, player, xy):
            x = torch.cat(
                [self.shot_emb(shot), self.family_emb(family), self.player_emb(player), torch.relu(self.xy_proj(xy))],
                dim=-1,
            )
            h, _ = self.gru(x)
            return {
                "hidden": h,
                "family": self.family_head(h),
                "phase": self.phase_head(h),
                "terminal": self.terminal_head(h).squeeze(-1),
            }

    idx = np.arange(len(arrays["shot"]))
    train_idx, valid_idx = train_test_split(idx, test_size=0.2, random_state=RNG) if len(idx) > 5 else (idx, idx)
    model = V257SequenceEncoder(
        n_shot=len(config["vocabs"]["shot"]) + 1,
        n_family=len(config["vocabs"]["family"]) + 1,
        n_player=len(config["vocabs"]["player"]) + 1,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    metrics = []
    batch_size = 64
    for epoch in range(4):
        model.train()
        rng = np.random.default_rng(RNG + epoch)
        losses = []
        for start in range(0, len(train_idx), batch_size):
            batch = rng.permutation(train_idx)[start : start + batch_size]
            shot = torch.as_tensor(arrays["shot"][batch], dtype=torch.long)
            family = torch.as_tensor(arrays["family"][batch], dtype=torch.long)
            player = torch.as_tensor(arrays["player"][batch], dtype=torch.long)
            xy = torch.as_tensor(arrays["xy"][batch], dtype=torch.float32)
            y_family = torch.as_tensor(arrays["y_family"][batch], dtype=torch.long)
            y_phase = torch.as_tensor(arrays["y_phase"][batch], dtype=torch.long)
            y_terminal = torch.as_tensor(arrays["y_terminal"][batch], dtype=torch.float32)
            mask = torch.as_tensor(arrays["mask"][batch], dtype=torch.float32)
            out = model(shot, family, player, xy)
            fam_loss = ce(out["family"].reshape(-1, out["family"].shape[-1]), y_family.reshape(-1)).reshape_as(mask)
            phase_loss = ce(out["phase"].reshape(-1, out["phase"].shape[-1]), y_phase.reshape(-1)).reshape_as(mask)
            term_loss = bce(out["terminal"], y_terminal) * mask
            denom = mask.sum().clamp_min(1.0)
            loss = (fam_loss * mask).sum() / denom + 0.3 * (phase_loss * mask).sum() / denom + 0.3 * term_loss.sum() / denom
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        metrics.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})

    model.eval()
    with torch.no_grad():
        out = model(
            torch.as_tensor(arrays["shot"][valid_idx], dtype=torch.long),
            torch.as_tensor(arrays["family"][valid_idx], dtype=torch.long),
            torch.as_tensor(arrays["player"][valid_idx], dtype=torch.long),
            torch.as_tensor(arrays["xy"][valid_idx], dtype=torch.float32),
        )
    pred_family = out["family"].argmax(-1).cpu().numpy().reshape(-1)
    true_family = arrays["y_family"][valid_idx].reshape(-1)
    mask = arrays["mask"][valid_idx].reshape(-1).astype(bool)
    family_f1 = f1_score(true_family[mask], pred_family[mask], average="macro", zero_division=0)
    majority = np.bincount(true_family[mask]).argmax() if mask.any() else 0
    majority_f1 = f1_score(true_family[mask], np.full(mask.sum(), majority), average="macro", zero_division=0) if mask.any() else 0.0

    OUTDIR.mkdir(exist_ok=True)
    ckpt = OUTDIR / "v257_encoder.pt"
    torch.save({"model_state": model.state_dict(), "config": config}, ckpt)
    config_path = OUTDIR / "v257_encoder_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    metrics[-1]["valid_family_macro_f1"] = float(family_f1)
    metrics[-1]["valid_family_majority_macro_f1"] = float(majority_f1)
    pd.DataFrame(metrics).to_csv(OUTDIR / "v257_pretrain_metrics.csv", index=False)
    return {"checkpoint": str(ckpt), "valid_family_macro_f1": float(family_f1), "majority_macro_f1": float(majority_f1)}


def main() -> None:
    corpus = load_corpus()
    arrays, config = encode_sequences(corpus)
    result = train_torch(arrays, config)
    print(json.dumps(result))


if __name__ == "__main__":
    main()

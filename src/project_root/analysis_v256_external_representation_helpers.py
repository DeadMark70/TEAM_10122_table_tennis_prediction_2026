from __future__ import annotations

import numpy as np
import pandas as pd


FAMILY_CLASSES = ["Zero", "Attack", "Control", "Defensive", "Serve"]
PHASE_CLASSES = [
    "receive_like",
    "third_ball_like",
    "fourth_ball_like",
    "rally_like",
    "serve_like",
    "terminal_like",
]
BIN_CLASSES = ["low", "mid", "high", "missing"]


def normalize_rows_safe(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError("matrix must be 2D")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, 0.0, None)
    sums = arr.sum(axis=1, keepdims=True)
    out = arr.copy()
    zero = sums[:, 0] <= 0
    if np.any(~zero):
        out[~zero] = out[~zero] / sums[~zero]
    if np.any(zero):
        out[zero] = 1.0 / arr.shape[1]
    return out


def numeric_bin(values: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    result = pd.Series("missing", index=values.index, dtype="object")
    present = numeric.notna()
    if present.any():
        cut = pd.cut(
            numeric[present],
            bins=[-np.inf, *bins, np.inf],
            labels=labels,
            include_lowest=True,
            right=True,
        )
        result.loc[present] = cut.astype(str)
    return result


def external_target_frame(corpus: pd.DataFrame) -> pd.DataFrame:
    if "action_family" in corpus.columns:
        family = corpus["action_family"]
    else:
        family = corpus.get("coarse_family", pd.Series(index=corpus.index, dtype="object"))
    family = family.fillna("Zero")
    family = family.where(family.isin(FAMILY_CLASSES), "Zero")

    phase = corpus.get("phase", pd.Series(index=corpus.index, dtype="object")).fillna("rally_like")
    phase = phase.where(phase.isin(PHASE_CLASSES), "rally_like")

    terminal = pd.to_numeric(corpus.get("terminal_like", 0), errors="coerce").fillna(0).astype(int)
    terminal = terminal.clip(0, 1)

    speed_bin = numeric_bin(corpus.get("speed", pd.Series(index=corpus.index)), [0.5, 2.0], ["low", "mid", "high"])
    spin_bin = numeric_bin(corpus.get("spin", pd.Series(index=corpus.index)), [1.0, 6.0], ["low", "mid", "high"])
    depth_bin = numeric_bin(corpus.get("landing_y", pd.Series(index=corpus.index)), [0.33, 0.66], ["low", "mid", "high"])

    return pd.DataFrame(
        {
            "family": family.astype(str),
            "phase": phase.astype(str),
            "terminal": terminal.astype(int),
            "speed_bin": speed_bin.astype(str),
            "spin_bin": spin_bin.astype(str),
            "depth_bin": depth_bin.astype(str),
        },
        index=corpus.index,
    )


def ensure_probability_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in columns:
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else 0.0
    arr = normalize_rows_safe(out.to_numpy(dtype=float))
    return pd.DataFrame(arr, columns=columns, index=df.index)

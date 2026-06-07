from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BADMINTON_FAMILY_MAP = {
    "serve": "Serve",
    "short service": "Serve",
    "long service": "Serve",
    "發短球": "Serve",
    "發長球": "Serve",
    "clear": "Defensive",
    "lob": "Defensive",
    "defensive clear": "Defensive",
    "長球": "Defensive",
    "挑球": "Defensive",
    "smash": "Attack",
    "push/rush": "Attack",
    "drive": "Attack",
    "點扣": "Attack",
    "殺球": "Attack",
    "平球": "Attack",
    "drop": "Control",
    "net shot": "Control",
    "net": "Control",
    "切球": "Control",
    "網前球": "Control",
    "擋小球": "Control",
    "放小球": "Control",
    "unknown": "Zero",
}


ACTION_FAMILY_NAMES = ["Serve", "Attack", "Control", "Defensive", "Zero"]


def forbid_ttmatch_path(path: str | Path) -> None:
    text = str(path).replace("\\", "/").lower()
    if "ttmatch" in text:
        raise RuntimeError(f"TTMATCH is banned for clean V257 training: {path}")


def canonicalize_phase(stroke_index_zero_based: int) -> str:
    idx = int(stroke_index_zero_based)
    if idx == 0:
        return "serve_like"
    if idx == 1:
        return "receive_like"
    if idx == 2:
        return "third_ball_like"
    if idx == 3:
        return "fourth_ball_like"
    return "rally_like"


def sequence_pad(values: list[int] | np.ndarray, max_len: int, pad_value: int = 0) -> np.ndarray:
    arr = np.full(int(max_len), int(pad_value), dtype=int)
    src = np.asarray(values, dtype=int)[: int(max_len)]
    arr[: len(src)] = src
    return arr


def build_padding_mask(values: list[int] | np.ndarray, pad_value: int = 0) -> np.ndarray:
    arr = np.asarray(values)
    return (arr != pad_value).astype(int)


def normalize_xy(
    x: pd.Series,
    y: pd.Series,
    width: float = 355.0,
    height: float = 480.0,
) -> tuple[np.ndarray, np.ndarray]:
    nx = pd.to_numeric(x, errors="coerce").fillna(width / 2.0).to_numpy(dtype=float)
    ny = pd.to_numeric(y, errors="coerce").fillna(height / 2.0).to_numpy(dtype=float)
    nx = np.clip((nx / width) * 2.0 - 1.0, -1.0, 1.0)
    ny = np.clip((ny / height) * 2.0 - 1.0, -1.0, 1.0)
    return nx, ny


def badminton_family(label: object) -> str:
    key = str(label).strip().lower()
    return BADMINTON_FAMILY_MAP.get(key, "Zero")

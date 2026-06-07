"""V319 clean server value/state research.

Server-only candidates around the V306 clean point anchor. Action and point are
fixed, while serverGetPoint receives tiny rank/MAD-capped blends from a clean
value/state model and V300 server artifacts. Outputs stay under
v319_clean_server_value_state.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "v319_clean_server_value_state"
ANCHOR_PATH = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
V300_SEARCH_PATH = ROOT / "v300_clean_server_blend_recycler" / "v300_server_search.csv"
V300_RANK_SOURCE_PATH = (
    ROOT
    / "v300_clean_server_blend_recycler"
    / "submission_v300_rankavg_w0p02__v173action_v261point_server.csv"
)
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
BANNED_PATH_TOKENS = ("TTMATCH", "OLD_SERVER", "OLDSERVER")

SEARCH_PATH = OUT_DIR / "v319_server_value_state_search.csv"
REPORT_JSON_PATH = OUT_DIR / "v319_report.json"
REPORT_MD_PATH = OUT_DIR / "v319_report.md"

RANKTINY_NAME = "submission_v319_server_ranktiny__v173action_v306point.csv"
MAD002_NAME = "submission_v319_server_valueblend_mad0p002__v173action_v306point.csv"
MAD005_NAME = "submission_v319_server_valueblend_mad0p005__v173action_v306point.csv"


@dataclass(frozen=True)
class ServerCandidate:
    key: str
    server: np.ndarray
    source: str
    oof_auc: float
    anchor_auc: float
    target_mad: float


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def relative_path(path: Path | str) -> str:
    path_obj = Path(path)
    try:
        return path_obj.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path_obj.as_posix()


def no_banned_path_guard(paths: list[Path | str]) -> None:
    bad = []
    for path in paths:
        upper = str(path).upper()
        if any(token in upper for token in BANNED_PATH_TOKENS):
            bad.append(str(path))
    if bad:
        raise ValueError(f"Banned clean-branch input path: {bad}")


def cap_prob(values: np.ndarray | pd.Series) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(arr, nan=0.5, posinf=1.0 - 1e-6, neginf=1e-6)
    return np.clip(arr, 1e-6, 1.0 - 1e-6)


def corr(a: np.ndarray | pd.Series, b: np.ndarray | pd.Series) -> float:
    left = np.asarray(a, dtype=float)
    right = np.asarray(b, dtype=float)
    if len(left) < 2 or np.std(left) == 0.0 or np.std(right) == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def validate_submission_frame(df: pd.DataFrame, *, expected_rows: int = EXPECTED_ROWS) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"columns={list(df.columns)} expected={SUBMISSION_COLUMNS}")
    if len(df) != expected_rows:
        raise ValueError(f"rows={len(df)} expected={expected_rows}")
    if not df["actionId"].astype(int).between(0, 18).all():
        raise ValueError("actionId out of range")
    if not df["pointId"].astype(int).between(0, 9).all():
        raise ValueError("pointId out of range")
    server = pd.to_numeric(df["serverGetPoint"], errors="coerce")
    if server.isna().any() or not np.isfinite(server.to_numpy(dtype=float)).all():
        raise ValueError("serverGetPoint must be finite")
    if not server.between(0.0, 1.0).all():
        raise ValueError("serverGetPoint must be in [0, 1]")


def load_submission(path: Path, *, expected_rows: int = EXPECTED_ROWS) -> pd.DataFrame:
    no_banned_path_guard([path])
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    validate_submission_frame(df, expected_rows=expected_rows)
    return df


def rank_normalize_to_anchor(source: np.ndarray | pd.Series, anchor: np.ndarray | pd.Series) -> np.ndarray:
    source_arr = np.asarray(source, dtype=float)
    finite = source_arr[np.isfinite(source_arr)]
    fill = float(np.median(finite)) if len(finite) else 0.0
    source_arr = np.nan_to_num(source_arr, nan=fill, posinf=fill, neginf=fill)
    anchor_sorted = np.sort(cap_prob(anchor))
    if len(source_arr) != len(anchor_sorted):
        raise ValueError("source and anchor lengths differ")
    if len(source_arr) == 1:
        return np.array([anchor_sorted[0]], dtype=float)
    ranks = pd.Series(source_arr).rank(method="average").to_numpy(dtype=float) - 1.0
    return cap_prob(np.interp(ranks, np.arange(len(anchor_sorted), dtype=float), anchor_sorted))


def blend_to_target_mad(anchor: np.ndarray | pd.Series, target: np.ndarray | pd.Series, *, target_mad: float) -> np.ndarray:
    if not np.isfinite(target_mad) or target_mad < 0.0:
        raise ValueError(f"target_mad must be finite and non-negative, got {target_mad}")
    anchor_arr = cap_prob(anchor)
    target_arr = cap_prob(target)
    if len(anchor_arr) != len(target_arr):
        raise ValueError("anchor and target lengths differ")
    delta = target_arr - anchor_arr
    full_mad = float(np.mean(np.abs(delta)))
    if full_mad == 0.0 or target_mad == 0.0:
        return anchor_arr.copy()
    scale = min(1.0, float(target_mad) / full_mad)
    return cap_prob(anchor_arr + scale * delta)


def build_value_state_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if "rally_uid" not in frame.columns:
        raise ValueError("frame must contain rally_uid")
    df = frame.copy()
    sort_cols = [col for col in ["rally_uid", "strikeNumber"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    out = pd.DataFrame(index=df.index)
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"serverGetPoint"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    for col in numeric_cols:
        out[f"raw_{col}"] = pd.to_numeric(df[col], errors="coerce")

    strike = pd.to_numeric(df.get("strikeNumber", pd.Series(1, index=df.index)), errors="coerce").fillna(1)
    score_self = pd.to_numeric(df.get("scoreSelf", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    score_other = pd.to_numeric(df.get("scoreOther", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
    action = pd.to_numeric(df.get("actionId", pd.Series(-1, index=df.index)), errors="coerce").fillna(-1)
    point = pd.to_numeric(df.get("pointId", pd.Series(-1, index=df.index)), errors="coerce").fillna(-1)

    out["prefix_len"] = strike
    out["log_prefix_len"] = np.log1p(strike)
    out["score_total"] = score_self + score_other
    out["score_margin"] = score_self - score_other
    out["abs_score_margin"] = np.abs(out["score_margin"])
    out["is_deuce_like"] = ((score_self >= 10) & (score_other >= 10)).astype(int)
    out["server_score_ahead"] = (out["score_margin"] > 0).astype(int)
    out["phase_code"] = np.select(
        [strike <= 1, strike <= 3, strike <= 6],
        [0, 1, 2],
        default=3,
    )
    out["anchor_actionId"] = action
    out["anchor_pointId"] = point
    out["is_terminal_anchor_point"] = point.isin([0, 7, 8, 9]).astype(int)

    group = df["rally_uid"]
    for col in ["actionId", "pointId", "positionId", "strengthId", "spinId"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").fillna(-1)
            out[f"lag1_{col}"] = values.groupby(group).shift(1).fillna(-1)
            out[f"lag2_{col}"] = values.groupby(group).shift(2).fillna(-1)

    out = out.replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    columns = list(out.columns)
    return out, columns


def _fit_oof_and_test_predictions(
    x: pd.DataFrame,
    y: np.ndarray,
    x_test: pd.DataFrame,
    *,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"sklearn unavailable: {exc}") from exc

    if len(np.unique(y)) != 2:
        raise ValueError("server target is not binary")
    folds = min(5, int(np.bincount(y).min()))
    if folds < 2:
        raise ValueError("not enough class support for OOF")

    oof = np.zeros(len(y), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    for train_idx, valid_idx in splitter.split(x, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=600,
                class_weight="balanced",
                solver="liblinear",
                random_state=random_state,
            ),
        )
        model.fit(x.iloc[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(x.iloc[valid_idx])[:, 1]

    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=600,
            class_weight="balanced",
            solver="liblinear",
            random_state=random_state,
        ),
    )
    final.fit(x, y)
    test_pred = cap_prob(final.predict_proba(x_test)[:, 1])
    auc = float(roc_auc_score(y, oof))
    return cap_prob(oof), test_pred, auc, "LogisticRegression"


def train_value_state_model() -> tuple[np.ndarray, dict[str, Any]]:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("train.csv or test_new.csv missing")
    no_banned_path_guard([TRAIN_PATH, TEST_PATH])
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    if "serverGetPoint" not in train.columns:
        raise ValueError("train.csv missing serverGetPoint")

    y = train["serverGetPoint"].astype(int).to_numpy()
    base_common = [
        col
        for col in train.columns
        if col in test.columns
        and col not in {"rally_uid", "serverGetPoint"}
        and pd.api.types.is_numeric_dtype(train[col])
    ]
    if not base_common:
        raise ValueError("no common numeric baseline features")

    x_base = train[base_common].replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    x_test_base = test[base_common].replace([np.inf, -np.inf], np.nan).fillna(-1.0)
    baseline_oof, _, baseline_auc, baseline_model = _fit_oof_and_test_predictions(
        x_base,
        y,
        x_test_base,
        random_state=3190,
    )

    value_train, value_cols = build_value_state_features(train)
    value_test, _ = build_value_state_features(test)
    value_test = value_test.reindex(columns=value_cols, fill_value=-1.0)
    value_oof, row_test_pred, value_auc, value_model = _fit_oof_and_test_predictions(
        value_train,
        y,
        value_test,
        random_state=3191,
    )

    submission_uids = pd.read_csv(ANCHOR_PATH, usecols=["rally_uid"])["rally_uid"].astype(int)
    test_pred = pd.DataFrame({"rally_uid": test["rally_uid"].astype(int), "server": row_test_pred})
    server_by_uid = test_pred.groupby("rally_uid", sort=False)["server"].mean()
    if not submission_uids.isin(server_by_uid.index).all():
        missing = submission_uids[~submission_uids.isin(server_by_uid.index)].head(5).tolist()
        raise ValueError(f"test predictions missing submission rally_uid values: {missing}")
    submission_signal = cap_prob(server_by_uid.reindex(submission_uids).to_numpy(dtype=float))

    diagnostics = {
        "available": True,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "baseline_model": baseline_model,
        "value_model": value_model,
        "baseline_feature_count": int(len(base_common)),
        "value_feature_count": int(len(value_cols)),
        "anchor_auc_oof": baseline_auc,
        "value_state_auc_oof": value_auc,
        "value_state_auc_delta_vs_anchor": value_auc - baseline_auc,
        "oof_corr_value_vs_baseline": corr(value_oof, baseline_oof),
        "feature_columns": value_cols,
    }
    return submission_signal, diagnostics


def _v300_proxy_auc() -> float:
    if not V300_SEARCH_PATH.exists():
        return float("nan")
    search = pd.read_csv(V300_SEARCH_PATH)
    values = pd.to_numeric(search.get("proxy_local_auc", pd.Series(dtype=float)), errors="coerce")
    finite = values[np.isfinite(values)]
    return float(finite.max()) if len(finite) else float("nan")


def build_ranktiny_signal(anchor_server: np.ndarray, value_signal: np.ndarray) -> np.ndarray:
    sources: list[np.ndarray] = [rank_normalize_to_anchor(value_signal, anchor_server)]
    if V300_RANK_SOURCE_PATH.exists():
        source_df = load_submission(V300_RANK_SOURCE_PATH)
        if len(source_df) == len(anchor_server):
            sources.append(rank_normalize_to_anchor(source_df["serverGetPoint"].to_numpy(dtype=float), anchor_server))
    target = cap_prob(np.mean(np.column_stack(sources), axis=1))
    return blend_to_target_mad(anchor_server, target, target_mad=0.002)


def build_candidates(anchor: pd.DataFrame, value_signal: np.ndarray, diagnostics: dict[str, Any]) -> list[ServerCandidate]:
    anchor_server = cap_prob(anchor["serverGetPoint"].to_numpy(dtype=float))
    ranked_value = rank_normalize_to_anchor(value_signal, anchor_server)
    anchor_auc = float(diagnostics["anchor_auc_oof"])
    value_auc = float(diagnostics["value_state_auc_oof"])
    return [
        ServerCandidate(
            key="server_ranktiny",
            server=build_ranktiny_signal(anchor_server, value_signal),
            source="value_state_rank + v300_rankavg_w0p02_rank",
            oof_auc=max(value_auc, _v300_proxy_auc()),
            anchor_auc=anchor_auc,
            target_mad=0.002,
        ),
        ServerCandidate(
            key="server_valueblend_mad0p002",
            server=blend_to_target_mad(anchor_server, ranked_value, target_mad=0.002),
            source="value_state_rank",
            oof_auc=value_auc,
            anchor_auc=anchor_auc,
            target_mad=0.002,
        ),
        ServerCandidate(
            key="server_valueblend_mad0p005",
            server=blend_to_target_mad(anchor_server, ranked_value, target_mad=0.005),
            source="value_state_rank",
            oof_auc=value_auc,
            anchor_auc=anchor_auc,
            target_mad=0.005,
        ),
    ]


def build_packaged_submission(
    anchor: pd.DataFrame,
    candidate: ServerCandidate,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> pd.DataFrame:
    validate_submission_frame(anchor, expected_rows=expected_rows)
    if len(anchor) != len(candidate.server):
        raise ValueError("anchor and candidate server lengths differ")
    out = anchor.copy()
    out["serverGetPoint"] = cap_prob(candidate.server)
    out = out.loc[:, SUBMISSION_COLUMNS]
    validate_submission_frame(out, expected_rows=expected_rows)
    return out


def decision_for_candidate(
    *,
    oof_auc: float,
    anchor_auc: float,
    mad: float,
    server_min: float,
    server_max: float,
) -> str:
    sane_distribution = (
        np.isfinite(server_min)
        and np.isfinite(server_max)
        and 0.0 <= server_min < server_max <= 1.0
        and (server_max - server_min) >= 0.01
    )
    if (
        np.isfinite(oof_auc)
        and np.isfinite(anchor_auc)
        and oof_auc > anchor_auc
        and np.isfinite(mad)
        and mad <= 0.0100000001
        and sane_distribution
    ):
        return "REVIEW_SERVER"
    return "DIAGNOSTIC"


def summarize_candidate(
    candidate: ServerCandidate,
    anchor: pd.DataFrame,
    packaged: pd.DataFrame,
    output_path: str,
) -> dict[str, Any]:
    validate_submission_frame(anchor, expected_rows=len(anchor))
    validate_submission_frame(packaged, expected_rows=len(anchor))
    anchor_server = cap_prob(anchor["serverGetPoint"].to_numpy(dtype=float))
    server = cap_prob(packaged["serverGetPoint"].to_numpy(dtype=float))
    mad = float(np.mean(np.abs(server - anchor_server)))
    server_min = float(np.min(server))
    server_max = float(np.max(server))
    return {
        "candidate": Path(output_path).name,
        "path": output_path,
        "source": candidate.source,
        "target_mad": float(candidate.target_mad),
        "oof_auc": float(candidate.oof_auc),
        "anchor_auc": float(candidate.anchor_auc),
        "oof_auc_delta_vs_anchor": float(candidate.oof_auc - candidate.anchor_auc),
        "server_mad_vs_v306_server": mad,
        "server_corr_vs_v306_server": corr(server, anchor_server),
        "server_min": server_min,
        "server_max": server_max,
        "server_mean": float(np.mean(server)),
        "server_std": float(np.std(server)),
        "row_count": int(len(packaged)),
        "action_changed_rows_vs_anchor": int(
            np.sum(packaged["actionId"].astype(int).to_numpy() != anchor["actionId"].astype(int).to_numpy())
        ),
        "point_changed_rows_vs_anchor": int(
            np.sum(packaged["pointId"].astype(int).to_numpy() != anchor["pointId"].astype(int).to_numpy())
        ),
        "decision": decision_for_candidate(
            oof_auc=candidate.oof_auc,
            anchor_auc=candidate.anchor_auc,
            mad=mad,
            server_min=server_min,
            server_max=server_max,
        ),
    }


def output_filename(candidate: ServerCandidate) -> str:
    names = {
        "server_ranktiny": RANKTINY_NAME,
        "server_valueblend_mad0p002": MAD002_NAME,
        "server_valueblend_mad0p005": MAD005_NAME,
    }
    return names[candidate.key]


def write_submission(path: Path, df: pd.DataFrame) -> None:
    no_banned_path_guard([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_submission_frame(df, expected_rows=len(df))
    df.loc[:, SUBMISSION_COLUMNS].to_csv(path, index=False, float_format="%.8f")


def write_reports(search: pd.DataFrame, summary: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# V319 Clean Server Value State",
        "",
        "Server-only research around fixed V173 action and V306 point anchor.",
        "",
        f"- Generated submissions: `{summary['generated_submission_count']}`",
        f"- Review server candidates: `{summary['review_server_count']}`",
        f"- Value/state OOF AUC: `{summary['model_diagnostics'].get('value_state_auc_oof')}`",
        f"- Anchor OOF AUC: `{summary['model_diagnostics'].get('anchor_auc_oof')}`",
        "",
        "## Candidates",
        "",
        "| candidate | source | OOF AUC | delta | MAD | corr | min | max | action | point | decision |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in search.to_dict("records"):
        lines.append(
            f"| `{row['candidate']}` | `{row['source']}` | "
            f"{float(row['oof_auc']):.6f} | {float(row['oof_auc_delta_vs_anchor']):.6f} | "
            f"{float(row['server_mad_vs_v306_server']):.8f} | "
            f"{float(row['server_corr_vs_v306_server']):.8f} | "
            f"{float(row['server_min']):.8f} | {float(row['server_max']):.8f} | "
            f"{int(row['action_changed_rows_vs_anchor'])} | {int(row['point_changed_rows_vs_anchor'])} | "
            f"`{row['decision']}` |"
        )
    lines.extend(["", f"Search CSV: `{relative_path(SEARCH_PATH)}`", ""])
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    no_banned_path_guard([ANCHOR_PATH, V300_SEARCH_PATH, V300_RANK_SOURCE_PATH, TRAIN_PATH, TEST_PATH, OUT_DIR])

    anchor = load_submission(ANCHOR_PATH)
    value_signal, diagnostics = train_value_state_model()
    candidates = build_candidates(anchor, value_signal, diagnostics)

    rows: list[dict[str, Any]] = []
    generated: list[str] = []
    for candidate in candidates:
        packaged = build_packaged_submission(anchor, candidate)
        out_path = OUT_DIR / output_filename(candidate)
        write_submission(out_path, packaged)
        out_text = relative_path(out_path)
        generated.append(out_text)
        rows.append(summarize_candidate(candidate, anchor, packaged, out_text))

    search = pd.DataFrame(rows)
    search["decision_rank"] = search["decision"].map({"REVIEW_SERVER": 0, "DIAGNOSTIC": 1}).fillna(9)
    search = search.sort_values(
        ["decision_rank", "oof_auc_delta_vs_anchor", "server_mad_vs_v306_server"],
        ascending=[True, False, True],
    ).drop(columns=["decision_rank"])
    search.to_csv(SEARCH_PATH, index=False)
    review = search[search["decision"].eq("REVIEW_SERVER")]

    summary = {
        "version": "V319",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": relative_path(OUT_DIR),
        "anchor_path": relative_path(ANCHOR_PATH),
        "policy": {
            "server_only": True,
            "fixed_action": "V173 inherited from V306 anchor",
            "fixed_point": "V306 p0 cap0p01",
            "no_ttmatch": True,
            "no_old_server": True,
            "no_upload_copy": True,
        },
        "model_diagnostics": diagnostics,
        "generated_submissions": generated,
        "generated_submission_count": len(generated),
        "search_path": relative_path(SEARCH_PATH),
        "report_json_path": relative_path(REPORT_JSON_PATH),
        "report_md_path": relative_path(REPORT_MD_PATH),
        "review_server_count": int(len(review)),
        "top_candidates": review.head(10).to_dict(orient="records"),
        "decision_rule": "REVIEW_SERVER if no action/point changes, server MAD <= 0.01, sane server distribution, and OOF AUC improves over the clean baseline feature anchor.",
    }
    write_reports(search, summary)
    return summary


def main() -> None:
    summary = run_pipeline()
    print(
        json.dumps(
            {
                "outdir": summary["outdir"],
                "generated_submissions": summary["generated_submission_count"],
                "review_server_count": summary["review_server_count"],
                "anchor_auc_oof": summary["model_diagnostics"]["anchor_auc_oof"],
                "value_state_auc_oof": summary["model_diagnostics"]["value_state_auc_oof"],
                "top": [row["candidate"] for row in summary["top_candidates"][:5]],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""V314 clean server value research around V306/V307 point anchors.

Fixed action is inherited from the V306/V307 V173-action point anchors.
Server candidates are clean V300/V302 reuse branches plus a small optional
prefix-feature server model. Outputs stay under v314_clean_server_value_research.
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
OUT_DIR = ROOT / "v314_clean_server_value_research"

V306_ANCHOR_PATH = ROOT / "v306_point0_addition_probe" / "submission_v306_p0_cap0p01__v173action_v300server.csv"
POINT_ANCHORS = {
    "v306_p0_cap0p01": V306_ANCHOR_PATH,
    "v307_p0_budget24": ROOT / "v307_point0_dose_extension" / "submission_v307_p0_budget24__v173action_v300server.csv",
    "v307_p0_cap0p02": ROOT / "v307_point0_dose_extension" / "submission_v307_p0_cap0p02__v173action_v300server.csv",
}

V300_DIR = ROOT / "v300_clean_server_blend_recycler"
V302_DIR = ROOT / "v302_clean_server_calibration_sweep"
V300_SEARCH = V300_DIR / "v300_server_search.csv"
V302_SEARCH = V302_DIR / "v302_server_search.csv"
TRAIN_PATH = ROOT / "train.csv"
TEST_PATH = ROOT / "test_new.csv"

SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
EXPECTED_ROWS = 1845
BANNED_PATH_TOKENS = ("TTMATCH", "OLD_SERVER", "OLDSERVER")
SEARCH_PATH = OUT_DIR / "v314_server_point_anchor_search.csv"
REPORT_JSON_PATH = OUT_DIR / "v314_report.json"
REPORT_MD_PATH = OUT_DIR / "v314_report.md"


@dataclass(frozen=True)
class ServerVariant:
    key: str
    family: str
    source_path: str
    server: np.ndarray
    source_is_clean: bool
    server_auc_oof: float
    risk_hint: str


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


def source_is_clean(path: Path | str) -> bool:
    upper = str(path).upper()
    return not any(token in upper for token in BANNED_PATH_TOKENS)


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


def shrink_to_anchor(server: np.ndarray, anchor: np.ndarray, *, strength: float) -> np.ndarray:
    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"strength must be in [0, 1], got {strength}")
    if len(server) != len(anchor):
        raise ValueError("server and anchor lengths differ")
    return cap_prob((1.0 - strength) * cap_prob(server) + strength * cap_prob(anchor))


def apply_temperature(server: np.ndarray, *, temperature: float) -> np.ndarray:
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError(f"temperature must be positive and finite, got {temperature}")
    p = cap_prob(server)
    logits = np.log(p) - np.log1p(-p)
    return cap_prob(1.0 / (1.0 + np.exp(-(logits / float(temperature)))))


def rank_normalize_to_anchor(source: np.ndarray, anchor: np.ndarray) -> np.ndarray:
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


def decision_for_variant(*, source_is_clean: bool, mad: float, server_min: float, server_max: float) -> str:
    sensible_distribution = (
        np.isfinite(server_min)
        and np.isfinite(server_max)
        and 0.0 <= server_min < server_max <= 1.0
        and (server_max - server_min) >= 0.01
    )
    if source_is_clean and np.isfinite(mad) and mad <= 0.0100000001 and sensible_distribution:
        return "REVIEW_SERVER"
    return "DIAGNOSTIC"


def build_packaged_submission(
    point_anchor: pd.DataFrame,
    variant: ServerVariant,
    *,
    expected_rows: int = EXPECTED_ROWS,
) -> pd.DataFrame:
    validate_submission_frame(point_anchor, expected_rows=expected_rows)
    if len(point_anchor) != len(variant.server):
        raise ValueError("point anchor and server variant length mismatch")
    out = point_anchor.copy()
    out["serverGetPoint"] = cap_prob(variant.server)
    out = out.loc[:, SUBMISSION_COLUMNS]
    validate_submission_frame(out, expected_rows=expected_rows)
    return out


def summarize_combination(
    *,
    point_key: str,
    point_path: str,
    v306_anchor: pd.DataFrame,
    point_anchor: pd.DataFrame,
    variant: ServerVariant,
    packaged: pd.DataFrame,
    output_path: str,
) -> dict[str, Any]:
    validate_submission_frame(v306_anchor, expected_rows=len(v306_anchor))
    validate_submission_frame(point_anchor, expected_rows=len(point_anchor))
    validate_submission_frame(packaged, expected_rows=len(packaged))
    server = cap_prob(packaged["serverGetPoint"].to_numpy(dtype=float))
    current = cap_prob(v306_anchor["serverGetPoint"].to_numpy(dtype=float))
    mad = float(np.mean(np.abs(server - current)))
    server_min = float(np.min(server))
    server_max = float(np.max(server))
    return {
        "candidate": Path(output_path).name,
        "path": output_path,
        "point_anchor": point_key,
        "point_anchor_path": point_path,
        "server_source": variant.key,
        "server_source_path": variant.source_path,
        "source_family": variant.family,
        "source_is_clean": bool(variant.source_is_clean),
        "server_auc_oof": float(variant.server_auc_oof),
        "server_mad_vs_current_v306": mad,
        "server_corr_vs_current_v306": corr(server, current),
        "server_min": server_min,
        "server_max": server_max,
        "server_mean": float(np.mean(server)),
        "server_std": float(np.std(server)),
        "risk": classify_risk(mad, variant.risk_hint),
        "row_count": int(len(packaged)),
        "action_changed_rows_vs_point_anchor": int(
            np.sum(packaged["actionId"].astype(int).to_numpy() != point_anchor["actionId"].astype(int).to_numpy())
        ),
        "point_changed_rows_vs_point_anchor": int(
            np.sum(packaged["pointId"].astype(int).to_numpy() != point_anchor["pointId"].astype(int).to_numpy())
        ),
        "decision": decision_for_variant(
            source_is_clean=variant.source_is_clean,
            mad=mad,
            server_min=server_min,
            server_max=server_max,
        ),
    }


def classify_risk(mad: float, hint: str) -> str:
    if hint and hint not in {"nan", "None"}:
        return str(hint)
    if mad <= 0.002:
        return "safe"
    if mad <= 0.01:
        return "review_band"
    return "diagnostic"


def metric_value(row: pd.Series, names: list[str]) -> float:
    for name in names:
        if name in row and pd.notna(row[name]):
            return float(pd.to_numeric(row[name], errors="coerce"))
    return float("nan")


def search_metadata(search_path: Path) -> dict[str, dict[str, Any]]:
    if not search_path.exists():
        return {}
    search = pd.read_csv(search_path)
    meta: dict[str, dict[str, Any]] = {}
    for _, row in search.iterrows():
        candidate = str(row.get("candidate", ""))
        meta[candidate] = {
            "server_auc_oof": metric_value(row, ["server_auc_oof", "server_auc", "proxy_local_auc"]),
            "risk_hint": str(row.get("risk_tier", row.get("risk", row.get("verdict", "")))),
        }
    return meta


def variant_from_submission(path: Path, key: str, family: str, meta: dict[str, Any]) -> ServerVariant | None:
    if not path.exists():
        return None
    df = load_submission(path)
    return ServerVariant(
        key=key,
        family=family,
        source_path=relative_path(path),
        server=cap_prob(df["serverGetPoint"].to_numpy(dtype=float)),
        source_is_clean=source_is_clean(path),
        server_auc_oof=float(meta.get("server_auc_oof", float("nan"))),
        risk_hint=str(meta.get("risk_hint", "")),
    )


def discover_reuse_variants() -> list[ServerVariant]:
    variants: list[ServerVariant] = []
    seen: set[bytes] = set()
    v300_meta = search_metadata(V300_SEARCH)
    v302_meta = search_metadata(V302_SEARCH)

    def add(variant: ServerVariant | None) -> None:
        if variant is None:
            return
        no_banned_path_guard([variant.source_path])
        fp = np.round(cap_prob(variant.server), 10).tobytes()
        if fp in seen:
            return
        variants.append(variant)
        seen.add(fp)

    for path in sorted(V300_DIR.glob("submission_v300_*.csv")):
        key = path.stem.replace("submission_", "")
        add(variant_from_submission(path, key, "v300", v300_meta.get(path.name, {})))

    for path in sorted(V302_DIR.glob("submission_v302_*.csv")):
        key = path.stem.replace("submission_", "")
        add(variant_from_submission(path, key, "v302", v302_meta.get(path.name, {})))

    if not variants:
        raise ValueError("No clean V300/V302 server reuse variants found.")
    return variants


def train_prefix_server_variant(anchor_server: np.ndarray) -> tuple[list[ServerVariant], dict[str, Any]]:
    diagnostics: dict[str, Any] = {"attempted": True, "available": False}
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        diagnostics["reason"] = "train.csv or test_new.csv missing"
        return [], diagnostics
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import StratifiedKFold
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:  # pragma: no cover - environment dependent
        diagnostics["reason"] = f"sklearn unavailable: {exc}"
        return [], diagnostics

    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    common = [col for col in train.columns if col in test.columns]
    feature_cols = [
        col
        for col in common
        if col not in {"rally_uid", "serverGetPoint"} and pd.api.types.is_numeric_dtype(train[col])
    ]
    if "serverGetPoint" not in train.columns or not feature_cols:
        diagnostics["reason"] = "server target or numeric common features missing"
        return [], diagnostics

    y = train["serverGetPoint"].astype(int).to_numpy()
    if len(np.unique(y)) != 2:
        diagnostics["reason"] = "server target is not binary"
        return [], diagnostics
    x = train[feature_cols].fillna(-1).to_numpy(dtype=float)
    x_test = test[feature_cols].fillna(-1).to_numpy(dtype=float)
    folds = min(5, int(np.bincount(y).min()))
    if folds < 2:
        diagnostics["reason"] = "not enough class support for OOF"
        return [], diagnostics

    oof = np.zeros(len(train), dtype=float)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=314)
    for train_idx, valid_idx in splitter.split(x, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"),
        )
        model.fit(x[train_idx], y[train_idx])
        oof[valid_idx] = model.predict_proba(x[valid_idx])[:, 1]

    auc = float(roc_auc_score(y, oof))
    final = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=500, class_weight="balanced", solver="lbfgs"),
    )
    final.fit(x, y)
    test_row_pred = cap_prob(final.predict_proba(x_test)[:, 1])
    test_pred = pd.DataFrame({"rally_uid": test["rally_uid"].astype(int), "server": test_row_pred})
    submission_uids = pd.read_csv(V306_ANCHOR_PATH, usecols=["rally_uid"])["rally_uid"].astype(int)
    server_by_uid = test_pred.groupby("rally_uid", sort=False)["server"].mean()
    if not submission_uids.isin(server_by_uid.index).all():
        missing = submission_uids[~submission_uids.isin(server_by_uid.index)].head(5).tolist()
        diagnostics["reason"] = f"test predictions missing submission rally_uid values: {missing}"
        return [], diagnostics
    model_server = cap_prob(server_by_uid.reindex(submission_uids).to_numpy(dtype=float))
    if len(model_server) != len(anchor_server):
        diagnostics["reason"] = f"aggregated test rows {len(model_server)} do not match submission rows {len(anchor_server)}"
        return [], diagnostics
    ranked = rank_normalize_to_anchor(model_server, anchor_server)
    shrunk = shrink_to_anchor(ranked, anchor_server, strength=0.75)

    diagnostics.update(
        {
            "available": True,
            "server_auc_oof": auc,
            "feature_count": len(feature_cols),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
        }
    )
    return [
        ServerVariant(
            key="v314_prefix_logreg_rank_to_v306",
            family="v314_prefix_model",
            source_path=relative_path(TRAIN_PATH) + "+" + relative_path(TEST_PATH),
            server=ranked,
            source_is_clean=True,
            server_auc_oof=auc,
            risk_hint="model_ranked",
        ),
        ServerVariant(
            key="v314_prefix_logreg_rank_shrink0p75_to_v306",
            family="v314_prefix_model",
            source_path=relative_path(TRAIN_PATH) + "+" + relative_path(TEST_PATH),
            server=shrunk,
            source_is_clean=True,
            server_auc_oof=auc,
            risk_hint="model_shrunk",
        ),
    ], diagnostics


def add_calibration_variants(variants: list[ServerVariant], anchor_server: np.ndarray) -> list[ServerVariant]:
    out = list(variants)
    base = variants[0]
    for temperature in (0.95, 1.05):
        token = str(temperature).replace(".", "p")
        out.append(
            ServerVariant(
                key=f"v314_temperature_t_{token}_from_{base.key}",
                family="v314_calibration",
                source_path=base.source_path,
                server=apply_temperature(base.server, temperature=temperature),
                source_is_clean=base.source_is_clean,
                server_auc_oof=base.server_auc_oof,
                risk_hint="temperature",
            )
        )
    for strength in (0.25, 0.50, 0.75):
        token = str(strength).replace(".", "p")
        out.append(
            ServerVariant(
                key=f"v314_shrink_s_{token}_from_{base.key}",
                family="v314_calibration",
                source_path=base.source_path,
                server=shrink_to_anchor(base.server, anchor_server, strength=strength),
                source_is_clean=base.source_is_clean,
                server_auc_oof=base.server_auc_oof,
                risk_hint="shrink",
            )
        )
    out.append(
        ServerVariant(
            key=f"v314_rank_to_v306_from_{base.key}",
            family="v314_calibration",
            source_path=base.source_path,
            server=rank_normalize_to_anchor(base.server, anchor_server),
            source_is_clean=base.source_is_clean,
            server_auc_oof=base.server_auc_oof,
            risk_hint="rank",
        )
    )
    return out


def output_filename(point_key: str, variant: ServerVariant) -> str:
    key = variant.key.replace(".", "p").replace("-", "_").replace("/", "_")
    return f"submission_v314_{point_key}__{key}.csv"


def write_submission(path: Path, df: pd.DataFrame) -> None:
    no_banned_path_guard([path])
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_submission_frame(df, expected_rows=len(df))
    df.loc[:, SUBMISSION_COLUMNS].to_csv(path, index=False, float_format="%.8f")


def write_reports(search: pd.DataFrame, summary: dict[str, Any]) -> None:
    REPORT_JSON_PATH.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# V314 Clean Server Value Research",
        "",
        "Fixed V173 action via point anchors; clean server-only variants around V306/V307 anchors.",
        "",
        f"- Generated submissions: `{summary['generated_submission_count']}`",
        f"- Review server combinations: `{summary['review_server_count']}`",
        f"- Prefix model available: `{summary['prefix_model_diagnostics'].get('available')}`",
        "",
        "## Top Combinations",
        "",
        "| point | server | AUC OOF | MAD vs V306 | corr | min | max | risk | decision |",
        "|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in search.head(20).to_dict("records"):
        auc = row["server_auc_oof"]
        auc_text = "" if pd.isna(auc) else f"{float(auc):.6f}"
        lines.append(
            f"| `{row['point_anchor']}` | `{row['server_source']}` | {auc_text} | "
            f"{float(row['server_mad_vs_current_v306']):.8f} | "
            f"{float(row['server_corr_vs_current_v306']):.8f} | "
            f"{float(row['server_min']):.8f} | {float(row['server_max']):.8f} | "
            f"`{row['risk']}` | `{row['decision']}` |"
        )
    lines.extend(["", f"Search CSV: `{relative_path(SEARCH_PATH)}`", ""])
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    no_banned_path_guard([*POINT_ANCHORS.values(), V300_DIR, V302_DIR])

    v306_anchor = load_submission(V306_ANCHOR_PATH)
    point_anchors = {
        key: load_submission(path)
        for key, path in POINT_ANCHORS.items()
        if path.exists()
    }
    if "v306_p0_cap0p01" not in point_anchors:
        raise FileNotFoundError(V306_ANCHOR_PATH)

    anchor_server = cap_prob(v306_anchor["serverGetPoint"].to_numpy(dtype=float))
    reuse_variants = discover_reuse_variants()
    model_variants, model_diagnostics = train_prefix_server_variant(anchor_server)
    variants = add_calibration_variants(reuse_variants + model_variants, anchor_server)

    rows: list[dict[str, Any]] = []
    generated: list[str] = []
    for point_key, point_anchor in point_anchors.items():
        point_path = POINT_ANCHORS[point_key]
        if not point_anchor["rally_uid"].equals(v306_anchor["rally_uid"]):
            raise ValueError(f"{point_key} rally_uid does not match V306 anchor")
        for variant in variants:
            packaged = build_packaged_submission(point_anchor, variant)
            out_path = OUT_DIR / output_filename(point_key, variant)
            write_submission(out_path, packaged)
            out_text = relative_path(out_path)
            generated.append(out_text)
            rows.append(
                summarize_combination(
                    point_key=point_key,
                    point_path=relative_path(point_path),
                    v306_anchor=v306_anchor,
                    point_anchor=point_anchor,
                    variant=variant,
                    packaged=packaged,
                    output_path=out_text,
                )
            )

    search = pd.DataFrame(rows)
    search["decision_rank"] = search["decision"].map({"REVIEW_SERVER": 0, "DIAGNOSTIC": 1}).fillna(9)
    search = search.sort_values(
        ["decision_rank", "server_mad_vs_current_v306", "point_anchor", "server_source"],
        ascending=[True, True, True, True],
    ).drop(columns=["decision_rank"])
    search.to_csv(SEARCH_PATH, index=False)

    review = search[search["decision"].eq("REVIEW_SERVER")]
    summary = {
        "version": "V314",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "outdir": relative_path(OUT_DIR),
        "policy": {
            "fixed_action": "V173 inherited from V306/V307 anchors",
            "point_anchors": list(point_anchors.keys()),
            "clean_sources_only": True,
            "no_old_server": True,
            "no_ttmatch": True,
            "no_upload_copy": True,
        },
        "v306_anchor_path": relative_path(V306_ANCHOR_PATH),
        "generated_submissions": generated,
        "generated_submission_count": len(generated),
        "server_variant_count": len(variants),
        "point_anchor_count": len(point_anchors),
        "search_path": relative_path(SEARCH_PATH),
        "report_json_path": relative_path(REPORT_JSON_PATH),
        "report_md_path": relative_path(REPORT_MD_PATH),
        "prefix_model_diagnostics": model_diagnostics,
        "review_server_count": int(len(review)),
        "top_server_point_combinations": review.head(20).to_dict(orient="records"),
        "decision_rule": "REVIEW_SERVER if source is clean, server MAD vs current V306 <= 0.01, and server distribution is non-degenerate; otherwise DIAGNOSTIC.",
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
                "top": [
                    [row["point_anchor"], row["server_source"]]
                    for row in summary["top_server_point_combinations"][:10]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

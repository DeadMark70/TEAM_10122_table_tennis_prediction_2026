"""R157/R158 conservative physics pretraining smoke probes.

R157: Masked Physics Modeling proxy.
  Estimate external conditional distributions such as
  P(canonical_grid | speed_bin, spin_bin).  This is a lightweight proxy for a
  masked physics reconstruction head; it measures whether external physics
  attributes reconstruct landing-grid priors that help AICUP point prediction.

R158: Physics-informed contrastive/prototype proxy.
  Build canonical external state prototypes and retrieve a landing-grid prior
  from AI CUP prefix proxy states.  This is a conservative nearest-prototype
  substitute for full contrastive learning.

Both probes exclude TTMATCH, do not use test labels, preserve pointId=0, and
apply only low-weight non-terminal point residuals.
"""

from __future__ import annotations

import json
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score

ROOT_DIR = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent
for p in [ROOT_DIR, ANALYSIS_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from analysis_r151b_r154_physics_prior_integration import blend_nonterminal_point, normalize_nonzero_prior, point_pred  # noqa: E402
from analysis_r67_r70_meta_priors import align_prefix_meta, prepare_prefix_features  # noqa: E402
from baseline_lgbm import POINT_CLASSES  # noqa: E402
from generate_r42_golden_soft_blends import normalize_rows  # noqa: E402


OUTDIR = Path("r157_r158_physics_pretraining_smoke")
UPLOAD_DIR = Path("upload_candidates_20260519")
SELECTED_DIR = Path("submissions/selected")
R151_DIR = Path("r151_safe_physics_priors")
R67_ANCHOR = UPLOAD_DIR / "submission_r67_r63_blend_w0p2_current_point_server.csv"
R142_SERVER = UPLOAD_DIR / "submission_r142_r67_anchor_oldsharpen005095.csv"
R143_SERVER = UPLOAD_DIR / "submission_r143_r67_anchor_oldsharpen005095_newscore_gapcal.csv"
R101_OOF = Path("r101_r103_destiny_gru/oof_proba_r101_r103.pkl")
R101_TEST = Path("r101_r103_destiny_gru/test_proba_r101_r103.pkl")
R111_OOF = Path("r111_remaining_moe_gru/oof_proba_r111.pkl")
R111_TEST = Path("r111_remaining_moe_gru/test_proba_r111.pkl")


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


GRID_TO_POINT_DIRECT = {
    "left_near": 1,
    "middle_near": 2,
    "right_near": 3,
    "left_mid": 4,
    "middle_mid": 5,
    "right_mid": 6,
    "left_far": 7,
    "middle_far": 8,
    "right_far": 9,
}
GRID_TO_POINT_MIRROR = {
    "left_near": 3,
    "middle_near": 2,
    "right_near": 1,
    "left_mid": 6,
    "middle_mid": 5,
    "right_mid": 4,
    "left_far": 9,
    "middle_far": 8,
    "right_far": 7,
}


def load_pickle(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def clean_float(x: float) -> str:
    return str(float(x)).replace(".", "p").replace("-", "m")


def point_to_grid(point: int, mirror: bool) -> str:
    if point <= 0:
        return "unknown"
    inv = GRID_TO_POINT_MIRROR if mirror else GRID_TO_POINT_DIRECT
    for grid, p in inv.items():
        if p == point:
            return grid
    return "unknown"


def grid_to_point_dist(counter: Counter, mirror: bool, alpha: float = 1.0) -> np.ndarray:
    mapping = GRID_TO_POINT_MIRROR if mirror else GRID_TO_POINT_DIRECT
    vals = np.zeros(9, dtype=float) + alpha
    for grid, cnt in counter.items():
        p = mapping.get(str(grid))
        if p is not None:
            vals[p - 1] += float(cnt)
    vals = np.clip(vals, 1e-9, None)
    return vals / vals.sum()


def strength_to_speed_bin(x: int) -> str:
    # AI CUP convention appears to be 1=strong, 2=medium, 3=weak.
    if x == 1:
        return "fast"
    if x == 2:
        return "medium"
    if x == 3:
        return "slow"
    return "unknown"


def spin_to_spin_bin(x: int) -> str:
    if x in {1, 2}:
        return "high"
    if x in {3, 4, 5}:
        return "medium"
    return "unknown"


def point_depth(point: int) -> str:
    if point in {1, 2, 3}:
        return "near"
    if point in {4, 5, 6}:
        return "mid"
    if point in {7, 8, 9}:
        return "far"
    return "unknown"


def speed_ordinal(x: str) -> float:
    return {"very_slow": 0.0, "slow": 1.0, "medium": 2.0, "fast": 3.0, "very_fast": 4.0, "unknown": 2.0}.get(str(x), 2.0)


def spin_ordinal(x: str) -> float:
    return {"low": 0.0, "medium": 1.0, "high": 2.0, "unknown": 1.0}.get(str(x), 1.0)


def depth_ordinal(x: str) -> float:
    return {"near": 0.0, "mid": 1.0, "far": 2.0, "unknown": 1.0}.get(str(x), 1.0)


def side_ordinal(x: str) -> float:
    return {"left": -1.0, "middle": 0.0, "right": 1.0, "unknown": 0.0}.get(str(x), 0.0)


def split_grid(grid: str) -> tuple[str, str]:
    parts = str(grid).split("_")
    if len(parts) >= 2:
        return parts[0], parts[-1]
    return "unknown", "unknown"


def load_external_states() -> tuple[pd.DataFrame, pd.DataFrame]:
    dm = pd.read_csv(R151_DIR / "deepmind_canonical_physics_states.csv")
    md = pd.read_csv(R151_DIR / "tt_matchdynamics_canonical_states.csv")
    return dm, md


def build_mpm_tables(dm: pd.DataFrame, mirror: bool) -> tuple[dict, np.ndarray, pd.DataFrame]:
    global_counter = Counter(dm["canonical_grid_3x3"].astype(str))
    global_prior = grid_to_point_dist(global_counter, mirror, alpha=1.0)
    tables: dict[tuple[str, str], np.ndarray] = {}
    supports: list[dict] = []
    for (speed, spin), group in dm.groupby(["physics_speed_bin", "physics_spin_bin"]):
        counter = Counter(group["canonical_grid_3x3"].astype(str))
        prior = grid_to_point_dist(counter, mirror, alpha=2.0)
        tables[(str(speed), str(spin))] = prior
        supports.append(
            {
                "speed_bin": speed,
                "spin_bin": spin,
                "support": int(len(group)),
                "top_point": int(np.argmax(prior) + 1),
                "top_prob": float(np.max(prior)),
                "entropy": float(-np.sum(prior * np.log(np.clip(prior, 1e-12, 1.0)))),
            }
        )
    return tables, global_prior, pd.DataFrame(supports)


def mpm_prior_for_rows(rows: pd.DataFrame, tables: dict, global_prior: np.ndarray) -> np.ndarray:
    out = np.tile(global_prior.reshape(1, -1), (len(rows), 1))
    for i, row in enumerate(rows.itertuples(index=False)):
        strength = int(getattr(row, "lag0_strengthId", 0))
        spin = int(getattr(row, "lag0_spinId", 0))
        key = (strength_to_speed_bin(strength), spin_to_spin_bin(spin))
        if key in tables:
            out[i] = tables[key]
    return normalize_nonzero_prior(out)


def build_external_prototypes(dm: pd.DataFrame, md: pd.DataFrame, mirror: bool) -> pd.DataFrame:
    rows: list[dict] = []
    mapping = GRID_TO_POINT_MIRROR if mirror else GRID_TO_POINT_DIRECT
    for row in dm.itertuples(index=False):
        grid = str(row.canonical_grid_3x3)
        side, depth = split_grid(grid)
        p = mapping.get(grid)
        if p is None:
            continue
        rows.append(
            {
                "source": "deepmind",
                "speed": speed_ordinal(str(row.physics_speed_bin)),
                "spin": spin_ordinal(str(row.physics_spin_bin)),
                "side": side_ordinal(side),
                "depth": depth_ordinal(depth),
                "point": int(p),
            }
        )
    for row in md.itertuples(index=False):
        grid = str(row.md_grid_3x3)
        side, depth = split_grid(grid)
        p = mapping.get(grid)
        if p is None:
            continue
        spin = 2.0 if str(row.md_spin_family) in {"topspin", "backspin"} else 1.0
        rows.append(
            {
                "source": "tt_matchdynamics",
                "speed": 1.5,
                "spin": spin,
                "side": side_ordinal(side),
                "depth": depth_ordinal(depth),
                "point": int(p),
            }
        )
    return pd.DataFrame(rows)


def row_proxy_vector(row) -> np.ndarray:
    lag_point = int(getattr(row, "lag0_pointId", 0))
    grid = point_to_grid(lag_point, mirror=False)
    side, depth = split_grid(grid)
    return np.array(
        [
            speed_ordinal(strength_to_speed_bin(int(getattr(row, "lag0_strengthId", 0)))),
            spin_ordinal(spin_to_spin_bin(int(getattr(row, "lag0_spinId", 0)))),
            side_ordinal(side),
            depth_ordinal(depth),
        ],
        dtype=float,
    )


def prototype_prior_for_rows(rows: pd.DataFrame, prototypes: pd.DataFrame, tau: float = 0.75, k: int = 250) -> np.ndarray:
    x_ext = prototypes[["speed", "spin", "side", "depth"]].to_numpy(dtype=float)
    y_ext = prototypes["point"].astype(int).to_numpy()
    out = np.zeros((len(rows), 9), dtype=float)
    # Group identical proxy vectors for speed.
    cache: dict[tuple[float, float, float, float], np.ndarray] = {}
    for i, row in enumerate(rows.itertuples(index=False)):
        vec = row_proxy_vector(row)
        key = tuple(vec.tolist())
        if key in cache:
            out[i] = cache[key]
            continue
        d2 = np.sum((x_ext - vec.reshape(1, -1)) ** 2, axis=1)
        idx = np.argpartition(d2, min(k, len(d2) - 1))[:k]
        weights = np.exp(-d2[idx] / max(tau, 1e-6))
        vals = np.zeros(9, dtype=float) + 1.0
        for p, w in zip(y_ext[idx], weights):
            vals[int(p) - 1] += float(w)
        vals = vals / vals.sum()
        cache[key] = vals
        out[i] = vals
    return normalize_nonzero_prior(out)


def eval_point(meta: pd.DataFrame, prob: np.ndarray, base_prob: np.ndarray, tuning: GrUTuning, name: str, extra: dict | None = None) -> dict:
    y = meta["next_pointId"].astype(int).to_numpy()
    pred = point_pred(meta, prob, tuning)
    base = point_pred(meta, base_prob, tuning)
    rec = {
        "candidate": name,
        "point_macro_f1": float(f1_score(y, pred, labels=POINT_CLASSES, average="macro", zero_division=0)),
        "point_churn_vs_base": float(np.mean(pred != base)),
        "changed_rows": int(np.sum(pred != base)),
    }
    report = classification_report(y, pred, labels=POINT_CLASSES, output_dict=True, zero_division=0)
    for k in [0, 5, 7, 8, 9]:
        rec[f"f1_point_{k}"] = float(report[str(k)]["f1-score"])
        rec[f"pred_count_point_{k}"] = int(np.sum(pred == k))
    if extra:
        rec.update(extra)
    return rec


def load_submission(path: Path, rally_uids: np.ndarray | None = None) -> pd.DataFrame:
    sub = pd.read_csv(path)
    if rally_uids is None:
        return sub
    return pd.DataFrame({"rally_uid": rally_uids}).merge(sub, on="rally_uid", how="left", validate="one_to_one")


def write_submission(name: str, template: pd.DataFrame, point_pred_values: np.ndarray, server_values: np.ndarray) -> dict:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)
    sub = template.copy()
    sub["pointId"] = point_pred_values.astype(int)
    sub["serverGetPoint"] = np.round(server_values.astype(float), 8)
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    upload = UPLOAD_DIR / name
    selected = SELECTED_DIR / name
    shutil.copy2(path, upload)
    shutil.copy2(path, selected)
    return {"path": str(path), "upload_path": str(upload), "selected_path": str(selected)}


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    SELECTED_DIR.mkdir(parents=True, exist_ok=True)

    dm, md = load_external_states()
    r101_oof = load_pickle(R101_OOF)
    r101_test = load_pickle(R101_TEST)
    r111_oof = load_pickle(R111_OOF)
    r111_test = load_pickle(R111_TEST)
    tuning: GrUTuning = r111_oof["tuning"]
    meta = r111_oof["valid_meta"].copy().reset_index(drop=True)
    test_meta = r111_test["test_meta"].copy().reset_index(drop=True)

    _train_raw, _test_raw, prefix, test_prefix, _features = prepare_prefix_features()
    meta_for_align = meta.copy()
    if "fold" not in meta_for_align.columns:
        meta_for_align["fold"] = 0
    rows = align_prefix_meta(meta_for_align, prefix).reset_index(drop=True)
    test_rows = test_prefix.reset_index(drop=True)
    if not test_rows["rally_uid"].reset_index(drop=True).equals(test_meta["rally_uid"].reset_index(drop=True)):
        test_rows = pd.DataFrame({"rally_uid": test_meta["rally_uid"].astype(int)}).merge(
            test_prefix,
            on="rally_uid",
            how="left",
            validate="one_to_one",
        )

    base_point_oof = normalize_rows(0.97 * r101_oof["gru_point"] + 0.03 * r111_oof["gru_point"])
    base_point_test = normalize_rows(0.97 * r101_test["gru_point"] + 0.03 * r111_test["gru_point"])

    search_rows: list[dict] = []
    search_rows.append(eval_point(meta, base_point_oof, base_point_oof, tuning, "r101_r111_point_base", {"family": "base"}))

    mpm_reports: list[pd.DataFrame] = []
    candidate_probs_test: dict[str, np.ndarray] = {}
    candidate_scores: dict[str, dict] = {}
    for mirror in [False, True]:
        suffix = "mirror" if mirror else "direct"
        tables, global_prior, support = build_mpm_tables(dm, mirror)
        support["orientation"] = suffix
        mpm_reports.append(support)
        prior_oof = mpm_prior_for_rows(rows, tables, global_prior)
        prior_test = mpm_prior_for_rows(test_rows, tables, global_prior)
        for alpha in [0.001, 0.0025, 0.005, 0.01, 0.02]:
            prob_oof = blend_nonterminal_point(base_point_oof, prior_oof, alpha)
            prob_test = blend_nonterminal_point(base_point_test, prior_test, alpha)
            name = f"r157_mpm_{suffix}_a{clean_float(alpha)}"
            rec = eval_point(meta, prob_oof, base_point_oof, tuning, name, {"family": "r157_mpm", "alpha": alpha, "orientation": suffix})
            search_rows.append(rec)
            candidate_probs_test[name] = prob_test
            candidate_scores[name] = rec

    pd.concat(mpm_reports, ignore_index=True).to_csv(OUTDIR / "r157_mpm_conditional_support.csv", index=False)

    for mirror in [False, True]:
        suffix = "mirror" if mirror else "direct"
        prototypes = build_external_prototypes(dm, md, mirror)
        prototypes.to_csv(OUTDIR / f"r158_external_prototypes_{suffix}.csv", index=False)
        for tau in [0.35, 0.75, 1.25]:
            prior_oof = prototype_prior_for_rows(rows, prototypes, tau=tau, k=300)
            prior_test = prototype_prior_for_rows(test_rows, prototypes, tau=tau, k=300)
            for alpha in [0.001, 0.0025, 0.005, 0.01]:
                prob_oof = blend_nonterminal_point(base_point_oof, prior_oof, alpha)
                prob_test = blend_nonterminal_point(base_point_test, prior_test, alpha)
                name = f"r158_picp_{suffix}_tau{clean_float(tau)}_a{clean_float(alpha)}"
                rec = eval_point(
                    meta,
                    prob_oof,
                    base_point_oof,
                    tuning,
                    name,
                    {"family": "r158_prototype", "alpha": alpha, "orientation": suffix, "tau": tau},
                )
                search_rows.append(rec)
                candidate_probs_test[name] = prob_test
                candidate_scores[name] = rec

    search = pd.DataFrame(search_rows).sort_values(["point_macro_f1", "point_churn_vs_base"], ascending=[False, True])
    search.to_csv(OUTDIR / "r157_r158_point_search.csv", index=False)

    template = pd.read_csv(R67_ANCHOR)
    rally_uids = template["rally_uid"].astype(int).to_numpy()
    server_template = None
    for path in [R143_SERVER, R142_SERVER, R67_ANCHOR]:
        if path.exists():
            server_template = load_submission(path, rally_uids)["serverGetPoint"].to_numpy(dtype=float)
            break
    if server_template is None:
        server_template = template["serverGetPoint"].to_numpy(dtype=float)

    generated: list[dict] = []
    for name in [n for n in search["candidate"].tolist() if n != "r101_r111_point_base"][:5]:
        prob_test = candidate_probs_test[name]
        pred_point = point_pred(test_meta, prob_test, tuning)
        sub_name = f"submission_r67action_{name}_r142server.csv"
        info = write_submission(sub_name, template, pred_point, server_template)
        generated.append({**candidate_scores[name], **info, "submission": sub_name})
    pd.DataFrame(generated).to_csv(OUTDIR / "r157_r158_generated_candidates.csv", index=False)

    report = {
        "safety": {
            "uses_ttmatch": False,
            "uses_test_labels": False,
            "uses_rally_uid_lookup": False,
            "external_rows_appended_to_train": False,
            "point0_preserved": True,
        },
        "external_rows": {
            "deepmind": int(len(dm)),
            "tt_matchdynamics": int(len(md)),
        },
        "base_point_macro_f1": float(search[search["candidate"].eq("r101_r111_point_base")]["point_macro_f1"].iloc[0]),
        "best": search.head(10).to_dict(orient="records"),
        "generated_count": len(generated),
        "outputs": {
            "search": str(OUTDIR / "r157_r158_point_search.csv"),
            "generated": str(OUTDIR / "r157_r158_generated_candidates.csv"),
            "mpm_support": str(OUTDIR / "r157_mpm_conditional_support.csv"),
        },
    }
    (OUTDIR / "r157_r158_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTDIR / "r157_r158_report.md").write_text(
        "# R157/R158 Physics Pretraining Smoke\n\n"
        "R157 is a masked-physics conditional prior proxy. R158 is a physics-informed contrastive/prototype prior proxy.\n\n"
        "## Safety\n\n"
        "- No TTMATCH is used.\n"
        "- No AI CUP test labels are used.\n"
        "- No rally_uid lookup or external row append is used.\n"
        "- pointId=0 is preserved; only non-terminal point mass is redistributed.\n\n"
        f"Base point Macro-F1: `{report['base_point_macro_f1']:.6f}`\n\n"
        "## Top Rows\n\n"
        + search.head(12).to_csv(index=False)
        + "\n## Generated Candidates\n\n"
        + pd.DataFrame(generated).to_csv(index=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

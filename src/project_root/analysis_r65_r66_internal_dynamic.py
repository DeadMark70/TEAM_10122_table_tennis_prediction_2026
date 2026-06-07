"""R65/R66 action experiments.

R65:
  Test/internal-transition supervised action expert. In CV, validation rallies
  contribute only their observed prefix-internal transitions 1->2 ... L-1->L,
  never the held-out target L->L+1.

R66:
  Very-low-DoF dynamic rare-action multiplier on action probabilities. Only
  acts on rare/control classes and simple phase/pressure masks.

Point/server are fixed to the current R34 branch for generated submissions.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

from analysis_r7_phase_features import add_phase_features
from analysis_r48_action_meta_stacker import build_current_oof_action
from analysis_r57_player_style_clustering import add_player_id_features
from baseline_lgbm import (
    ACTION_CLASSES,
    add_role_and_score_features,
    build_test_prefix_table,
    build_train_prefix_table,
    class_weight_sample,
    feature_columns,
    sample_validation_prefixes,
    validate_raw_data,
)
from baseline_v3 import add_remaining_bucket, apply_segmented_multipliers
from generate_r42_golden_soft_blends import CURRENT_SUB_PATH, UPLOAD_DIR, normalize_rows


OUTDIR = Path("r65_r66_internal_dynamic")
ARTIFACT_PATH = Path("v47_v50_action_experts/v47_v50_action_experts.pkl")


@dataclass
class V3Tuning:
    action_ngram_weight: float
    point_ngram_weight: float
    server_weights: dict
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class GrUTuning:
    action_gru_weight: float
    point_gru_weight: float
    server_gru_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


@dataclass
class TransformerTuning:
    action_weight: float
    point_weight: float
    server_weight: float
    action_multipliers: dict
    point_multipliers: dict
    metrics: dict
    bins_mode: str


def load_artifact() -> dict:
    with open(ARTIFACT_PATH, "rb") as f:
        return pickle.load(f)


def apply_action(prob: np.ndarray, meta: pd.DataFrame, mult: dict) -> np.ndarray:
    return apply_segmented_multipliers(meta, prob, mult, ACTION_CLASSES, "two")


def make_action_model(seed: int, n_estimators: int = 180) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="multiclass",
        num_class=len(ACTION_CLASSES),
        n_estimators=n_estimators,
        learning_rate=0.04,
        num_leaves=39,
        min_child_samples=24,
        subsample=0.88,
        subsample_freq=1,
        colsample_bytree=0.88,
        reg_alpha=0.15,
        reg_lambda=2.0,
        random_state=seed,
        n_jobs=-1,
        verbosity=-1,
    )


def aligned_action_proba(model: lgb.LGBMClassifier, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    out = np.zeros((len(x), len(ACTION_CLASSES)), dtype=float)
    for i, cls in enumerate([int(c) for c in model.classes_]):
        out[:, ACTION_CLASSES.index(cls)] = proba[:, i]
    return normalize_rows(out)


def prepare_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train0 = pd.read_csv("train.csv")
    test0 = pd.read_csv("test_new.csv")
    validate_raw_data(train0, test0)
    train = add_role_and_score_features(train0)
    test = add_role_and_score_features(test0)
    prefix = add_remaining_bucket(build_train_prefix_table(train, 6))
    test_prefix = build_test_prefix_table(test, 6)
    prefix = add_phase_features(prefix, train)
    test_prefix = add_phase_features(test_prefix, test)
    prefix = add_player_id_features(prefix, train)
    test_prefix = add_player_id_features(test_prefix, test)
    player_cols = {"server_id", "receiver_id", "next_hitter_id", "next_receiver_id"}
    features = [c for c in feature_columns(prefix) if c != "remaining_len_bucket" and c not in player_cols]
    return train, test, prefix, test_prefix, features


def internal_transition_rows_from_prefixes(raw: pd.DataFrame, prefixes: pd.DataFrame, max_lag: int = 6) -> pd.DataFrame:
    """Build supervised rows from observed prefix-internal transitions only."""
    if prefixes.empty:
        return pd.DataFrame()
    keep = prefixes[["rally_uid", "prefix_len"]].drop_duplicates()
    part = raw.merge(keep, on="rally_uid", how="inner")
    part = part[part["strikeNumber"].le(part["prefix_len"])].drop(columns=["prefix_len"]).copy()
    if part.empty:
        return pd.DataFrame()
    if "serverGetPoint" not in part.columns:
        part["serverGetPoint"] = 0
    # build_train_prefix_table on a truncated observed prefix yields exactly
    # 1->2 ... L-1->L rows. Terminal/remaining fields are not used as features.
    table = add_remaining_bucket(build_train_prefix_table(part, max_lag))
    if table.empty:
        return table
    table = add_phase_features(table, part)
    table = add_player_id_features(table, part)
    table["is_internal_transition_row"] = 1
    return table


def describe(name: str, prob: np.ndarray, meta: pd.DataFrame, y: np.ndarray, base_pred: np.ndarray, mult: dict, extra: dict) -> dict:
    pred = apply_action(prob, meta, mult)
    row = {
        "candidate": name,
        "action_macro_f1": float(f1_score(y, pred, average="macro", labels=ACTION_CLASSES, zero_division=0)),
        "churn_vs_r42": float(np.mean(pred != base_pred)),
        "pred8_count": int((pred == 8).sum()),
        "pred9_count": int((pred == 9).sum()),
        "pred11_count": int((pred == 11).sum()),
        "pred12_count": int((pred == 12).sum()),
        "pred14_count": int((pred == 14).sum()),
    }
    row.update(extra)
    return row


def dynamic_multiplier(prob: np.ndarray, meta: pd.DataFrame, classes: list[int], rule: str, factor: float) -> np.ndarray:
    out = prob.copy()
    prefix_len = meta["prefix_len"].to_numpy(dtype=int)
    phase = meta["phase_id"].to_numpy(dtype=int) if "phase_id" in meta.columns else np.select(
        [prefix_len == 1, prefix_len == 2, prefix_len == 3, prefix_len >= 4],
        [1, 2, 3, 4],
        default=0,
    )
    score_total = meta["scoreTotal"].to_numpy(dtype=int)
    score_diff_abs = np.abs(meta["serverScoreDiff"].to_numpy(dtype=int))
    pressure = (
        (meta.get("is_deuce_like", pd.Series(np.zeros(len(meta), dtype=int))).to_numpy(dtype=int).astype(bool))
        | (meta.get("server_at_game_point_like", pd.Series(np.zeros(len(meta), dtype=int))).to_numpy(dtype=int).astype(bool))
        | (meta.get("receiver_at_game_point_like", pd.Series(np.zeros(len(meta), dtype=int))).to_numpy(dtype=int).astype(bool))
        | ((score_total >= 16) & (score_diff_abs <= 2))
    )
    if rule == "len1":
        mask = prefix_len == 1
    elif rule == "len2":
        mask = prefix_len == 2
    elif rule == "short":
        mask = prefix_len <= 2
    elif rule == "rally":
        mask = prefix_len >= 3
    elif rule == "pressure":
        mask = pressure
    elif rule == "rally_pressure":
        mask = (prefix_len >= 3) & pressure
    elif rule == "phase3plus":
        mask = phase >= 3
    else:
        raise ValueError(rule)
    if mask.any():
        out[np.ix_(mask, classes)] *= factor
    return normalize_rows(out)


def write_submission(test_meta: pd.DataFrame, pred: np.ndarray, current_sub: pd.DataFrame, name: str) -> dict:
    sub = pd.DataFrame(
        {
            "rally_uid": test_meta["rally_uid"].astype(int),
            "actionId": pred.astype(int),
            "pointId": current_sub["pointId"].astype(int),
            "serverGetPoint": np.round(np.clip(current_sub["serverGetPoint"].to_numpy(dtype=float), 1e-6, 1 - 1e-6), 8),
        }
    )
    path = OUTDIR / name
    sub.to_csv(path, index=False, float_format="%.8f")
    (UPLOAD_DIR / name).write_bytes(path.read_bytes())
    return {
        "candidate": name,
        "path": str(path),
        "upload_path": str(UPLOAD_DIR / name),
        "action_diff_vs_current_r34": float(np.mean(pred != current_sub["actionId"].to_numpy(dtype=int))),
        "action8_count": int((pred == 8).sum()),
        "action9_count": int((pred == 9).sum()),
        "action12_count": int((pred == 12).sum()),
        "action14_count": int((pred == 14).sum()),
    }


def main() -> None:
    OUTDIR.mkdir(exist_ok=True)
    UPLOAD_DIR.mkdir(exist_ok=True)
    art = load_artifact()
    train, test, prefix, test_prefix, features = prepare_tables()

    rally_meta = prefix[["rally_uid", "match"]].drop_duplicates().reset_index(drop=True)
    test_lengths = test_prefix["prefix_len"].to_numpy(dtype=int)
    meta_parts = []
    r65_parts: dict[float, list[np.ndarray]] = {w: [] for w in [0.1, 0.25, 0.5, 1.0]}

    for fold, (tr_rally_idx, va_rally_idx) in enumerate(GroupKFold(n_splits=5).split(rally_meta, groups=rally_meta["match"]), start=1):
        train_rallies = set(rally_meta.iloc[tr_rally_idx]["rally_uid"])
        valid_rallies = set(rally_meta.iloc[va_rally_idx]["rally_uid"])
        tr = prefix[prefix["rally_uid"].isin(train_rallies)].copy().reset_index(drop=True)
        valid_pool = prefix[prefix["rally_uid"].isin(valid_rallies)].copy()
        sampled_idx = sample_validation_prefixes(valid_pool, test_lengths, 42 + fold)
        va = valid_pool.loc[sampled_idx].copy().reset_index(drop=True)
        va_internal = internal_transition_rows_from_prefixes(train[train["rally_uid"].isin(valid_rallies)], va)
        meta_parts.append(va[["rally_uid", "match", "prefix_len", "next_actionId", "phase_id", "scoreTotal", "serverScoreDiff", "is_deuce_like", "server_at_game_point_like", "receiver_at_game_point_like"]])

        for internal_w in r65_parts:
            if va_internal.empty:
                aug = tr.copy()
                sw = class_weight_sample(aug["next_actionId"])
            else:
                aug = pd.concat([tr, va_internal], ignore_index=True)
                sw = class_weight_sample(aug["next_actionId"])
                sw[-len(va_internal) :] *= internal_w
                sw = sw / np.mean(sw)
            model = make_action_model(seed=6500 + fold + int(internal_w * 100))
            model.fit(aug[features], aug["next_actionId"], sample_weight=sw)
            r65_parts[internal_w].append(aligned_action_proba(model, va[features]))
        print(f"fold {fold} done; internal rows={len(va_internal)}")

    meta = pd.concat(meta_parts, ignore_index=True).reset_index(drop=True)
    y = meta["next_actionId"].to_numpy(dtype=int)
    current_oof = build_current_oof_action()
    golden_oof = art["experts_oof"]["v47_v64_oof_soft"]
    r42_full = normalize_rows(0.80 * current_oof + 0.20 * golden_oof)
    src = art["valid_meta"].copy().reset_index(drop=True)
    src["_row"] = np.arange(len(src))
    align = meta[["rally_uid", "prefix_len", "next_actionId"]].merge(
        src[["rally_uid", "prefix_len", "next_actionId", "_row"]],
        on=["rally_uid", "prefix_len", "next_actionId"],
        how="left",
        validate="one_to_one",
    )
    if align["_row"].isna().any():
        raise ValueError("Could not align R42 OOF to R65/R66 meta.")
    idx = align["_row"].to_numpy(dtype=int)
    r42_oof = r42_full[idx]
    mult = art["selected"]["action_multipliers"]
    base_pred = apply_action(r42_oof, meta, mult)
    base_f1 = float(f1_score(y, base_pred, average="macro", labels=ACTION_CLASSES, zero_division=0))

    rows = [{"candidate": "r42_base", "experiment": "base", "action_macro_f1": base_f1, "churn_vs_r42": 0.0}]
    oof_by_name = {"r42_base": r42_oof}
    r65_oof = {f"r65_internal_w{w}": np.vstack(parts) for w, parts in r65_parts.items()}

    class_sets = {
        "rare_control": [8, 9, 11, 12],
        "rare_only": [8, 9, 12, 14],
        "low_action": [0, 3, 4, 7, 8, 9, 11, 12, 14],
    }

    for name, prob in r65_oof.items():
        rows.append(describe(name, prob, meta, y, base_pred, mult, {"experiment": "R65", "blend_type": "single"}))
        for w in [0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]:
            blend = normalize_rows((1 - w) * r42_oof + w * prob)
            label = f"{name}_row_w{w}"
            rows.append(describe(label, blend, meta, y, base_pred, mult, {"experiment": "R65", "blend_type": "row", "weight": w}))
            oof_by_name[label] = blend
            for set_name, classes in class_sets.items():
                cb = r42_oof.copy()
                for cls in classes:
                    cb[:, cls] = (1 - w) * r42_oof[:, cls] + w * prob[:, cls]
                cb = normalize_rows(cb)
                label = f"{name}_cls_{set_name}_w{w}"
                rows.append(describe(label, cb, meta, y, base_pred, mult, {"experiment": "R65", "blend_type": "class", "class_set": set_name, "weight": w}))
                oof_by_name[label] = cb

    # R66 dynamic multiplier on R42 and on the best R65 blends.
    r66_sources = {"r42": r42_oof}
    for key in list(oof_by_name):
        if key.startswith("r65_internal_w") and ("_cls_" in key or "_row_w" in key):
            r66_sources[key] = oof_by_name[key]
    r66_classes = {
        "rare_control": [8, 9, 11, 12],
        "rare_only": [8, 9, 12, 14],
    }
    for source_name, source_prob in r66_sources.items():
        for cls_name, classes in r66_classes.items():
            for rule in ["len1", "short", "rally", "pressure", "rally_pressure", "phase3plus"]:
                for factor in [0.50, 0.70, 0.85, 1.10, 1.20, 1.35]:
                    prob = dynamic_multiplier(source_prob, meta, classes, rule, factor)
                    label = f"r66_{source_name}_{cls_name}_{rule}_f{factor}"
                    rows.append(describe(label, prob, meta, y, base_pred, mult, {"experiment": "R66", "source": source_name, "class_set": cls_name, "rule": rule, "factor": factor}))
                    oof_by_name[label] = prob

    search = pd.DataFrame(rows).sort_values(["action_macro_f1", "churn_vs_r42"], ascending=[False, True]).reset_index(drop=True)
    search.to_csv(OUTDIR / "r65_r66_search.csv", index=False)
    for name, prob in r65_oof.items():
        np.save(OUTDIR / f"{name}_oof_action.npy", prob)

    # Full train/test predictions for R65.
    test_internal = internal_transition_rows_from_prefixes(test, test_prefix)
    r65_test: dict[str, np.ndarray] = {}
    for internal_w in r65_parts:
        if test_internal.empty:
            aug = prefix.copy()
            sw = class_weight_sample(aug["next_actionId"])
        else:
            aug = pd.concat([prefix, test_internal], ignore_index=True)
            sw = class_weight_sample(aug["next_actionId"])
            sw[-len(test_internal) :] *= internal_w
            sw = sw / np.mean(sw)
        model = make_action_model(seed=7500 + int(internal_w * 100))
        model.fit(aug[features], aug["next_actionId"], sample_weight=sw)
        r65_test[f"r65_internal_w{internal_w}"] = aligned_action_proba(model, test_prefix[features])

    r42_test = normalize_rows(0.80 * art["current_test_action"] + 0.20 * art["experts_test"]["v47_golden_test_soft"])
    current_sub = test_prefix[["rally_uid", "prefix_len"]].merge(pd.read_csv(CURRENT_SUB_PATH), on="rally_uid", how="left")
    if current_sub[["actionId", "pointId", "serverGetPoint"]].isna().any().any():
        raise ValueError("Current R34 submission did not align.")

    def build_test_prob(row: pd.Series) -> np.ndarray | None:
        label = str(row["candidate"])
        if label.startswith("r65_internal_w") and "_row_w" in label:
            source = label.split("_row_w")[0]
            return normalize_rows((1 - float(row["weight"])) * r42_test + float(row["weight"]) * r65_test[source])
        if label.startswith("r65_internal_w") and "_cls_" in label:
            source = label.split("_cls_")[0]
            set_name = str(row["class_set"])
            w = float(row["weight"])
            prob = r42_test.copy()
            for cls in class_sets[set_name]:
                prob[:, cls] = (1 - w) * r42_test[:, cls] + w * r65_test[source][:, cls]
            return normalize_rows(prob)
        if label.startswith("r66_"):
            source = str(row["source"])
            if source == "r42":
                source_prob = r42_test
            elif source in r65_test:
                source_prob = r65_test[source]
            elif "_row_w" in source:
                s = source.split("_row_w")[0]
                # parse source weight from name.
                w = float(source.split("_row_w")[1])
                source_prob = normalize_rows((1 - w) * r42_test + w * r65_test[s])
            elif "_cls_" in source:
                s = source.split("_cls_")[0]
                w = float(source.split("_w")[-1])
                set_name = source.split("_cls_")[1].rsplit("_w", 1)[0]
                source_prob = r42_test.copy()
                for cls in class_sets[set_name]:
                    source_prob[:, cls] = (1 - w) * r42_test[:, cls] + w * r65_test[s][:, cls]
                source_prob = normalize_rows(source_prob)
            else:
                return None
            return dynamic_multiplier(source_prob, test_prefix, r66_classes[str(row["class_set"])], str(row["rule"]), float(row["factor"]))
        return None

    generated = []
    for _, row in search[(search["candidate"] != "r42_base") & (search["churn_vs_r42"] <= 0.10)].head(10).iterrows():
        prob = build_test_prob(row)
        if prob is None:
            continue
        pred = apply_action(prob, test_prefix, mult)
        safe = str(row["candidate"]).replace(".", "p").replace(" ", "_")
        name = f"submission_{safe}_current_point_server.csv"
        info = write_submission(test_prefix, pred, current_sub, name)
        info.update({"source_oof_action_f1": row["action_macro_f1"], "source_oof_churn": row["churn_vs_r42"], "experiment": row["experiment"]})
        generated.append(info)
        if len(generated) >= 8:
            break
    pd.DataFrame(generated).to_csv(OUTDIR / "r65_r66_generated_candidates.csv", index=False)
    report = {
        "base_action_f1": base_f1,
        "test_internal_rows": int(len(test_internal)),
        "best": search.head(30).to_dict(orient="records"),
        "generated": generated,
        "note": "R65/R66 generated candidates change action only; point/server fixed to current R34.",
    }
    (OUTDIR / "r65_r66_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(search.head(30).to_string(index=False))
    print(pd.DataFrame(generated).to_string(index=False))


if __name__ == "__main__":
    main()

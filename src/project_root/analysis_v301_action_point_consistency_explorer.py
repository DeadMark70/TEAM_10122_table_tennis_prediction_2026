"""V301 ultra-low-churn action/point consistency explorer.

Diagnostic-only variants over the clean V261 anchor.  The explorer uses only
empirical support from train.csv: P(point|action), P(action|point), joint pair
support, terminal action0/point0 consistency, and a next-action prior that keeps
serve labels 15-18 from being introduced as test next-label predictions.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "train.csv"
ANCHOR_PATH = ROOT / "upload_candidates_20260519" / "submission_v261_cap0p01__v173action_r121server.csv"
OUT_DIR = ROOT / "v301_action_point_consistency_explorer"
SRC_DEST = ROOT / "src" / "analysis" / "analysis_v301_action_point_consistency_explorer.py"
REQUIRED_SUBMISSION_COLUMNS = ["rally_uid", "actionId", "pointId", "serverGetPoint"]
ACTION_CLASSES = list(range(19))
POINT_CLASSES = list(range(10))
SERVE_ACTIONS = set(range(15, 19))
TERMINAL_PAIR = (0, 0)


@dataclass(frozen=True)
class Variant:
    name: str
    candidate_type: str
    cap: float
    require_nonterminal: bool = False
    require_action0_anchor: bool = False


VARIANTS = [
    Variant("terminal_consistency_cap0p0025", "terminal_consistency", 0.0025),
    Variant(
        "nonterminal_no_action0_cap0p0025",
        "nonterminal_no_action0",
        0.0025,
        require_nonterminal=True,
        require_action0_anchor=True,
    ),
    Variant("point_action_support_cap0p0025", "point_action_support", 0.0025),
    Variant("support_pair_cap0p005", "support_pair", 0.005),
]


@dataclass
class EmpiricalSupport:
    action_point_counts: pd.DataFrame
    action_counts: dict[int, int]
    point_counts: dict[int, int]
    pair_counts: dict[tuple[int, int], int]
    next_action_counts: dict[int, int]
    total_next_actions: int
    smoothing: float = 1.0

    @property
    def total_pairs(self) -> int:
        return int(sum(self.pair_counts.values()))

    def point_given_action(self, action_id: int, point_id: int) -> float:
        count = self.pair_counts.get((int(action_id), int(point_id)), 0)
        denom = self.action_counts.get(int(action_id), 0) + self.smoothing * len(POINT_CLASSES)
        return float((count + self.smoothing) / denom)

    def action_given_point(self, action_id: int, point_id: int) -> float:
        count = self.pair_counts.get((int(action_id), int(point_id)), 0)
        denom = self.point_counts.get(int(point_id), 0) + self.smoothing * len(ACTION_CLASSES)
        return float((count + self.smoothing) / denom)

    def pair_prob(self, action_id: int, point_id: int) -> float:
        count = self.pair_counts.get((int(action_id), int(point_id)), 0)
        denom = self.total_pairs + self.smoothing * len(ACTION_CLASSES) * len(POINT_CLASSES)
        return float((count + self.smoothing) / denom)

    def pair_count(self, action_id: int, point_id: int) -> int:
        return int(self.pair_counts.get((int(action_id), int(point_id)), 0))

    def serve_next_prior(self, action_id: int) -> float:
        if self.total_next_actions <= 0:
            return 0.0
        return float(self.next_action_counts.get(int(action_id), 0) / self.total_next_actions)

    def support_score(self, action_id: int, point_id: int) -> float:
        action_id = int(action_id)
        point_id = int(point_id)
        return float(
            0.45 * self.point_given_action(action_id, point_id)
            + 0.45 * self.action_given_point(action_id, point_id)
            + 0.10 * self.pair_prob(action_id, point_id)
        )

    def best_point_for_action(self, action_id: int, exclude_current: int | None = None) -> tuple[int, float]:
        choices = []
        for point_id in POINT_CLASSES:
            if exclude_current is not None and int(point_id) == int(exclude_current):
                continue
            choices.append((point_id, self.support_score(int(action_id), point_id), self.pair_count(int(action_id), point_id)))
        choices.sort(key=lambda item: (item[1], item[2], -item[0]), reverse=True)
        return int(choices[0][0]), float(choices[0][1])

    def best_action_for_point(
        self,
        point_id: int,
        exclude_current: int | None = None,
        allow_terminal_action0: bool = False,
    ) -> tuple[int, float]:
        choices = []
        for action_id in ACTION_CLASSES:
            if exclude_current is not None and int(action_id) == int(exclude_current):
                continue
            if action_id in SERVE_ACTIONS and self.serve_next_prior(action_id) < 0.001:
                continue
            if action_id == 0 and not allow_terminal_action0:
                continue
            choices.append((action_id, self.support_score(action_id, int(point_id)), self.pair_count(action_id, int(point_id))))
        if not choices:
            return int(exclude_current or 0), 0.0
        choices.sort(key=lambda item: (item[1], item[2], -item[0]), reverse=True)
        return int(choices[0][0]), float(choices[0][1])

    def best_supported_pair(self, anchor_action: int, anchor_point: int) -> tuple[int, int, float]:
        choices = []
        for action_id in ACTION_CLASSES:
            if action_id in SERVE_ACTIONS and self.serve_next_prior(action_id) < 0.001:
                continue
            for point_id in POINT_CLASSES:
                if (action_id, point_id) == (int(anchor_action), int(anchor_point)):
                    continue
                if (action_id == 0) != (point_id == 0):
                    continue
                score = self.support_score(action_id, point_id)
                choices.append((action_id, point_id, score, self.pair_count(action_id, point_id)))
        choices.sort(key=lambda item: (item[2], item[3], -item[0], -item[1]), reverse=True)
        best = choices[0]
        return int(best[0]), int(best[1]), float(best[2])


def _require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def build_empirical_support(train: pd.DataFrame, smoothing: float = 1.0) -> EmpiricalSupport:
    _require_columns(train, ["rally_uid", "strikeNumber", "actionId", "pointId"], "train")
    clean = train[["rally_uid", "strikeNumber", "actionId", "pointId"]].dropna().copy()
    clean["actionId"] = clean["actionId"].astype(int)
    clean["pointId"] = clean["pointId"].astype(int)
    clean = clean[clean["actionId"].isin(ACTION_CLASSES) & clean["pointId"].isin(POINT_CLASSES)]

    pair_counts_series = clean.groupby(["actionId", "pointId"], dropna=False).size()
    pair_counts = {(int(a), int(p)): int(v) for (a, p), v in pair_counts_series.items()}
    action_counts = {int(k): int(v) for k, v in clean["actionId"].value_counts().items()}
    point_counts = {int(k): int(v) for k, v in clean["pointId"].value_counts().items()}

    next_actions = []
    for _, rally in clean.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False):
        actions = rally["actionId"].astype(int).tolist()
        next_actions.extend(actions[1:])
    next_counts = pd.Series(next_actions, dtype="int64").value_counts() if next_actions else pd.Series(dtype="int64")
    next_action_counts = {int(k): int(v) for k, v in next_counts.items()}

    table = (
        clean.groupby(["actionId", "pointId"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["actionId", "pointId"])
    )
    return EmpiricalSupport(
        action_point_counts=table,
        action_counts=action_counts,
        point_counts=point_counts,
        pair_counts=pair_counts,
        next_action_counts=next_action_counts,
        total_next_actions=len(next_actions),
        smoothing=float(smoothing),
    )


def max_changed_rows(row_count: int, cap: float) -> int:
    return min(int(math.floor(int(row_count) * float(cap))), 10)


def _candidate_record(
    row: Any,
    candidate_action: int,
    candidate_point: int,
    candidate_type: str,
    support: EmpiricalSupport,
    reason: str,
) -> dict[str, Any] | None:
    anchor_action = int(row.actionId)
    anchor_point = int(row.pointId)
    candidate_action = int(candidate_action)
    candidate_point = int(candidate_point)
    if (candidate_action, candidate_point) == (anchor_action, anchor_point):
        return None
    if candidate_action in SERVE_ACTIONS and candidate_action != anchor_action and support.serve_next_prior(candidate_action) < 0.001:
        return None
    if (candidate_action == 0) != (candidate_point == 0):
        return None
    anchor_score = support.support_score(anchor_action, anchor_point)
    candidate_score = support.support_score(candidate_action, candidate_point)
    support_delta = candidate_score - anchor_score
    if support_delta <= 0.0:
        return None
    return {
        "row_id": int(row.Index),
        "rally_uid": int(row.rally_uid),
        "candidate_type": candidate_type,
        "reason": reason,
        "anchor_action": anchor_action,
        "anchor_point": anchor_point,
        "candidate_action": candidate_action,
        "candidate_point": candidate_point,
        "action_changed": bool(candidate_action != anchor_action),
        "point_changed": bool(candidate_point != anchor_point),
        "pair_changed": True,
        "anchor_support_score": anchor_score,
        "candidate_support_score": candidate_score,
        "support_delta": support_delta,
        "pair_count": support.pair_count(candidate_action, candidate_point),
        "pair_prob": support.pair_prob(candidate_action, candidate_point),
        "point_given_action": support.point_given_action(candidate_action, candidate_point),
        "action_given_point": support.action_given_point(candidate_action, candidate_point),
        "serve_next_prior": support.serve_next_prior(candidate_action),
    }


def candidate_pool(anchor: pd.DataFrame, support: EmpiricalSupport) -> pd.DataFrame:
    _require_columns(anchor, REQUIRED_SUBMISSION_COLUMNS, "anchor")
    rows: list[dict[str, Any]] = []
    for row in anchor.reset_index(drop=True).itertuples():
        action_id = int(row.actionId)
        point_id = int(row.pointId)

        if (action_id == 0) != (point_id == 0):
            record = _candidate_record(row, 0, 0, "terminal_consistency", support, "make_action0_point0_pair")
            if record is not None:
                rows.append(record)

        if action_id == 0 and point_id != 0:
            cand_action, _ = support.best_action_for_point(point_id, exclude_current=0)
            record = _candidate_record(row, cand_action, point_id, "nonterminal_no_action0", support, "replace_action0_for_nonterminal_point")
            if record is not None:
                rows.append(record)

        if action_id not in SERVE_ACTIONS and action_id != 0:
            cand_point, _ = support.best_point_for_action(action_id, exclude_current=point_id)
            record = _candidate_record(row, action_id, cand_point, "point_action_support", support, "replace_point_with_best_empirical_point_given_action")
            if record is not None:
                rows.append(record)

        cand_action, cand_point, _ = support.best_supported_pair(action_id, point_id)
        record = _candidate_record(row, cand_action, cand_point, "support_pair", support, "replace_pair_with_best_empirical_support_pair")
        if record is not None:
            rows.append(record)

    columns = [
        "row_id",
        "rally_uid",
        "candidate_type",
        "reason",
        "anchor_action",
        "anchor_point",
        "candidate_action",
        "candidate_point",
        "action_changed",
        "point_changed",
        "pair_changed",
        "anchor_support_score",
        "candidate_support_score",
        "support_delta",
        "pair_count",
        "pair_prob",
        "point_given_action",
        "action_given_point",
        "serve_next_prior",
    ]
    return pd.DataFrame(rows, columns=columns)


def select_variant(candidates: pd.DataFrame, variant: Variant, row_count: int) -> pd.DataFrame:
    cap_rows = max_changed_rows(row_count, variant.cap)
    if cap_rows <= 0 or candidates.empty:
        return candidates.head(0).copy()
    selected = candidates[candidates["candidate_type"].eq(variant.candidate_type)].copy()
    if variant.require_nonterminal:
        selected = selected[(selected["anchor_point"] != 0) & (selected["candidate_point"] != 0)]
    if variant.require_action0_anchor:
        selected = selected[selected["anchor_action"].eq(0)]
    selected = selected[~selected["candidate_action"].between(15, 18)]
    selected = selected[selected["pair_count"] > 0]
    if selected.empty:
        return selected.head(0).copy()
    selected = selected.sort_values(
        ["support_delta", "candidate_support_score", "pair_count", "pair_prob", "row_id"],
        ascending=[False, False, False, False, True],
    )
    return selected.drop_duplicates("row_id").head(cap_rows).copy()


def apply_selected_changes(anchor: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    _require_columns(anchor, REQUIRED_SUBMISSION_COLUMNS, "anchor")
    out = anchor.loc[:, REQUIRED_SUBMISSION_COLUMNS].copy()
    if not selected.empty:
        repl = selected.set_index("row_id")[["candidate_action", "candidate_point"]].to_dict("index")
        for row_id, values in repl.items():
            out.loc[int(row_id), "actionId"] = int(values["candidate_action"])
            out.loc[int(row_id), "pointId"] = int(values["candidate_point"])
    return out.loc[:, REQUIRED_SUBMISSION_COLUMNS]


def _rate_delta(anchor: pd.DataFrame, submission: pd.DataFrame, column: str, value: int) -> float:
    return float(submission[column].astype(int).eq(value).mean() - anchor[column].astype(int).eq(value).mean())


def _variant_metrics(anchor: pd.DataFrame, submission: pd.DataFrame, selected: pd.DataFrame, variant: Variant) -> dict[str, Any]:
    action_changed = submission["actionId"].astype(int).ne(anchor["actionId"].astype(int))
    point_changed = submission["pointId"].astype(int).ne(anchor["pointId"].astype(int))
    pair_changed = action_changed | point_changed
    return {
        "candidate": variant.name,
        "cap": variant.cap,
        "max_changed_rows": max_changed_rows(len(anchor), variant.cap),
        "action_churn": float(action_changed.mean()),
        "point_churn": float(point_changed.mean()),
        "pair_changed_rows": int(pair_changed.sum()),
        "action_changed_rows": int(action_changed.sum()),
        "point_changed_rows": int(point_changed.sum()),
        "point0_rate_delta": _rate_delta(anchor, submission, "pointId", 0),
        "action0_rate_delta": _rate_delta(anchor, submission, "actionId", 0),
        "support_score": float(selected["candidate_support_score"].mean()) if not selected.empty else 0.0,
        "support_delta": float(selected["support_delta"].mean()) if not selected.empty else 0.0,
        "min_pair_count": int(selected["pair_count"].min()) if not selected.empty else 0,
        "serve_15_18_added_rows": int(((selected["candidate_action"].between(15, 18)) & (~selected["anchor_action"].between(15, 18))).sum()) if not selected.empty else 0,
        "recommendation": "DO_NOT_UPLOAD",
    }


def conservative_train_proxy(train: pd.DataFrame, support: EmpiricalSupport, max_rows: int = 2500) -> dict[str, float]:
    proxy_anchor = train[["rally_uid", "actionId", "pointId"]].dropna().copy().reset_index(drop=True)
    proxy_anchor = proxy_anchor.head(max_rows).copy()
    proxy_anchor["serverGetPoint"] = 0
    proxy_candidates = candidate_pool(proxy_anchor, support)
    changed = 0
    for variant in VARIANTS:
        changed += len(select_variant(proxy_candidates, variant, len(proxy_anchor)))
    return {
        "local_oof_proxy_delta": float(-changed / max(len(proxy_anchor), 1)),
        "local_oof_proxy_changed_rows": float(changed),
    }


def _write_report_md(path: Path, search: pd.DataFrame, report: dict[str, Any]) -> None:
    best = search.sort_values(["support_delta", "pair_changed_rows"], ascending=[False, False]).iloc[0].to_dict()
    lines = [
        "# V301 ultra-low-churn action-point consistency explorer",
        "",
        f"- Anchor: `{report['anchor_submission']}`",
        f"- Train source: `{report['train_source']}`",
        f"- Best diagnostic by support_delta: `{best['candidate']}`",
        f"- Best support_delta: `{float(best['support_delta']):.8f}`",
        f"- Best pair_changed_rows: `{int(best['pair_changed_rows'])}`",
        f"- Upload recommendation: `{report['upload_recommendation']}`",
        "",
        "Notes:",
        "- Diagnostic only; recommendation remains DO_NOT_UPLOAD without a positive local OOF proxy.",
        "- TTMATCH and old-server inputs are not read.",
        "- serverGetPoint is copied from the anchor without modification.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(
    train_path: Path = TRAIN_PATH,
    anchor_path: Path = ANCHOR_PATH,
    out_dir: Path = OUT_DIR,
    copy_to_src: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(train_path)
    anchor = pd.read_csv(anchor_path)
    _require_columns(anchor, REQUIRED_SUBMISSION_COLUMNS, "anchor")
    anchor = anchor.loc[:, REQUIRED_SUBMISSION_COLUMNS].copy()
    if anchor["rally_uid"].duplicated().any():
        raise ValueError("anchor rally_uid contains duplicates")

    support = build_empirical_support(train)
    candidates = candidate_pool(anchor, support)
    candidates.to_csv(out_dir / "v301_candidate_pool.csv", index=False)

    search_rows = []
    audit_rows = []
    variants: dict[str, dict[str, Any]] = {}
    generated_submissions = []
    for variant in VARIANTS:
        selected = select_variant(candidates, variant, len(anchor))
        submission = apply_selected_changes(anchor, selected)
        metrics = _variant_metrics(anchor, submission, selected, variant)
        variants[variant.name] = metrics
        search_rows.append(metrics)

        sub_path = out_dir / f"submission_v301_{variant.name}.csv"
        submission.to_csv(sub_path, index=False)
        generated_submissions.append(str(sub_path.relative_to(ROOT)) if sub_path.is_relative_to(ROOT) else str(sub_path))

        if not selected.empty:
            audit = selected.copy()
            audit["variant"] = variant.name
            audit_rows.append(audit)

    search = pd.DataFrame(search_rows)
    search.to_csv(out_dir / "v301_pair_search.csv", index=False)
    if audit_rows:
        audit_df = pd.concat(audit_rows, ignore_index=True)
    else:
        audit_df = pd.DataFrame(columns=["variant", *candidates.columns.tolist()])
    audit_df.to_csv(out_dir / "v301_changed_row_audit.csv", index=False)

    proxy = conservative_train_proxy(train, support)
    report = {
        "version": "V301",
        "anchor_submission": str(Path(anchor_path).relative_to(ROOT)) if Path(anchor_path).is_relative_to(ROOT) else str(anchor_path),
        "train_source": str(Path(train_path).relative_to(ROOT)) if Path(train_path).is_relative_to(ROOT) else str(train_path),
        "output_dir": str(out_dir.relative_to(ROOT)) if out_dir.is_relative_to(ROOT) else str(out_dir),
        "variants": variants,
        "generated_submissions": generated_submissions,
        "local_oof_proxy": proxy,
        "upload_recommendation": "DO_NOT_UPLOAD",
        "no_ttmatch": True,
        "no_old_server": True,
        "no_manual_row_fixes": True,
        "server_fixed": True,
    }
    (out_dir / "v301_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    _write_report_md(out_dir / "v301_report.md", search, report)

    if copy_to_src:
        SRC_DEST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(__file__).resolve(), SRC_DEST)
    return report


def main() -> None:
    report = run_pipeline()
    best = max(report["variants"].values(), key=lambda row: (row["support_delta"], row["pair_changed_rows"]))
    print(
        json.dumps(
            {
                "outdir": report["output_dir"],
                "best_candidate": best["candidate"],
                "best_pair_changed_rows": best["pair_changed_rows"],
                "best_support_delta": best["support_delta"],
                "upload_recommendation": report["upload_recommendation"],
                "generated_submissions": len(report["generated_submissions"]),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

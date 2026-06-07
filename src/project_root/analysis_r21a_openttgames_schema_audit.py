"""R21A Extended OpenTTGames parser / schema audit.

This script parses the downloaded Extended OpenTTGames repository into a
neutral event schema and count reports. It does not train any model and does
not map labels into direct AICUP targets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd


STROKE_TECHNIQUES = {"block", "chop", "flick", "lob", "loop", "push", "serve", "smash"}
LEAN_LABELS = {"back_heavy", "front_heavy", "right_leaning", "left_leaning", "neutral", "unknown"}
FEET_LABELS = {
    "both_feet_planted",
    "both_feet_lifted",
    "right_foot_lifted",
    "left_foot_lifted",
    "unknown",
}
SIMPLE_EVENTS = {"empty_event", "bounce", "net"}
ENDING_TYPES = {
    "net",
    "not_hitting_ball",
    "winner",
    "double_bounce",
    "out",
    "miss_on_own_side",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run R21A Extended OpenTTGames schema audit.")
    parser.add_argument("--root", default="external_data/openttgames")
    parser.add_argument("--out-dir", default="external_data/openttgames/processed")
    return parser.parse_args()


def safe_action_family(technique: str | None) -> str:
    if technique is None:
        return ""
    if technique == "serve":
        return "serve_family"
    if technique in {"loop", "smash", "flick"}:
        return "attack_family"
    if technique == "push":
        return "control_family"
    if technique in {"block", "chop", "lob"}:
        return "defensive_or_control_family"
    return "unknown_family"


def parse_stroke_token(token: str) -> dict:
    parts = token.split("_")
    if len(parts) < 3:
        return {"ok": False, "reason": "stroke_token_too_short"}
    side, hand = parts[0], parts[1]
    technique = "_".join(parts[2:])
    if side not in {"left", "right"}:
        return {"ok": False, "reason": "invalid_player_side"}
    if hand not in {"forehand", "backhand"}:
        return {"ok": False, "reason": "invalid_stroke_hand"}
    if technique not in STROKE_TECHNIQUES:
        return {"ok": False, "reason": "invalid_technique"}
    return {"ok": True, "player_side": side, "stroke_hand": hand, "technique": technique}


def parse_label(raw: str) -> dict:
    raw = str(raw).strip()
    base = {
        "event_raw_label": raw,
        "event_type": "unknown",
        "player_side": "",
        "stroke_hand": "",
        "technique": "",
        "lean": "",
        "feet": "",
        "rally_ending_type": "",
        "is_stroke": 0,
        "is_bounce": 0,
        "is_net": 0,
        "is_rally_ending": 0,
        "safe_action_family": "",
        "safe_terminal_label": "",
        "unsafe_notes": "",
        "parse_status": "ok",
    }
    if raw in SIMPLE_EVENTS:
        base["event_type"] = raw
        base["is_bounce"] = int(raw == "bounce")
        base["is_net"] = int(raw == "net")
        return base

    tokens = raw.split()
    first = tokens[0] if tokens else ""
    ending = parse_ending_token(first)
    if ending["ok"] and len(tokens) == 1:
        base.update(
            {
                "event_type": "rally_ending",
                "player_side": ending["player_side"],
                "rally_ending_type": ending["rally_ending_type"],
                "is_net": int(ending["rally_ending_type"] == "net"),
                "is_rally_ending": 1,
                "safe_terminal_label": "1",
            }
        )
        return base

    stroke = parse_stroke_token(first)
    if stroke["ok"]:
        lean = tokens[1] if len(tokens) >= 2 else ""
        feet = "_".join(tokens[2:]) if len(tokens) >= 3 else ""
        base.update(
            {
                "event_type": "stroke",
                "player_side": stroke["player_side"],
                "stroke_hand": stroke["stroke_hand"],
                "technique": stroke["technique"],
                "lean": lean,
                "feet": feet,
                "is_stroke": 1,
                "safe_action_family": safe_action_family(stroke["technique"]),
            }
        )
        notes = []
        if lean not in LEAN_LABELS:
            notes.append("unknown_or_invalid_lean")
        if feet not in FEET_LABELS:
            notes.append("unknown_or_invalid_feet")
        base["parse_status"] = "ok" if not notes else "partial"
        base["unsafe_notes"] = ";".join(notes)
        return base

    base["parse_status"] = "unknown"
    base["unsafe_notes"] = stroke.get("reason", "unrecognized_label")
    return base


def parse_ending_token(token: str) -> dict:
    for side in ("left", "right"):
        prefix = side + "_"
        if token.startswith(prefix):
            rest = token[len(prefix) :]
            if rest in ENDING_TYPES:
                return {"ok": True, "player_side": side, "rally_ending_type": rest}
    return {"ok": False}


def iter_game_files(root: Path):
    game_root = root / "data" / "raw" / "game_data"
    for split in ["train", "test"]:
        for path in sorted((game_root / split).glob("*.json")):
            yield split, path


def iter_ball_files(root: Path):
    ball_root = root / "data" / "raw" / "ball_data"
    for split in ["train", "test"]:
        for path in sorted((ball_root / split).glob("*.json")):
            yield split, path


def parse_game_events(root: Path) -> pd.DataFrame:
    rows = []
    for split, path in iter_game_files(root):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for frame, raw_label in sorted(data.items(), key=lambda kv: int(kv[0])):
            parsed = parse_label(raw_label)
            parsed.update(
                {
                    "source": "Extended OpenTTGames",
                    "video_id": path.stem,
                    "split": split,
                    "frame": int(frame),
                }
            )
            rows.append(parsed)
    cols = [
        "source",
        "video_id",
        "split",
        "frame",
        "event_raw_label",
        "event_type",
        "player_side",
        "stroke_hand",
        "technique",
        "lean",
        "feet",
        "rally_ending_type",
        "is_stroke",
        "is_bounce",
        "is_net",
        "is_rally_ending",
        "safe_action_family",
        "safe_terminal_label",
        "parse_status",
        "unsafe_notes",
    ]
    return pd.DataFrame(rows)[cols]


def parse_ball_summary(root: Path) -> pd.DataFrame:
    rows = []
    for split, path in iter_ball_files(root):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        valid = 0
        invalid = 0
        for xy in data.values():
            x = xy.get("x")
            y = xy.get("y")
            if x is None or y is None or int(x) < 0 or int(y) < 0:
                invalid += 1
            else:
                valid += 1
        rows.append(
            {
                "split": split,
                "video_id": path.stem,
                "frames_with_ball_rows": len(data),
                "valid_xy_rows": valid,
                "invalid_xy_rows": invalid,
                "valid_rate": valid / len(data) if data else 0.0,
            }
        )
    return pd.DataFrame(rows)


def mapping_audit(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in events.groupby("event_raw_label", sort=True):
        first = sub.iloc[0]
        if first["event_type"] == "stroke":
            if first["technique"] == "serve":
                unsafe = "AICUP actionId 15/16/17/18 exact"
                reason = "external serve labels do not encode AICUP serve subtype"
            elif first["technique"] == "push":
                unsafe = "AICUP actionId 6/8/9/10/11 exact"
                reason = "push/control labels are coarser than AICUP control subtypes"
            elif first["technique"] == "loop":
                unsafe = "AICUP actionId 1/2 exact"
                reason = "loop label does not separate drive/counter-drive"
            elif first["technique"] == "flick":
                unsafe = "AICUP actionId 4/7 exact"
                reason = "flick label does not separate twist/flip reliably"
            else:
                unsafe = "direct exact actionId without validation"
                reason = "external taxonomy differs from AICUP taxonomy"
            safe = first["safe_action_family"]
        elif first["event_type"] == "rally_ending":
            safe = "terminal_auxiliary"
            unsafe = "direct serverGetPoint"
            reason = "server side and scoring semantics are not audited"
        elif first["event_type"] in {"bounce", "net", "empty_event"}:
            safe = first["event_type"] + "_event_auxiliary"
            unsafe = "direct AICUP pointId"
            reason = "AICUP pointId is receiver-relative and not available from event name"
        else:
            safe = "none"
            unsafe = "all direct targets"
            reason = str(first["unsafe_notes"])
        rows.append(
            {
                "external_label": label,
                "count": int(len(sub)),
                "parsed_type": first["event_type"],
                "parse_status": first["parse_status"],
                "safe_auxiliary_target": safe,
                "unsafe_direct_target": unsafe,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows).sort_values(["parse_status", "parsed_type", "external_label"])


def write_counts(events: pd.DataFrame, out_dir: Path) -> None:
    event_counts = (
        events.groupby(["split", "video_id", "event_type", "parse_status"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "video_id", "event_type", "parse_status"])
    )
    event_counts.to_csv(out_dir / "openttgames_event_counts.csv", index=False)

    strokes = events[events["event_type"].eq("stroke")]
    stroke_counts = (
        strokes.groupby(["split", "video_id", "technique", "stroke_hand", "lean", "feet", "safe_action_family"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "video_id", "technique", "stroke_hand", "count"], ascending=[True, True, True, True, False])
    )
    stroke_counts.to_csv(out_dir / "openttgames_stroke_counts.csv", index=False)

    endings = events[events["event_type"].eq("rally_ending")]
    ending_counts = (
        endings.groupby(["split", "video_id", "player_side", "rally_ending_type"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["split", "video_id", "rally_ending_type", "player_side"])
    )
    ending_counts.to_csv(out_dir / "openttgames_ending_counts.csv", index=False)


def schema_report(root: Path, events: pd.DataFrame, ball_summary: pd.DataFrame) -> dict:
    license_text = ""
    license_path = root / "LICENSE"
    if license_path.exists():
        license_text = license_path.read_text(encoding="utf-8", errors="replace").strip()
    unknown = events[events["parse_status"].eq("unknown")]
    partial = events[events["parse_status"].eq("partial")]
    return {
        "source": "Extended OpenTTGames",
        "root": str(root),
        "license_file_text": license_text,
        "game_files": int(events["video_id"].nunique()),
        "event_rows": int(len(events)),
        "stroke_rows": int(events["is_stroke"].sum()),
        "rally_ending_rows": int(events["is_rally_ending"].sum()),
        "bounce_rows": int(events["is_bounce"].sum()),
        "net_rows": int(events["is_net"].sum()),
        "parse_status_counts": {str(k): int(v) for k, v in Counter(events["parse_status"]).items()},
        "unknown_labels": unknown["event_raw_label"].value_counts().to_dict(),
        "partial_label_issue_counts": partial["unsafe_notes"].value_counts().to_dict(),
        "technique_counts": events[events["event_type"].eq("stroke")]["technique"].value_counts().to_dict(),
        "safe_action_family_counts": events[events["event_type"].eq("stroke")]["safe_action_family"].value_counts().to_dict(),
        "rally_ending_type_counts": events[events["event_type"].eq("rally_ending")]["rally_ending_type"].value_counts().to_dict(),
        "ball_summary_rows": int(len(ball_summary)),
        "ball_valid_rate_mean": float(ball_summary["valid_rate"].mean()) if len(ball_summary) else None,
        "decision": "R21A parser audit only. No training or direct AICUP target mapping.",
    }


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    events = parse_game_events(root)
    ball_summary = parse_ball_summary(root)
    events.to_csv(out_dir / "openttgames_events.csv", index=False)
    ball_summary.to_csv(out_dir / "openttgames_ball_summary.csv", index=False)
    write_counts(events, out_dir)
    audit = mapping_audit(events)
    audit.to_csv(out_dir / "openttgames_mapping_audit.csv", index=False)
    report = schema_report(root, events, ball_summary)
    (out_dir / "openttgames_schema_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

"""Describe own actions associated with losses in negative OOF sequences.

This is a post-discovery descriptive audit. It does not add terminal-action
features to Popper or alter the learned rules.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactical_ilp.config import PROCESSED_DIR, PROLOG_DIR, REPORTS_DIR  # noqa: E402
from tactical_ilp.data.freeze_frames import build_all as build_merged_all  # noqa: E402
from tactical_ilp.features.geometry import is_diagonal_pass, pass_direction, pass_length_class  # noqa: E402
from tactical_ilp.features.frames import frame_points  # noqa: E402
from tactical_ilp.features.tactical import line_summary  # noqa: E402
from phase5_cv_common import COMPARISON_PROTOCOL_REVISION  # noqa: E402

OUT_JSON = REPORTS_DIR / "phase5_terminal_action_analysis.json"
OUT_MD = REPORTS_DIR / "phase5_terminal_action_analysis.md"
PASS_TYPES = {"pass", "cross", "freekick_short", "goalkick", "throw_in", "corner_short"}
CV_METHODS = [
    ("full_tactical", "phase5_cross_validation.json", "popper_cv_fold_", "dict"),
    ("hybrid_sequential", "phase5_sequential_cv.json", "popper_sequential_cv_fold_", "list"),
]


def _atom(text: str) -> str:
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _covered_negative_ids(task_dir: Path, body: list[str], swipl: str) -> list[str]:
    clean_body = ", ".join(f"{name}(S)" for name in body).strip().rstrip(".")
    query = (
        f"consult({_atom(str(task_dir / 'bk.pl'))}), "
        f"consult({_atom(str(task_dir / 'exs_holdout.pl'))}), "
        f"assertz((cv_terminal_rule(S) :- {clean_body})), "
        "findall(S,(pos(negative_outcome(S)),cv_terminal_rule(S)),RawIds), sort(RawIds,Ids), "
        "forall(member(Id,Ids),(write(Id),nl)), halt"
    )
    proc = subprocess.run(
        [swipl, "-D", "encoding=utf8", "-q", "-g", query],
        check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if proc.returncode:
        raise RuntimeError(f"SWI-Prolog failed for {task_dir.name}: {proc.stderr[-3000:]}")
    return [line.strip() for line in proc.stdout.splitlines() if re.fullmatch(r"s_\d+_\d+", line.strip())]


def _action_description(action: pd.Series, has_frame: bool) -> tuple[str, list[str]]:
    kind = str(action.get("type_name", "unknown"))
    result = str(action.get("result_name", "unknown"))
    tags = [f"result:{result}"]
    if kind not in PASS_TYPES:
        return kind, tags
    sx, sy, ex, ey = (action.get(k) for k in ("start_x", "start_y", "end_x", "end_y"))
    if not all(pd.notna(v) for v in (sx, sy, ex, ey)):
        tags.append("pass_geometry:unknown")
        return "pass", tags
    tags.append(f"direction:{pass_direction(float(sx), float(sy), float(ex), float(ey))}")
    tags.append(f"length:{pass_length_class(float(sx), float(sy), float(ex), float(ey))}")
    if is_diagonal_pass(float(sx), float(sy), float(ex), float(ey)):
        tags.append("diagonal")
    if result == "success" and float(ex) - float(sx) >= 10.0:
        tags.append("successful_forward_progress_10m")
    if has_frame and isinstance(action.get("original_event_id"), str):
        pts = frame_points(int(action.match_id), action.original_event_id, bool(action.get("is_barca", True)))
        if pts:
            summary = line_summary(pts, float(sx), float(ex))
            if summary.get("breaks_defensive_line"):
                tags.append("line_break_attempt_proxy")
                if result == "success":
                    tags.append("successful_line_break_proxy")
    return "pass", tags


def _fmt_counter(counter: Counter) -> str:
    if not counter:
        return "—"
    return "; ".join(f"{k}: {v}" for k, v in sorted(counter.items()))


def _loss_actions(actions: pd.DataFrame, label_row: pd.Series) -> tuple[pd.Series | None, str]:
    """Return the own action associated with the loss and the acquisition event."""
    match_rows = actions[actions.match_id.astype(str) == str(label_row.match_id)]
    turnover_row = int(label_row.turnover_row)
    turnover = match_rows[match_rows.row == turnover_row]
    if turnover.empty:
        return None, "unknown_turnover_event"
    event = turnover.iloc[0]
    if bool(label_row.final_action_failed):
        return event, f"failed_own_action:{event.type_name}"

    period = int(label_row.period_id)
    own_before = match_rows[
        (match_rows.period_id == period)
        & (match_rows.row < turnover_row)
        & (match_rows.row >= int(label_row.last_action_row))
        & match_rows.is_barca.astype(bool)
    ]
    if own_before.empty:
        own_before = match_rows[
            (match_rows.period_id == period)
            & (match_rows.row < turnover_row)
            & match_rows.is_barca.astype(bool)
        ]
    own_action = own_before.sort_values("row").iloc[-1] if not own_before.empty else None
    return own_action, f"opponent_acquisition:{event.type_name}"


def main() -> int:
    swipl = shutil.which("swipl")
    if not swipl:
        raise RuntimeError("SWI-Prolog (swipl) is required to identify OOF rule matches")
    reports = []
    stale_reports = []
    for method, filename, task_prefix, clause_format in CV_METHODS:
        path = REPORTS_DIR / filename
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            if report.get("comparison_protocol_revision") != COMPARISON_PROTOCOL_REVISION:
                stale_reports.append(str(path))
                continue
            reports.append((method, path, task_prefix, clause_format, report))
    if not reports:
        expected = ", ".join(name for _, name, _, _ in CV_METHODS)
        raise FileNotFoundError(f"Run at least one grouped CV first; expected one of: {expected}")
    labelled = pd.read_parquet(PROCESSED_DIR / "labeled_sequences.parquet")
    actions = build_merged_all()
    actions = actions.sort_values(["match_id", "period_id", "time_seconds", "action_id"]).copy()
    actions["row"] = actions.groupby("match_id").cumcount()
    neg_ids = set(labelled.loc[labelled.label == "negative", "sequence_id"].astype(str))
    labelled = labelled.set_index("sequence_id")
    fold_sections = []
    all_rows = []
    for method, report_path, task_prefix, clause_format, report in reports:
        for fold in report.get("fold_results", []):
            fold_no = int(fold["fold"])
            task_dir = PROLOG_DIR / f"{task_prefix}{fold_no:02d}"
            raw_clauses = fold.get("clauses", [])
            clauses = [c["body"] for c in raw_clauses] if clause_format == "dict" else raw_clauses
            for rule_no, body in enumerate(clauses, 1):
                ids = [sid for sid in _covered_negative_ids(task_dir, body, swipl) if sid in neg_ids]
                types, tags, turnover_events = Counter(), defaultdict(Counter), Counter()
                unknown_terminal = 0
                for sid in ids:
                    if sid not in labelled.index:
                        unknown_terminal += 1
                        continue
                    action, turnover_event = _loss_actions(actions, labelled.loc[sid])
                    if action is None:
                        unknown_terminal += 1
                        continue
                    has_frame = bool(action.get("has_freeze_frame", False)) if pd.notna(action.get("has_freeze_frame", False)) else False
                    kind, action_tags = _action_description(action, has_frame)
                    types[kind] += 1
                    turnover_events[turnover_event] += 1
                    for tag in action_tags:
                        tags[tag][kind] += 1
                    all_rows.append({"method": method, "fold": fold_no, "rule": rule_no, "sequence_id": sid,
                                     "last_own_action_type": kind, "turnover_event": turnover_event,
                                     "tags": action_tags})
                fold_sections.append({
                    "method": method, "cv_report": str(report_path),
                    "fold": fold_no, "rule": rule_no, "body": body,
                    "n_covered_negative_sequences": len(ids),
                    "n_last_own_actions_found": sum(types.values()),
                    "n_last_own_actions_unknown": unknown_terminal,
                    "last_own_action_types": dict(types),
                    "turnover_events": dict(turnover_events),
                    "overlapping_action_tags": {tag: dict(counts) for tag, counts in tags.items()},
                })

    lines = [
        "# تحلیل توصیفی اکشنِ مرتبط با لو دادن توپ",
        "",
        "این گزارش پس از کشف قانون‌ها ساخته می‌شود و هیچ ویژگیِ اکشنِ لو دادن توپ را به ورودی Popper اضافه نمی‌کند. هر قانون فقط روی سکانس‌های منفیِ فولدِ نگه‌داشته‌شدهٔ خودش اعمال شده است. در باخت مستقیم، اکشن ناموفقِ خودی شمرده می‌شود؛ اگر ابتدا حریف مالکیت را گرفته باشد، آخرین اکشن خودی پیش از اکشنِ دریافت/مالکیت حریف شمرده می‌شود.",
        "",
        "دستهٔ نوع اکشن برای هر سکانس فقط یک‌بار شمرده می‌شود. برچسب‌های پاس (جهت، طول، قطری‌بودن، تلاش خط‌شکنی و پاس خط‌شکن موفق) هم‌پوشان‌اند و جمع آن‌ها می‌تواند از تعداد پاس‌ها بیشتر شود. تلاش خط‌شکنی از هندسهٔ اکشن و خط تخمینی در فریم ۳۶۰ می‌آید؛ فقط تلاش موفق با نتیجهٔ موفق پاس در برچسب successful_line_break_proxy می‌آید.",
        "",
        "| فولد / قانون | قانون کشف‌شده | پوشش منفی | اکشن مرتبط ناموجود | اکشن خودیِ مرتبط (انحصاری) | رخداد ثبت‌شدهٔ turnover | برچسب‌های توصیفی پاس (هم‌پوشان) |",
        "|---|---|---:|---:|---|---|---|",
    ]
    if stale_reports:
        lines.insert(2, "گزارش‌های CV با نسخهٔ پروتکل قدیمی کنار گذاشته شدند؛ ابتدا آن مسیرها را با کد جاری دوباره اجرا کنید: " + ", ".join(f"`{Path(p).name}`" for p in stale_reports) + ".")
    for item in fold_sections:
        type_txt = _fmt_counter(Counter(item["last_own_action_types"]))
        tag_totals = Counter()
        for tag, counts in item["overlapping_action_tags"].items():
            tag_totals[tag] = sum(counts.values())
        body = ", ".join(item["body"])
        lines.append(f"| {item['method']} / {item['fold']} / {item['rule']} | `{body}` | {item['n_covered_negative_sequences']} | {item['n_last_own_actions_unknown']} | {type_txt} | {_fmt_counter(Counter(item['turnover_events']))} | {_fmt_counter(tag_totals)} |")
    lines += [
        "",
        "## محدودیت تفسیر",
        "",
        "این یک توصیف از آخرین اکشن خودیِ مرتبط با turnover در سکانس‌های برچسب‌خوردهٔ منفی است؛ نه برآورد علّی و نه اثبات اینکه قانون یا اکشن باعث ازدست‌رفتن توپ شده است. تعریف مرز سکانس، اتصال اکشن حریف به رخداد باخت و ثبت ناقص رویدادها در StatsBomb بر نتیجه اثر می‌گذارند. قانون‌های فولدهای مختلف نیز ممکن است یک سکانس یا مفهوم را به شکل متفاوتی پوشش دهند.",
    ]
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({
        "analysis_type": "post_discovery_descriptive_terminal_action_audit",
        "does_not_enter_learning": True,
        "cv_reports": [str(path) for _, path, _, _, _ in reports],
        "fold_rule_results": fold_sections,
        "sequence_rule_rows": all_rows,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote -> {OUT_MD}")
    print(f"wrote -> {OUT_JSON}")
    print(f"fold-rule rows: {len(fold_sections)}; covered negative sequence-rule rows: {len(all_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

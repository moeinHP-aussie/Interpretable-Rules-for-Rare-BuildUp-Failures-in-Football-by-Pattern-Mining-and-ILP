"""
Phase 7 -- build the coach report as ONE self-contained HTML file.

    python -m tactical_ilp.reporting.build_report            # (re)generate
    python -m tactical_ilp.reporting.build_report --check    # exit 1 if stale

WHAT MAKES THIS SYSTEMATIC
--------------------------
* The template below contains NO number and NO football claim.  Every figure
  comes from :mod:`facts`; every football phrase from :mod:`concept_dictionary`.
* Only rules with ``validation_status: approved`` in the coach YAML are shown as
  findings.  Rejected rules appear in their own section with the coach's reason.
  A rule still ``pending`` blocks nothing but is listed, never presented as a
  finding.
* The report is regenerated from the current data every run.  A fingerprint of
  the coach YAML + labels + example files is embedded; ``--check`` compares it,
  so CI (or you) can tell whether the file on disk matches the data.
* A "what changed since the last report" section is built by diffing the new
  facts against ``report_state.json`` saved by the previous run.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import sys
from pathlib import Path

import pandas as pd

from tactical_ilp import config
from tactical_ilp.reporting import plots
from tactical_ilp.reporting.concept_dictionary import lookup, short_label
from tactical_ilp.reporting.facts import (
    ReportFacts,
    attach_positions,
    collect,
    literal_witnesses,
    pick_examples,
    rule_matches,
    sequence_actions_with_positions,
    baseline_start_x,
    location_vs_baseline,
)
from tactical_ilp.evaluation.rule_ensemble import build_rule_ensemble
from tactical_ilp.reporting.rule_translator import parse_body
from tactical_ilp.reporting.terminal_actions import (
    ACTION_LABELS as TERMINAL_ACTION_LABELS,
    DIRECTIONS as TERMINAL_DIRECTIONS,
    LENGTHS as TERMINAL_LENGTHS,
    LINE_BREAK as TERMINAL_LINE_BREAK,
    RESULT_LABELS as TERMINAL_RESULT_LABELS,
    build_global_terminal_action_summaries,
)

REPORT_PATH = config.REPORTS_DIR / "coach_report.html"
STATE_PATH = config.REPORTS_DIR / "report_state.json"
ENSEMBLE_PATH = config.REPORTS_DIR / "phase5_weighted_rules.json"

_FA = str.maketrans("0123456789.", "۰۱۲۳۴۵۶۷۸۹٫")

ACTION_LABELS = {
    "pass": "پاس", "dribble": "دریبل", "shot": "شوت", "cross": "ارسال",
    "clearance": "دفع توپ", "interception": "توپ‌گیری", "tackle": "تکل",
    "keeper_save": "مهار دروازه‌بان", "goal_kick": "ضربهٔ دروازه", "goalkick": "ضربهٔ دروازه",
    "throw_in": "پرتاب اوت", "freekick": "ضربهٔ آزاد", "corner": "کرنر",
    "foul": "خطا", "bad_touch": "کنترل ناموفق", "offside": "آفساید", "out": "خروج توپ",
    "take_on": "دریبل یک‌به‌یک", "keeper_pick_up": "جمع‌کردن توپ توسط دروازه‌بان",
}
RESULT_LABELS = {
    "success": "موفق", "fail": "ناموفق", "offside": "آفساید",
    "owngoal": "گل‌به‌خودی", "success_with_penalty": "موفق با خطا",
}


def _compact_rule_label(name: str) -> str:
    """Short, unique tick labels for train/holdout comparisons."""
    if name.startswith("current_popper_"):
        return "P" + name.removeprefix("current_popper_")
    if name.startswith("candidate_"):
        tail = name.removeprefix("candidate_")
        return "C" + tail.split("_", 1)[0]
    if name.startswith("discovered_"):
        tail = name.removeprefix("discovered_")
        return "D" + tail.split("_", 1)[0]
    return name.replace("_", " ")


def fa(x, nd: int = 0) -> str:
    return (f"{x:.{nd}f}" if isinstance(x, float) else str(x)).translate(_FA)


def pct(x: float, nd: int = 1) -> str:
    return fa(x * 100, nd) + "٪"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def img(png: bytes, alt: str) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    return f'<img alt="{esc(alt)}" src="data:image/png;base64,{b64}">'


def action_sequence_html(actions: pd.DataFrame) -> str:
    """Render the ordered actions beside the sequence plot as a visual timeline."""
    if actions.empty:
        return ""
    colours = plots.actor_color_map(actions)
    rows = []
    for r in actions.sort_values("order").itertuples():
        player_id = getattr(r, "player_id", None)
        colour = colours.get(player_id, plots.UNKNOWN_PLAYER_COLOR)
        raw_type = getattr(r, "type_name", None)
        raw_result = getattr(r, "result_name", None)
        action_type = str(raw_type) if pd.notna(raw_type) else "unknown"
        result = str(raw_result) if pd.notna(raw_result) else ""
        player_name = getattr(r, "display_name", None)
        player_name = player_name if isinstance(player_name, str) and player_name.strip() else "نامشخص"
        result_class = "success" if result.startswith("success") else "fail" if result == "fail" else "other"
        rows.append(
            f'<li class="timeline-action {result_class}">'
            f'<span class="action-index" style="background:{colour}">{int(r.order)}</span>'
            f'<span class="timeline-copy"><b>{esc(ACTION_LABELS.get(action_type, action_type.replace("_", " ")))}</b>'
            f'<small><span class="player-swatch" style="background:{colour}"></span>{esc(player_name)}</small></span>'
            f'<span class="timeline-result">{esc(RESULT_LABELS.get(result, result.replace("_", " ") or "—"))}</span>'
            '</li>'
        )
    return (
        f'<div class="sequence-timeline" aria-label="توالی {fa(len(actions))} اکشن">'
        f'<div class="timeline-heading">توالی اکشن‌ها <span>{fa(len(actions))} اکشن</span></div>'
        f'<ol>{"".join(rows)}</ol></div>'
    )


# ---------------------------------------------------------------------------
# per-rule evidence (matches, witnesses, figures) -- all computed
# ---------------------------------------------------------------------------
BASELINE_X = None  # filled once per run in render()


def build_rule_evidence(
    facts: ReportFacts,
    rule: dict,
    labels: dict[str, str],
    *,
    match=None,
    used_example_ids: set[str] | None = None,
) -> dict:
    m = match or rule_matches(rule, facts.n_sequences, facts.n_negative)
    lits = [l for l in parse_body(rule["prolog_body"])]
    positive_lits = [l for l in lits if not l.negated]

    wit_by_seq: dict[str, dict[str, set[int]]] = {}
    wit_frames: dict[str, pd.DataFrame] = {}
    for l in positive_lits:
        w = literal_witnesses(l.base, m.matched)
        wit_frames[l.base] = attach_positions(w)
        for sid, g in w.groupby("sequence_id"):
            wit_by_seq.setdefault(sid, {})[short_label(l.base)] = set(g["order"])

    # how often do the rule's action-level literals fall on ONE action?
    action_lits = [l.base for l in positive_lits if not wit_frames[l.base].empty]
    same_action = None
    if len(action_lits) >= 2:
        sets = []
        for base in action_lits:
            fr = wit_frames[base]
            sets.append({(r.sequence_id, r.action_id) for r in fr.itertuples()})
        common = set.intersection(*sets)
        seqs_common = {s for s, _ in common}
        same_action = (len(seqs_common), len(m.matched))

    acts = sequence_actions_with_positions(m.matched)
    used = used_example_ids if used_example_ids is not None else set()
    ex_neg = pick_examples(
        acts, labels, wit_by_seq, True, 2,
        expected_witness_literals=len(action_lits),
        exclude_sequence_ids=used,
    )
    used.update(ex_neg)
    ex_ok = pick_examples(
        acts, labels, wit_by_seq, False, 1,
        expected_witness_literals=len(action_lits),
        exclude_sequence_ids=used,
    )
    used.update(ex_ok)

    figures = {}
    action_sequences = {}
    primary = next((l.base for l in positive_lits if not wit_frames[l.base].empty), None)
    if primary:
        figures["location"] = plots.location_map(
            wit_frames[primary], f"{short_label(primary)}  (action start, own half)")
    for index, sid in enumerate(ex_neg, 1):
        sequence_actions = acts[acts.sequence_id == sid].copy()
        action_sequences[sid] = sequence_actions
        seq_witnesses = wit_by_seq.get(sid, {})
        figures[f"neg_{index}"] = plots.sequence_example(
            sequence_actions, seq_witnesses, True, sid,
            shared_witness_orders=(
                set.intersection(*seq_witnesses.values())
                if len(seq_witnesses) >= 2 else set()
            ))
    if ex_ok:
        sid = ex_ok[0]
        sequence_actions = acts[acts.sequence_id == sid].copy()
        action_sequences[sid] = sequence_actions
        seq_witnesses = wit_by_seq.get(sid, {})
        figures["ok"] = plots.sequence_example(
            sequence_actions, seq_witnesses, False, sid,
            shared_witness_orders=(
                set.intersection(*seq_witnesses.values())
                if len(seq_witnesses) >= 2 else set()
            ))

    return {"match": m, "figures": figures, "wit": wit_frames, "same_action": same_action,
            "primary": primary, "examples": {"neg": ex_neg, "ok": ex_ok},
            "action_sequences": action_sequences}


def _ensemble_html(ensemble: dict, facts: ReportFacts) -> str:
    if ensemble.get("status") != "fitted":
        return '<div class="warn">برای برازش وزن‌ها، قواعد تأییدشده و split مسابقه‌ای کامل در دسترس نیست.</div>'
    tier_names = {
        "critical_association": "ارتباط مثبت قوی",
        "secondary_association": "ارتباط مثبت ثانویه",
        "protective_association": "ارتباط منفی",
        "weak_association": "ارتباط ضعیف",
        "pruned": "حذف‌شده با L1",
    }
    rows = []
    for i, item in enumerate(sorted(ensemble["rule_weights"], key=lambda x: -x["weight"]), 1):
        sentence = facts.sentences.get(item["name"])
        wording = sentence.headline if sentence else item["name"]
        rows.append(
            "<tr>"
            f"<td>{fa(i)}</td><td>{esc(wording)}</td>"
            f"<td>{fa(item['weight'], 3)}</td><td>{fa(item['odds_ratio'], 2)}</td>"
            f"<td>{esc(tier_names.get(item['tier'], item['tier']))}</td></tr>"
        )
    hm = ensemble["holdout_metrics"]
    holdout_text = (
        f"Brier: {fa(hm['brier_score'], 3)} · AUC: {fa(hm['roc_auc'], 3)} · "
        f"پیش‌بینی بالاتر از آستانه: {fa(hm['predicted_positive'])} از {fa(hm['n'])}"
        if hm["n"] else "برای split نگه‌داشته‌شده داده‌ای در دسترس نیست."
    )
    top = sorted(
        (record for record in ensemble["records"] if record["split"] == "holdout"),
        key=lambda record: -record["probability"],
    )[:12]
    scenario_rows = []
    for record in top:
        active = "، ".join(_compact_rule_label(name) for name in record["active_rules"]) or "—"
        contributions = "؛ ".join(
            f"{_compact_rule_label(item['rule'])}: {fa(item['logit_contribution'], 3)}"
            for item in record["rule_contributions"]
        ) or "—"
        scenario_rows.append(
            f"<tr><td><code>{esc(record['sequence_id'])}</code></td>"
            f"<td>{fa(record['match_id'])}</td><td>{pct(record['probability'])}</td>"
            f"<td>{esc(record['label'])}</td>"
            f"<td>{esc(active)}</td><td>{esc(contributions)}</td></tr>"
        )
    scenarios = (
        "<div class=\"table-scroll\"><table><thead><tr><th>سکانس نگه‌داشته‌شده</th><th>مسابقه</th>"
        "<th>احتمال برآوردی</th><th>پیامد واقعی</th><th>قواعد فعال</th>"
        "<th>سهم قواعد در لگیت</th></tr></thead><tbody>"
        + "".join(scenario_rows) + "</tbody></table></div>"
        if scenario_rows else "<p>سکانس نگه‌داشته‌شده‌ای برای نمایش موجود نیست.</p>"
    )
    return (
        '<div class="card"><p>مدل لاجستیک L1 روی شاخص‌های دودوییِ قواعد تأییدشده برازش شده است. '
        'وزن‌ها و نسبت شانس، ارتباط شرطی را نشان می‌دهند؛ علت تاکتیکی را اثبات نمی‌کنند. '
        f"ارزیابی holdout: {holdout_text}</p>"
        "<div class=\"table-scroll\"><table><thead><tr><th>اولویت</th><th>قاعده</th>"
        "<th>وزن</th><th>نسبت شانس</th><th>تفسیر آماری</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div></div>"
        '<div class="card"><h3>سکانس‌های پرریسک‌تر در دادهٔ نگه‌داشته‌شده</h3>'
        '<p class="meta">سهم‌ها تغییر خطی در لگیت هستند؛ احتمال خروجی تابع سیگموید از مجموع این سهم‌هاست.</p>'
        + scenarios + "</div>"
        + f"<p class=\"meta\">آستانهٔ نمایشی: {pct(ensemble['risk_threshold'])}. "
        f"{esc(ensemble['interpretation_warning'])}</p>"
    )


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
CSS = """
:root{--ink:#1b2733;--muted:#5d6d7e;--line:#dfe5ea;--bg:#f6f8fa;--card:#fff;
--neg:#c0392b;--ok:#2c7fb8;--warn:#b9770e;--warnbg:#fef5e7;--good:#1e8449}
@media(prefers-color-scheme:dark){:root{--ink:#e8edf2;--muted:#9fb0bf;--line:#2c3a47;
--bg:#10171e;--card:#17212a;--warnbg:#2b2110}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:Vazirmatn,Tahoma,"Segoe UI",sans-serif;line-height:1.95;font-size:16px}
main{max-width:900px;margin:0 auto;padding:28px 18px 80px}
h1{font-size:1.7rem;margin:.2rem 0}h2{font-size:1.3rem;margin:2.2rem 0 .6rem;
border-bottom:2px solid var(--line);padding-bottom:.3rem}h3{font-size:1.08rem;margin:1.4rem 0 .3rem}
.meta{color:var(--muted);font-size:.86rem}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px 18px;margin:14px 0}
.finding{border-right:5px solid var(--neg)}
.rejected{border-right:5px solid var(--muted);opacity:.93}
.finding.risky{border-right-color:var(--warn)}
.risk{background:var(--warnbg);border:2px solid var(--neg);border-radius:10px;padding:10px 14px;margin:10px 0}
.risk b{color:var(--neg)}
.warn{background:var(--warnbg);border:1px solid var(--warn);border-radius:10px;padding:10px 14px;margin:12px 0;color:var(--ink)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:12px 0}
.stat{background:var(--bg);border:1px solid var(--line);border-radius:10px;padding:8px 12px}
.stat b{display:block;font-size:1.25rem}.stat span{color:var(--muted);font-size:.82rem}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;margin:12px 0}
figure{margin:0;text-align:center}figure img{max-width:100%;height:auto;border:1px solid var(--line);
border-radius:8px;background:#fff}figcaption{font-size:.82rem;color:var(--muted);margin-top:4px}
code{direction:ltr;unicode-bidi:embed;background:var(--bg);border:1px solid var(--line);
border-radius:5px;padding:1px 6px;font-size:.85em}
ul{padding-right:1.3rem}li{margin:.25rem 0}
.tag{display:inline-block;padding:1px 10px;border-radius:99px;font-size:.78rem;border:1px solid var(--line)}
.tag.ok{color:var(--good);border-color:var(--good)}.tag.no{color:var(--neg);border-color:var(--neg)}
table{border-collapse:collapse;width:100%;font-size:.9rem}th,td{border:1px solid var(--line);padding:6px 10px;text-align:right}
th{background:var(--bg)}
details{margin:.6rem 0}summary{cursor:pointer;color:var(--muted)}
.sequence-example{min-width:0;padding:10px;border:1px solid var(--line);border-radius:10px;background:var(--bg)}
.sequence-example h4{margin:.15rem 0 .5rem;font-size:.95rem}
.sequence-details{margin:.55rem 0 0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:7px 9px}
.sequence-details summary{font-weight:600;font-size:.85rem}
.table-scroll{overflow-x:auto;max-height:370px;overflow-y:auto;margin-top:8px}
.sequence-action-table{min-width:430px;font-size:.82rem}
.sequence-action-table th{position:sticky;top:0;z-index:1}
.action-index{display:inline-flex;align-items:center;justify-content:center;min-width:27px;height:27px;border-radius:50%;color:#fff;font-weight:700;direction:ltr}
.player-swatch{display:inline-block;width:10px;height:10px;border-radius:50%;margin-left:7px;vertical-align:middle}

/* Visual language adapted from the supplied coach-report reference. */
:root{color-scheme:light;--bg:#f5f7fb;--card:#fff;--ink:#0f172a;--muted:#64748b;--line:#e5e9f2;
--primary:#1d4ed8;--primary-ink:#1e3a8a;--primary-soft:#eef2ff;--accent:#9f1239;--neg:#be123c;
--ok:#047857;--good:#047857;--warn:#b45309;--warnbg:#fffbeb;--surface:#fff;
--r-lg:18px;--r:12px;--shadow-sm:0 1px 2px rgba(15,23,42,.06)}
body{background:var(--bg);color:#3d4a5f;font-size:15.5px;-webkit-font-smoothing:antialiased}
body:before{content:"";display:block;height:5px;background:linear-gradient(90deg,var(--primary),var(--accent))}
main{max-width:1060px;padding:26px 20px 72px}
h1{font-size:clamp(1.55rem,1.1rem + 2vw,2.2rem);font-weight:900;color:var(--ink);line-height:1.4;letter-spacing:-.2px}
h2{color:var(--ink);border:0;padding:0;margin:0;font-size:1.35rem}
h3{color:var(--ink);font-weight:800}h3:before{content:"";display:inline-block;width:4px;height:18px;border-radius:4px;background:linear-gradient(180deg,var(--primary),var(--accent))}
.hero{position:relative;overflow:hidden;background:var(--card);border-bottom:1px solid var(--line);padding:28px max(20px,calc((100% - 1020px)/2)) 24px}
.hero:after{content:"";position:absolute;inset:0;pointer-events:none;background:radial-gradient(640px 220px at 88% -50%,rgba(159,18,57,.07),transparent 60%),radial-gradient(640px 220px at 12% -50%,rgba(29,78,216,.1),transparent 60%)}
.hero>*{position:relative;z-index:1}.hero .meta{font-size:.84rem;margin-bottom:16px}
.eyebrow{display:inline-flex;padding:4px 14px;border:1px solid #dbe3ff;border-radius:999px;background:var(--primary-soft);color:var(--primary);font-size:.78rem;font-weight:700}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:11px;margin-top:14px}
.kpi{padding:13px 15px;border:1px solid var(--line);border-radius:var(--r);background:linear-gradient(180deg,#fbfcfe,#fff);box-shadow:var(--shadow-sm)}
.kpi b{display:block;color:var(--primary-ink);font-size:1.45rem;font-weight:800}.kpi span{color:var(--muted);font-size:.78rem}.kpi.warn b{color:var(--accent)}
.tabbar{position:sticky;top:0;z-index:60;border-bottom:1px solid var(--line);background:rgba(245,247,251,.94);backdrop-filter:blur(12px)}
.tabs{display:flex;gap:8px;max-width:1060px;margin:0 auto;padding:9px 16px;overflow-x:auto;scrollbar-width:none}.tabs::-webkit-scrollbar{display:none}
.tab{flex:0 0 auto;cursor:pointer;padding:8px 15px;border:1px solid var(--line);border-radius:999px;background:#fff;color:var(--muted);font:600 .83rem Vazirmatn,Tahoma,sans-serif;white-space:nowrap;transition:background .15s,color .15s}
.tab:hover{background:var(--primary-soft);color:var(--primary-ink)}.tab.is-active{border-color:transparent;background:var(--primary);color:#fff;box-shadow:0 5px 14px -7px rgba(29,78,216,.65)}.tab.tab-muted.is-active{background:#64748b}
.panel{display:none}.panel.is-active{display:block;animation:fadeUp .24s ease}.panel-head{display:flex;align-items:flex-start;justify-content:space-between;gap:14px;flex-wrap:wrap;margin-bottom:15px}
.kicker{display:block;margin-bottom:2px;color:var(--accent);font-size:.75rem;font-weight:700}.panel-head h2{margin:0;color:var(--ink);font-size:1.35rem}
body:not(.js) .panel{display:block;margin-bottom:36px}body:not(.js) .tabbar{position:static}
@keyframes fadeUp{from{opacity:0;transform:translateY(7px)}to{opacity:1;transform:none}}
.card{padding:19px 22px;border-color:var(--line);border-radius:var(--r-lg);background:#fff;box-shadow:var(--shadow-sm)}
.finding{border-right:0;border-inline-start:5px solid var(--neg)}.finding.risky{border-inline-start-color:var(--warn)}.rejected{border-right:0;border-inline-start:5px solid #94a3b8}
.risk{border:1px solid #fecdd3;border-inline-start:5px solid var(--neg);border-radius:var(--r);background:#fff1f2;padding:12px 17px}.risk b{color:var(--neg)}
.warn{border:1px solid #fde68a;border-inline-start:5px solid var(--warn);border-radius:var(--r);background:var(--warnbg);padding:12px 17px;color:var(--ink)}.warn b{color:var(--warn)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:11px;margin:15px 0}.stat{padding:12px 14px;border:1px solid var(--line);border-radius:var(--r);background:#fff;box-shadow:var(--shadow-sm)}
.stat b{display:block;color:var(--primary-ink);font-size:1.3rem;font-weight:800}.stat span{color:var(--muted);font-size:.78rem}
.terminal-analysis{margin:18px 0;padding:15px;border:1px solid var(--line);border-radius:var(--r-lg);background:#f8faff}
.terminal-analysis h4{margin:.25rem 0;color:var(--ink);font-size:1rem}.terminal-analysis h5{margin:1rem 0 .35rem;color:var(--primary-ink)}
.terminal-pass-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px}
.terminal-pass-grid>div{min-width:0;padding:10px;border:1px solid var(--line);border-radius:var(--r);background:#fff}
.terminal-pass-grid .stats{grid-template-columns:repeat(auto-fit,minmax(120px,1fr))}
.figs{display:grid;grid-template-columns:1fr;gap:16px;margin:14px 0}.figs>figure{width:100%}figure{padding:11px;border:1px solid var(--line);border-radius:var(--r);background:#fff;box-shadow:var(--shadow-sm)}figure img{display:block;width:100%;height:auto;border:0;border-radius:var(--r)}figcaption{margin-top:9px;color:var(--muted);font-size:.8rem;line-height:1.7}
.tag{display:inline-flex;padding:3px 12px;border:1px solid var(--line);border-radius:999px;background:#fff;font-size:.76rem;font-weight:700}.tag.ok{color:var(--ok);border-color:#a7f3d0;background:#ecfdf5}.tag.no{color:var(--neg);border-color:#fecdd3;background:#fff1f2}.tag.warn{color:var(--warn);border-color:#fde68a;background:var(--warnbg)}
table{font-size:.86rem}th,td{padding:8px 11px;border:0;border-bottom:1px solid var(--line);text-align:start}th{background:var(--primary-soft);color:var(--primary-ink)}tbody tr:nth-child(even){background:#fafbfe}
.sequence-example{display:grid;grid-template-columns:minmax(0,1.35fr) minmax(250px,.65fr);align-items:stretch;gap:15px;margin:12px 0;padding:16px;border:1px solid var(--line);border-radius:var(--r-lg);background:#fff;box-shadow:var(--shadow-sm)}.sequence-example h4{grid-column:1/-1;margin:0;font-weight:800;color:var(--ink);font-size:1rem}.sequence-example>figure{align-self:start;padding:8px}.sequence-example>figure img{width:100%;height:auto}.sequence-example>figure figcaption{font-size:.76rem}.sequence-timeline{min-width:0;max-height:540px;overflow:auto;border:1px solid var(--line);border-radius:var(--r);background:#fafbfe;scrollbar-color:#cbd5e1 transparent;scrollbar-width:thin}.timeline-heading{position:sticky;top:0;z-index:2;display:flex;justify-content:space-between;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid var(--line);background:#f8faff;color:var(--primary-ink);font-size:.82rem;font-weight:800}.timeline-heading span{color:var(--muted);font-weight:600;font-size:.74rem}.sequence-timeline ol{position:relative;list-style:none;margin:0;padding:9px 12px}.sequence-timeline ol:before{content:"";position:absolute;top:18px;bottom:18px;right:25px;width:2px;background:linear-gradient(var(--primary),#cbd5e1)}.timeline-action{position:relative;display:grid;grid-template-columns:27px minmax(0,1fr) auto;align-items:center;gap:9px;margin:0;padding:7px 0;border-bottom:1px solid #edf0f6}.timeline-action:last-child{border-bottom:0}.timeline-action .action-index{position:relative;z-index:1;width:27px;height:27px;min-width:27px;border:2px solid #fff;box-shadow:0 0 0 1px var(--line);font-size:.73rem}.timeline-copy{display:flex;min-width:0;flex-direction:column;line-height:1.45}.timeline-copy b{overflow:hidden;color:var(--ink);font-size:.78rem;text-overflow:ellipsis;white-space:nowrap}.timeline-copy small{overflow:hidden;color:var(--muted);font-size:.68rem;text-overflow:ellipsis;white-space:nowrap}.timeline-copy .player-swatch{width:8px;height:8px;margin-left:5px}.timeline-result{padding:2px 7px;border-radius:999px;background:#f1f5f9;color:#64748b;font-size:.64rem;white-space:nowrap}.timeline-action.success .timeline-result{background:#ecfdf5;color:#047857}.timeline-action.fail .timeline-result{background:#fff1f2;color:#be123c}
.sequence-details{margin-top:10px;padding:0;border:1px solid var(--line);border-radius:var(--r);background:#fafbfe;overflow:hidden}.sequence-details summary{padding:10px 13px;color:var(--primary-ink);font-weight:700}.sequence-action-table{min-width:460px}
.chips{display:flex;flex-wrap:wrap;gap:8px;margin:12px 0}.chip{cursor:pointer;padding:6px 14px;border:1px solid #dbe3ff;border-radius:999px;background:var(--primary-soft);color:var(--primary-ink);font:600 .8rem Vazirmatn,Tahoma,sans-serif}.chip:hover{background:var(--primary);color:#fff}
footer{padding:20px;border-top:1px solid var(--line);background:#fff;color:var(--muted);font-size:.8rem;text-align:center}
@media(max-width:560px){.hero{padding:28px 16px 22px}main{padding:20px 14px 52px}.card{padding:16px}.tabs{padding-inline:12px}}
@media(max-width:760px){.sequence-example{grid-template-columns:1fr}.sequence-example h4{grid-column:1}.sequence-timeline{max-height:360px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}*,*:before,*:after{animation-duration:.01ms!important;transition-duration:.01ms!important}}
@media print{.tabbar{display:none!important}body{background:#fff}.panel,.panel.is-active,body.js .panel{display:block!important;break-after:page;animation:none}body.js .panel[hidden]{display:block!important}main{max-width:none}.card,.stat,.kpi,figure{box-shadow:none;break-inside:avoid}}
"""


def _stat(label: str, value: str) -> str:
    return f'<div class="stat"><b>{value}</b><span>{esc(label)}</span></div>'


def _terminal_analysis_html(summary: dict) -> str:
    """Render corpus-wide terminal-action summaries inside an approved rule card."""
    n = int(summary["n_covered_sequences"])

    def metrics(counts: dict, labels: dict[str, str], denominator: int) -> str:
        if denominator == 0:
            return '<div class="warn">داده‌ای برای این بخش در دسترس نیست.</div>'
        return '<div class="stats">' + "".join(
            _stat(labels[key], f"{fa(counts.get(key, 0))} · {pct(counts.get(key, 0) / denominator)}")
            for key in labels
        ) + "</div>"

    pass_n = int(summary["pass_count"])
    if pass_n:
        pass_html = (
            f'<p>تعداد پاس‌های پایانی: <b>{fa(pass_n)}</b>. درصد هر مورد از کل پاس‌های پایانی است.</p>'
            f'<div class="terminal-pass-grid"><div><b>جهت</b>{metrics(summary["pass_direction_counts"], TERMINAL_DIRECTIONS, pass_n)}</div>'
            f'<div><b>طول</b>{metrics(summary["pass_length_counts"], TERMINAL_LENGTHS, pass_n)}</div>'
            f'<div><b>تلاش خط‌شکنی</b>{metrics(summary["line_break_attempt_counts"], TERMINAL_LINE_BREAK, pass_n)}</div></div>'
        )
    else:
        pass_html = '<p>در سکانس‌های منطبق، پاس پایانی ثبت نشده است.</p>'
    return (
        '<section class="terminal-analysis">'
        + '<h4>توصیف اکشن پایانی در کل سکانس‌های منطبق</h4>'
        + f'<p>این توزیع از <b>{fa(n)}</b> سکانس برچسب‌خورده‌ای محاسبه شده که قانون در کل دیتاست روی آن‌ها فعال است؛ '
        + 'محدود به فولد آزمون نیست. درصد نوع اکشن و نتیجه بر مبنای همین پوشش است.</p>'
        + '<h5>اکشن پایانی خودی</h5>'
        + metrics(summary["own_action_counts"], TERMINAL_ACTION_LABELS, n)
        + '<h5>نتیجهٔ گردش مالکیت</h5>'
        + metrics(summary["terminal_result_counts"], TERMINAL_RESULT_LABELS, n)
        + '<h5>ویژگی‌های پاس پایانی</h5>' + pass_html
        + '<p class="meta">تلاش خط‌شکنی یک proxy هندسی از عبور مسیر پاس از خط دفاع تخمینی در فریم ۳۶۰ است؛ '
        + 'نبود فریم یا مختصات کافی در گروه «نامشخص» می‌آید.</p></section>'
    )


def risk_notes(rule: dict) -> list[str]:
    """Reasons a coach-APPROVED rule should still carry a visible caution.

    Uses only the flags phase 6 already computed (rule_metrics.py); this module
    invents no threshold.  The coach owns the decision -- an approved rule is
    never hidden or rejected here -- but the report must not present a rule that
    collapsed on held-out data in the same visual register as one that held up.
    """
    fl = rule.get("flags", {})
    out = []
    if fl.get("holdout_confidence_collapsed"):
        out.append("این الگو روی داده‌ٔ کنترل فروپاشید؛ یعنی احتمالاً ویژگی همان نمونه‌های آموزش بوده، نه یک الگوی پایدار.")
    if fl.get("low_support"):
        out.append("تعداد سکانس‌های پشتوانهٔ این الگو کم است و نتیجه ممکن است اتفاقی باشد.")
    return out


def render_rule(i: int, rule: dict, facts: ReportFacts, ev: dict) -> str:
    sent = facts.sentences[rule["name"]]
    m = ev["match"]
    tr, ho = rule["metrics"]["train"], rule["metrics"]["holdout"]

    figs = ""
    if "location" in ev["figures"]:
        figs += f"<figure>{img(ev['figures']['location'], 'Location of actions matching the primary condition')}<figcaption>Action starts matching the primary rule condition</figcaption></figure>"
    for sample_no, sid in enumerate(ev["examples"]["neg"], 1):
        fig_key = f"neg_{sample_no}"
        if fig_key not in ev["figures"]:
            continue
        actions = ev["action_sequences"][sid]
        figs += (
            f"<section class=\"sequence-example\"><h4>Negative sequence example {sample_no}</h4>"
            f"<figure>{img(ev['figures'][fig_key], 'negative sequence example')}"
            "<figcaption>Arrow colors and numbers identify the player performing each action. Matching colors indicate the same player. Rings and marker shapes identify actions that satisfy a tactical condition. Numbers show the selected team's action order; intervening opponent actions are omitted.</figcaption></figure>"
            f"{action_sequence_html(actions)}</section>"
        )
    if "ok" in ev["figures"] and ev["examples"]["ok"]:
        sid = ev["examples"]["ok"][0]
        actions = ev["action_sequences"][sid]
        figs += (
            "<section class=\"sequence-example\"><h4>Non-negative sequence example</h4>"
            f"<figure>{img(ev['figures']['ok'], 'non-negative sequence example')}"
            "<figcaption>Arrow colors and numbers identify the player performing each action. Matching colors indicate the same player. Rings and marker shapes identify actions that satisfy a tactical condition. Numbers show the selected team's action order; intervening opponent actions are omitted.</figcaption></figure>"
            f"{action_sequence_html(actions)}</section>"
        )

    extra = ""
    if ev["primary"]:
        loc = location_vs_baseline(ev["wit"][ev["primary"]], BASELINE_X)
        if loc:
            phrase = lookup(ev["primary"]).fa_present
            direction = "جلوتر از" if loc["delta"] > 0 else "عقب‌تر از"
            extra += (
                f"<li><b>محل رخداد:</b> اکشن‌هایی که این شرط را برقرار کرده‌اند ({fa(loc['n'])} اکشن) "
                f"به‌طور میانه در x = {fa(loc['median'], 1)} شروع شده‌اند؛ این {fa(abs(loc['delta']), 1)} متر "
                f"{direction} میانهٔ همهٔ اکشن‌های بازیسازی (x = {fa(loc['baseline_median'], 1)}) است.</li>")
    if ev["same_action"]:
        a, b = ev["same_action"]
        extra += (f"<li><b>هم‌زمانی شرط‌ها:</b> فقط در {fa(a)} از {fa(b)} سکانس ({pct(a / b, 0)}) این شرط‌ها روی "
                  f"<i>یک اکشنِ واحد</i> برقرارند؛ در بقیه در لحظه‌های مختلف سکانس رخ داده‌اند.</li>")

    caveats = "".join(f"<li>{esc(c)}</li>" for c in sent.caveats)
    risks = risk_notes(rule)
    risk_html = ('<div class="risk"><b>هشدار اعتبار:</b><ul>'
                 + "".join(f"<li>{esc(r)}</li>" for r in risks) + "</ul></div>") if risks else ""
    return f"""
<div class="card finding{' risky' if risks else ''}">
  <h3>الگو {fa(i)} <span class="tag ok">تأییدشده توسط مربی</span></h3>
  {risk_html}
  <p>{esc(sent.headline)}</p>
  <p>{esc(sent.evidence)}</p>
  <div class="stats">
    {_stat("نرخ پیامد منفی در سکانس‌های منطبق (کل داده)", pct(m.confidence))}
    {_stat("نرخ پایه پیامد منفی (همه سکانس‌ها)", pct(m.base_rate))}
    {_stat("Lift روی کل داده", fa(m.lift, 2))}
    {_stat("سکانس‌های منطبق", fa(len(m.matched)))}
  </div>
  {_terminal_analysis_html(ev["terminal_actions"])}
  <ul>{extra}</ul>
  <div class="figs">{figs}</div>
  <details><summary>یادداشت مربی و محدودیت‌های اندازه‌گیری</summary>
    <p><b>یادداشت مربی:</b> {esc(rule.get('validation_note') or '')}</p>
    <ul>{caveats}</ul>
    <p class="meta">قانون Prolog: <code>{esc(rule['prolog_body'])}</code></p>
  </details>
</div>"""


def render_rejected(rr) -> str:
    return f"""
<div class="card rejected">
  <h3><span class="tag no">ردشده</span> <code>{esc(rr.name)}</code></h3>
  <p><b>دلیل مربی:</b> {esc(rr.note)}</p>
  <p class="meta">اطمینان روی آموزش: {pct(rr.train_conf, 0)} (n={fa(rr.train_support)}) ← روی کنترل: {pct(rr.holdout_conf, 0)} (n={fa(rr.holdout_support)})</p>
</div>"""


def diff_against_previous(facts: ReportFacts) -> list[str]:
    """Human-readable list of what changed since the last generated report."""
    now = {
        "approved": sorted(r["name"] for r in facts.approved),
        "rejected": sorted(r.name for r in facts.rejected),
        "n_sequences": facts.n_sequences,
        "n_negative": facts.n_negative,
    }
    if not STATE_PATH.exists():
        return [], now
    old = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    out = []
    for name in sorted(set(now["approved"]) - set(old.get("approved", []))):
        out.append(f"قانون جدید تأییدشده: <code>{esc(name)}</code>")
    for name in sorted(set(old.get("approved", [])) - set(now["approved"])):
        out.append(f"قانونی که قبلاً تأیید شده بود دیگر تأیید نیست: <code>{esc(name)}</code>")
    for name in sorted(set(now["rejected"]) - set(old.get("rejected", []))):
        out.append(f"قانون جدید ردشده: <code>{esc(name)}</code>")
    if old.get("n_negative") != now["n_negative"] or old.get("n_sequences") != now["n_sequences"]:
        out.append(
            f"داده تغییر کرده: {fa(old.get('n_negative'))}/{fa(old.get('n_sequences'))} ← "
            f"{fa(now['n_negative'])}/{fa(now['n_sequences'])} سکانس منفی")
    return out, now


def render(facts: ReportFacts) -> tuple[str, dict]:
    global BASELINE_X
    BASELINE_X = baseline_start_x()
    labeled = pd.read_parquet(config.PROCESSED_DIR / "labeled_sequences.parquet")
    labels = labeled.set_index("sequence_id")["label"].to_dict()
    matches = {
        rule["name"]: rule_matches(rule, facts.n_sequences, facts.n_negative)
        for rule in facts.approved
    }
    # Allocate examples to rules with fewer available sequences first. This
    # prevents a broad rule from consuming the only usable illustration of a
    # narrower rule. Within each rule, pick_examples ranks witness coverage.
    allocation_order = sorted(
        facts.approved,
        key=lambda rule: (
            len(matches[rule["name"]].matched_negative),
            len(matches[rule["name"]].matched) - len(matches[rule["name"]].matched_negative),
            rule["name"],
        ),
    )
    used_example_ids: set[str] = set()
    evidence = {}
    for rule in allocation_order:
        evidence[rule["name"]] = build_rule_evidence(
            facts,
            rule,
            labels,
            match=matches[rule["name"]],
            used_example_ids=used_example_ids,
        )
    terminal_summaries = build_global_terminal_action_summaries(
        labeled=labeled,
        matched_by_rule={name: set(item["match"].matched) for name, item in evidence.items()},
    )
    for name, item in evidence.items():
        item["terminal_actions"] = terminal_summaries[name]

    ensemble = build_rule_ensemble(
        labeled=labeled,
        approved_rules=facts.approved,
        matches_by_rule={name: set(item["match"].matched) for name, item in evidence.items()},
        training_match_ids=facts.training_match_ids,
        holdout_match_ids=facts.holdout_match_ids,
    )
    ENSEMBLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    ENSEMBLE_PATH.write_text(json.dumps({
        "schema_version": 1,
        "model_fingerprint": facts.model_fingerprint,
        "report_fingerprint": facts.fingerprint,
        **ensemble,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    weights_by_name = {row["name"]: row for row in ensemble.get("rule_weights", [])}
    changes, state = diff_against_previous(facts)

    # ---- comparison figures ------------------------------------------------
    bar_rows = [{"label": "baseline", "conf": facts.negative_rate, "n": facts.n_sequences}]
    for i, r in enumerate(facts.approved, 1):
        m = evidence[r["name"]]["match"]
        bar_rows.append({"label": f"R{i}", "conf": m.confidence, "n": len(m.matched)})
    tvh_rows = []
    for i, r in enumerate(facts.approved, 1):
        t, h = r["metrics"]["train"], r["metrics"]["holdout"]
        tvh_rows.append({"label": f"R{i}", "train": t["confidence"], "holdout": h["confidence"],
                         "n_train": t["support"], "n_holdout": h["support"], "rejected": False})
    for rr in facts.rejected:
        tvh_rows.append({"label": _compact_rule_label(rr.name),
                         "train": rr.train_conf, "holdout": rr.holdout_conf,
                         "n_train": rr.train_support, "n_holdout": rr.holdout_support, "rejected": True})

    rule_panels = []
    for i, r in enumerate(facts.approved, 1):
        weight = weights_by_name.get(r["name"])
        weight_html = (
            f'<p class="meta">وزن شرطی: <b>{fa(weight["weight"], 3)}</b> · '
            f'نسبت شانس: <b>{fa(weight["odds_ratio"], 2)}</b></p>'
            if weight else ""
        )
        rule_panels.append(
            f'<section class="panel" id="rule-{i}" role="tabpanel" '
            f'aria-labelledby="tab-rule-{i}" tabindex="0" hidden>'
            f'<div class="panel-head"><div><span class="kicker">الگوی تأییدشده · {fa(i)}</span>'
            f'<h2>بررسی قانون {fa(i)}</h2></div>'
            '<span class="tag ok">تأییدشده توسط مربی</span></div>'
            f'{weight_html}'
            f'{render_rule(i, r, facts, evidence[r["name"]])}</section>'
        )
    if not rule_panels:
        rule_panels.append(
            '<section class="panel" id="rule-1" role="tabpanel" '
            'aria-labelledby="tab-rule-1" tabindex="0" hidden>'
            '<div class="panel-head"><div><span class="kicker">یافته‌ها</span>'
            '<h2>الگوهای تأییدشده</h2></div></div>'
            '<div class="warn">هنوز هیچ قانونی توسط مربی تأیید نشده است.</div></section>'
        )
    rule_tabs = "".join(
        f'<button class="tab" id="tab-rule-{i}" type="button" role="tab" '
        f'aria-controls="rule-{i}" aria-selected="false" data-target="rule-{i}">'
        f'قانون {fa(i)}</button>'
        for i, _ in enumerate(facts.approved, 1)
    )
    if not rule_tabs:
        rule_tabs = ('<button class="tab" id="tab-rule-1" type="button" role="tab" '
                     'aria-controls="rule-1" aria-selected="false" data-target="rule-1">'
                     'قوانین تأییدشده</button>')
    rejected_html = "".join(render_rejected(r) for r in facts.rejected) or "<p>قانون ردشده‌ای وجود ندارد.</p>"

    warn = ""
    if not facts.validation_ok:
        warn += ('<div class="warn"><b>هشدار:</b> فایل اعتبارسنجی مربی کامل نیست: '
                 + "؛ ".join(esc(p) for p in facts.validation_problems) + "</div>")
    if facts.pending:
        warn += ('<div class="warn"><b>در انتظار بررسی مربی:</b> '
                 + "، ".join(f"<code>{esc(n)}</code>" for n in facts.pending)
                 + " — این قوانین به‌عنوان یافته نمایش داده نمی‌شوند.</div>")

    changes_html = ("<ul>" + "".join(f"<li>{c}</li>" for c in changes) + "</ul>") if changes \
        else "<p>نسبت به گزارش قبلی تغییری نکرده است.</p>"

    n_risky = sum(1 for r in facts.approved if risk_notes(r))
    risky_summary = (f"از الگوهای تأییدشده، <b>{fa(n_risky)}</b> مورد هشدار اعتبار دارد (پایین‌تر ببینید)."
                     if n_risky else "")
    ensemble_html = _ensemble_html(ensemble, facts)
    cfg = facts.config_params
    reason_names = {
        "own_half_loss_not_regained_within_5s": "باخت در نیمهٔ خودی بدون بازپس‌گیری ظرف ۵ ثانیه",
        "loss_regained_within_5s": "بازپس‌گیری توپ ظرف ۵ ثانیه",
        "loss_outside_own_half": "باخت خارج از نیمهٔ خودی",
        "loss_location_unknown": "محل باخت نامشخص",
        "opponent_foul_stoppage": "توقف با خطای حریف در مرز سکانس",
        "no_possession_loss": "شواهدی از ازدست‌رفتن توپ نیست",
    }
    label_breakdown = "، ".join(
        f"{esc(reason_names.get(reason, reason))}: {fa(count)}"
        for reason, count in sorted(facts.negative_reason_counts.items())
    ) or "برای این اجرا موردی ثبت نشده است"
    method_html = f"""
  <p><b>تعریف سکانس</b>: شروع از نیمهٔ خودی (x ≤ {fa(cfg['BUILD_UP_START_MAX_X'], 1)}). خروج موفق فقط با کنترل توپ
  در یک ناحیهٔ مصوب ثبت می‌شود: یک‌چهارم پایانی (x ≥ {fa(cfg['BUILD_UP_DEEP_ENTRY_X'], 1)}، در هر عرض زمین)،
  زون ۱۴ (x بین {fa(cfg['BUILD_UP_ZONE_ENTRY_X'], 1)} و {fa(cfg['BUILD_UP_ZONE14_MAX_X'], 1)}؛
  y بین {fa(cfg['BUILD_UP_ZONE14_Y_MIN'], 1)} و {fa(cfg['BUILD_UP_ZONE14_Y_MAX'], 1)})، یا نیم‌فضاهای عمیق
  (x ≥ {fa(cfg['BUILD_UP_ZONE_ENTRY_X'], 1)}؛ y بین {fa(cfg['BUILD_UP_LEFT_HALF_SPACE_MIN_Y'], 1)} و
  {fa(cfg['BUILD_UP_LEFT_HALF_SPACE_MAX_Y'], 1)} یا بین {fa(cfg['BUILD_UP_RIGHT_HALF_SPACE_MIN_Y'], 1)} و
  {fa(cfg['BUILD_UP_RIGHT_HALF_SPACE_MAX_Y'], 1)}). مقصد پاس بدون دریافت کنترل‌شده کافی نیست؛ شوت نیز مسیر خروج محسوب می‌شود.</p>
  <p><b>پیامد منفی</b>: از دست‌رفتن توپ در نیمهٔ خودی (x ≤ {fa(cfg['OWN_HALF_MAX_X'], 1)}), بدون بازپس‌گیری کنترل توپ ظرف
  {fa(cfg['MIDFIELD_REGAIN_WINDOW_S'], 0)} ثانیه و بدون خطای حریف به‌عنوان نخستین اکشن پس از مرز سکانس. هر اکشن موفق
  کنترل‌کنندهٔ بارسلونا در این پنجره بازپس‌گیری محسوب می‌شود. شوت حریف و ورود او به یک‌سوم دفاعی فقط پیامدهای توصیفی هستند و در برچسب دخالت ندارند.</p>
  <p><b>ترکیب برچسب‌های منفی این اجرا:</b> {label_breakdown}.</p>
  <p>شوت حریف و ورود او به یک‌سوم دفاعی در پنجرهٔ توصیفی ۱۲ثانیه‌ای ثبت می‌شوند؛ این داده‌ها برای تفسیر outcome هستند، نه تعیین برچسب.</p>
  <p>دو عدد «اطمینان» در این گزارش دو پرسش متفاوت را جواب می‌دهند: عدد داخل جملهٔ هر الگو روی
  مجموعهٔ نمونه‌برداری‌شدهٔ یادگیری (تمام سکانس‌های منفی + دو برابرشان مثبت) حساب شده؛
  عدد «کل داده» روی هر {fa(facts.n_sequences)} سکانس. اولی برای سنجش کیفیت قانون است، دومی برای اینکه
  در عمل چقدر پرتکرار و چقدر خطرناک است. مقایسهٔ مستقیم آن‌ها نادرست است.</p>
  <p>مدل ریسک تجمیعی جداگانه و فقط با قواعد تأییدشدهٔ مربی ساخته می‌شود. برازش از همهٔ labelهای سکانس‌های
  matchهای train (با توزیع طبیعی کلاس‌ها) استفاده می‌کند؛ معیارهای holdout نیز روی matchهای جداشده محاسبه می‌شوند.
  ضرایب/نسبت شانس وابستگی شرطی‌اند، نه اثر علّی؛ به‌علت انتخاب قواعد در همین پروژه، holdout همچنان کاملاً دست‌نخورده نیست.</p>
  <p>داده: {fa(facts.n_matches)} بازی بارسلونا با فریم ۳۶۰ استرتبامب. بازیکنان غیرتوپ‌دار در فریم ۳۶۰ شناسه ندارند،
  پس هر «کدام حریف» موقعیتی است نه هویتی.</p>"""

    doc = f"""<!doctype html>
<html lang="fa" dir="rtl"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#f5f7fb">
<title>گزارش تحلیل تاکتیکی بازیسازی بارسلونا</title><style>{CSS}</style></head>
<body>
<header class="hero"><div class="hero-inner">
<span class="eyebrow">گزارش تحلیلی · ویژهٔ کادر فنی · {esc(facts.method_title)}</span>
<h1>گزارش تحلیل تاکتیکی بازیسازی بارسلونا</h1>
<p class="meta">تولید خودکار از داده‌های این اجرا · شناسهٔ مدل: <code>{esc(facts.model_fingerprint)}</code> · شناسهٔ گزارش: <code>{esc(facts.fingerprint)}</code></p>
<div class="kpis">
<div class="kpi"><b>{fa(facts.n_sequences)}</b><span>سکانس بازیسازی</span></div>
<div class="kpi"><b>{fa(facts.n_matches)}</b><span>مسابقه</span></div>
<div class="kpi warn"><b>{fa(facts.n_negative)}</b><span>سکانس با پیامد منفی</span></div>
<div class="kpi"><b>{pct(facts.negative_rate)}</b><span>نرخ پایه پیامد منفی</span></div>
<div class="kpi"><b>{fa(len(facts.approved))}</b><span>قانون تأییدشده</span></div>
<div class="kpi"><b>{fa(len(facts.rejected))}</b><span>قانون ردشده</span></div>
</div></div></header>
<nav class="tabbar" aria-label="بخش‌های گزارش"><div class="tabs" role="tablist" aria-label="بخش‌های گزارش">
<button class="tab is-active" id="tab-summary" type="button" role="tab" aria-controls="summary" aria-selected="true" data-target="summary">خلاصه</button>
{rule_tabs}
<button class="tab" id="tab-stability" type="button" role="tab" aria-controls="stability" aria-selected="false" data-target="stability">پایداری</button>
<button class="tab" id="tab-risk" type="button" role="tab" aria-controls="risk" aria-selected="false" data-target="risk">ریسک تجمیعی</button>
<button class="tab tab-muted" id="tab-rejected" type="button" role="tab" aria-controls="rejected" aria-selected="false" data-target="rejected">ردشده‌ها · {fa(len(facts.rejected))}</button>
<button class="tab" id="tab-changes" type="button" role="tab" aria-controls="changes" aria-selected="false" data-target="changes">تغییرات</button>
<button class="tab" id="tab-method" type="button" role="tab" aria-controls="method" aria-selected="false" data-target="method">روش و محدودیت‌ها</button>
</div></nav>
<main>
<section class="panel is-active" id="summary" role="tabpanel" aria-labelledby="tab-summary" tabindex="0">
<div class="panel-head"><div><span class="kicker">نمای کلی</span><h2>خلاصهٔ این اجرا</h2></div></div>
{warn}
<div class="card"><p>بازوی مدل‌سازی این گزارش: <b>{esc(facts.method_title)}</b> (<code>{esc(facts.method_key)}</code>).
از میان <b>{fa(facts.n_sequences)}</b> سکانس بازیسازی در <b>{fa(facts.n_matches)}</b> بازی،
<b>{fa(facts.n_negative)}</b> سکانس ({pct(facts.negative_rate)}) به «پیامد منفی» رسیده‌اند.
الگوریتم یادگیری قانون (Popper) تعدادی الگو پیشنهاد داد و مربی
<b>{fa(len(facts.approved))}</b> الگو را تأیید و <b>{fa(len(facts.rejected))}</b> الگو را رد کرد. {risky_summary}</p>
<figure>{img(plots.rate_bars(bar_rows), 'Negative-outcome rates')}<figcaption>Negative-outcome rate for the full dataset and sequences matching each rule</figcaption></figure></div>
<div class="chips" aria-label="رفتن به بخش‌های گزارش">
{''.join(f'<button type="button" class="chip" data-goto="rule-{i}">مشاهدهٔ قانون {fa(i)}</button>' for i, _ in enumerate(facts.approved, 1))}
<button type="button" class="chip" data-goto="stability">مشاهدهٔ پایداری</button>
</div>
<p>هر قانون ترکیبی از شرایط است که در سکانس‌های منطبق با پیامد منفی همراه بوده است. این‌ها <b>همبستگی‌اند نه علت</b>.</p>
</section>
{''.join(rule_panels)}
<section class="panel" id="risk" role="tabpanel" aria-labelledby="tab-risk" tabindex="0" hidden>
<div class="panel-head"><div><span class="kicker">تجمیع شواهد</span><h2>مدل ریسک قواعد تأییدشده</h2></div></div>
{ensemble_html}
</section>
<section class="panel" id="stability" role="tabpanel" aria-labelledby="tab-stability" tabindex="0" hidden>
<div class="panel-head"><div><span class="kicker">ارزیابی تعمیم</span><h2>پایداری روی داده‌ٔ کنترل</h2></div></div>
<div class="card"><p>هر قانون روی داده‌ای که Popper ندیده بود دوباره سنجیده شد. قانونی که فقط روی داده‌ٔ آموزش خوب است
ولی روی داده‌ٔ کنترل فرو می‌ریزد، «حفظ‌کردن» است نه یادگیری.</p>
<figure>{img(plots.train_vs_holdout(tvh_rows), 'Training and holdout confidence')}<figcaption>Confidence on training and holdout data (n = number of sequences)</figcaption></figure></div>
</section>
<section class="panel" id="rejected" role="tabpanel" aria-labelledby="tab-rejected" tabindex="0" hidden>
<div class="panel-head"><div><span class="kicker">بازبینی مربی</span><h2>الگوهای ردشده</h2></div></div>{rejected_html}
</section>
<section class="panel" id="changes" role="tabpanel" aria-labelledby="tab-changes" tabindex="0" hidden>
<div class="panel-head"><div><span class="kicker">ردیابی گزارش</span><h2>چه چیزی نسبت به گزارش قبلی تغییر کرد</h2></div></div>
<div class="card">{changes_html}</div></section>
<section class="panel" id="method" role="tabpanel" aria-labelledby="tab-method" tabindex="0" hidden>
<div class="panel-head"><div><span class="kicker">راهنمای تفسیر</span><h2>روش و محدودیت‌ها</h2></div></div>
<div class="card">{method_html}</div></section>
</main><footer>گزارش خودکار تحلیل تاکتیکی · اثرانگشت داده: <code>{esc(facts.fingerprint)}</code></footer>
<script>
(()=>{{
 const tabs=[...document.querySelectorAll('[role="tab"][data-target]')];
 const panels=[...document.querySelectorAll('[role="tabpanel"]')];
 function activate(id,focus=false){{
   const panel=document.getElementById(id); if(!panel) return;
   document.body.classList.add('js');
   panels.forEach(p=>{{const active=p===panel;p.hidden=!active;p.classList.toggle('is-active',active)}});
   tabs.forEach(t=>{{const active=t.dataset.target===id;t.classList.toggle('is-active',active);t.setAttribute('aria-selected',String(active));t.tabIndex=active?0:-1}});
   if(focus) panel.focus({{preventScroll:true}});
   history.replaceState(null,'','#'+id);
 }}
 tabs.forEach((tab,index)=>{{
   tab.addEventListener('click',()=>activate(tab.dataset.target));
   tab.addEventListener('keydown',e=>{{
     if(!['ArrowLeft','ArrowRight','Home','End'].includes(e.key)) return;
     e.preventDefault();let next=index;
     if(e.key==='ArrowLeft') next=(index+1)%tabs.length;
     if(e.key==='ArrowRight') next=(index-1+tabs.length)%tabs.length;
     if(e.key==='Home') next=0;if(e.key==='End') next=tabs.length-1;
     tabs[next].focus();activate(tabs[next].dataset.target);
   }});
 }});
 document.querySelectorAll('[data-goto]').forEach(b=>b.addEventListener('click',()=>activate(b.dataset.goto)));
 const initial=location.hash.slice(1);if(panels.some(p=>p.id===initial)) activate(initial);
}})();
</script></body></html>"""
    return doc, state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="exit 1 if the report on disk is stale")
    args = ap.parse_args(argv)

    facts = collect()
    if args.check:
        artifact_current = False
        if ENSEMBLE_PATH.exists():
            try:
                artifact_current = json.loads(ENSEMBLE_PATH.read_text(encoding="utf-8")).get(
                    "report_fingerprint"
                ) == facts.fingerprint
            except (OSError, json.JSONDecodeError):
                artifact_current = False
        if (
            not REPORT_PATH.exists()
            or facts.fingerprint not in REPORT_PATH.read_text(encoding="utf-8")
            or not artifact_current
        ):
            print("STALE: report does not match current data", file=sys.stderr)
            return 1
        print("up to date")
        return 0

    doc, state = render(facts)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(doc, encoding="utf-8")
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {REPORT_PATH}  ({REPORT_PATH.stat().st_size // 1024} KB)")
    print(f"wrote {ENSEMBLE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Generate and optionally serve an interactive Persian coach-review form.

Run after ``scripts/run_phase6.py``. With ``--serve``, the Save button writes
the reviewed YAML directly to ``data/reports/phase6_coach_validation.yaml``.
When opened as a static file, it downloads a JSON-formatted YAML document with
that filename instead.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tactical_ilp.config import REPORTS_DIR  # noqa: E402
from tactical_ilp.evaluation.coach_validation import COACH_VALIDATION_PATH  # noqa: E402
from tactical_ilp.reporting.rule_translator import parse_body  # noqa: E402

DEFAULT_HTML = ROOT / "coach_review.html"

ENGLISH_TERMS = {
    "seq_dropped_between_lines": "the ball carrier was positioned between the opponent's estimated defensive and midfield lines",
    "seq_dead_end_support": "no unobstructed passing lane to a visible teammate was available",
    "seq_dwelt_long": "the ball was held for more than 2.5 seconds",
    "seq_diagonal_pass": "a diagonal pass was played",
    "fast_sequence": "the sequence tempo was in the fastest decile",
    "seq_triangulation_opponent_trapped": "a passing triangle trapped an opponent",
    "seq_first_time_play": "an action was played within 0.5 seconds of receiving the ball",
    "seq_connecting_pass": "a successful, short, non-progressive pass was played",
    "seq_line_breaking_pass": "a successful pass crossed the estimated defensive line",
    "seq_pass_in_behind": "a pass was played beyond the estimated defensive line",
    "seq_line_pass": "a successful pass stayed in the same non-central lane",
    "seq_switch_to_left": "a successful switch reached the left half-space or wing",
    "seq_switch_to_right": "a successful switch reached the right half-space or wing",
    "seq_aerial_pass": "a pass was labelled High Pass by StatsBomb",
    "seq_long_pass": "a pass travelled at least 30 metres",
    "seq_rest_defence_exposed": "fewer than two visible outfield teammates were at least 5 metres behind the ball",
    "seq_under_pressure": "an action was performed under opponent pressure",
    "seq_flank_dribble": "a dribble began in an outer wing lane",
    "seq_flank_line_carry": "a dribble advanced at least 5 metres in the same wing lane",
    "seq_flank_cut_inside": "a dribble moved from a wing into a half-space or the central channel",
    "seq_central_long_carry": "a carry in a non-wing lane covered at least 3.5 metres",
    "seq_central_take_on": "a take-on was attempted in a non-wing lane",
    "seq_central_backward_pass": "a pass from a non-wing lane travelled at least 7 metres, including at least 5 metres backwards",
    "seq_flank_backward_carry": "a carry moved at least 4 metres backwards within the same wing lane",
    "seq_flank_to_center_progression": "a pass progressed at least 5 metres from a wing into a half-space or the central channel, covering at least 10 metres",
    "seq_flank_to_center": "a progressive pass moved from a wing into a half-space or the central channel",
    "seq_pass_exchange": "two distinct players exchanged successful passes in both directions",
    "seq_central_switch": "two consecutive successful passes switched through the central channel to the opposite flank",
    "seq_recycle_to_same_flank": "two consecutive successful passes moved from a flank through the centre and back to that flank",
    "seq_center_cycling": "two consecutive successful passes began and ended in central lanes",
    "seq_supported_action": "an action had at least one visible teammate with an open passing lane",
    "seq_dead_end_support": "no visible teammate had an unobstructed passing lane",
    "seq_rondo_cage": "a rondo-like structure enclosed an opponent",
    "seq_press_resistance": "the team retained possession under pressure",
    "seq_counter_pressed": "the action followed a counter-press",
    "seq_persistent_half_space_occupier": "a player repeatedly occupied a half-space",
    "seq_switch_of_play": "the ball was switched across the pitch",
    "seq_open_support": "an action had a visible teammate with an open passing lane",
    "seq_no_open_support": "no visible teammate had an open passing lane",
    "opponent_visible_majority": "opponents formed the visible majority in the freeze frame",
    "slow_sequence": "the sequence tempo was slow",
    "fast_sequence": "the sequence tempo was in the fastest decile",
    "keeper_played_open_play": "the goalkeeper participated in open play",
    "seq_pattern": "ordered tactical pattern",
}


def _literal_names(rule: dict) -> list[str]:
    return re.findall(r"([a-z][a-z0-9_]*?)\(\w+\)", rule.get("prolog_body", ""))


def _non_pattern_literals(rule: dict):
    names = [name for name in _literal_names(rule) if not re.fullmatch(r"seq_pattern_\d+", name)]
    return parse_body(", ".join(f"{name}(S)" for name in names)) if names else []


def _english_description(rule: dict) -> str:
    names = _literal_names(rule)
    patterns = [name for name in names if re.fullmatch(r"seq_pattern_\d+", name)]
    phrases = [ENGLISH_TERMS.get(name, name.replace("_", " ")) for name in names if name not in patterns]
    if patterns and phrases:
        return "This rule holds when the ordered pattern below occurs together with: " + "; and ".join(phrases) + "."
    if patterns:
        return "This rule holds when the ordered pattern below occurs in the sequence."
    if phrases:
        return "In this sequence: " + "; and ".join(phrases) + "."
    return "No reviewed tactical description is available for this rule."


def _description(rule: dict) -> str:
    names = _literal_names(rule)
    patterns = [name for name in names if re.fullmatch(r"seq_pattern_\d+", name)]
    phrases = [literal.phrase for literal in _non_pattern_literals(rule)]
    if patterns:
        conjunction = " و ".join(phrases)
        if conjunction:
            return f"این قانون با الگوی ترتیبیِ شرح‌داده‌شده و شرط‌های «{conjunction}» برقرار می‌شود."
        return "این قانون با وقوع الگوی ترتیبیِ شرح‌داده‌شده در سکانس برقرار می‌شود."
    if phrases:
        return "در این توالی: " + "؛ و ".join(phrases) + "."
    return str(rule.get("human_readable") or "توضیح تاکتیکی برای این قانون ثبت نشده است.")


def _pattern_item(item: dict | str) -> dict:
    if isinstance(item, str):
        value = item.removeprefix("tactical:")
        parts = value.split("@")
        parts += [""] * (4 - len(parts))
        return {"predicate": parts[0], "action_type": parts[1], "lane": parts[2], "third": parts[3]}
    if isinstance(item, dict):
        symbol = str(item.get("symbol", ""))
        if symbol.startswith("tactical:") and not item.get("predicate"):
            return _pattern_item(symbol)
        return item
    return {}


def _pattern_details_from_feature(predicate: str, feature: dict) -> dict | None:
    chain = feature.get("pattern_chain")
    if not isinstance(chain, list) or not chain:
        pattern = feature.get("pattern")
        if isinstance(pattern, list):
            chain = [{"step": i, "items": event} for i, event in enumerate(pattern, start=1)]
    if not isinstance(chain, list) or not chain:
        supplied = feature.get("pattern_details")
        return supplied if isinstance(supplied, dict) and supplied.get("steps") else None

    fa_action = {"pass": "پاس", "cross": "ارسال", "carry": "حمل توپ", "dribble": "دریبل", "shot": "شوت"}
    en_action = {"pass": "pass", "cross": "cross", "carry": "carry", "dribble": "dribble", "shot": "shot"}
    fa_lane = {"left_wing": "بال چپ", "left_half_space": "نیم‌فضای چپ", "central_channel": "کانال مرکزی", "right_half_space": "نیم‌فضای راست", "right_wing": "بال راست", "left_channel": "کانال چپ", "right_channel": "کانال راست"}
    en_lane = {"left_wing": "left wing", "left_half_space": "left half-space", "central_channel": "central channel", "right_half_space": "right half-space", "right_wing": "right wing", "left_channel": "left channel", "right_channel": "right channel"}
    fa_third = {"defensive_third": "یک‌سوم دفاعی", "middle_third": "یک‌سوم میانی", "final_third": "یک‌سوم هجومی"}
    en_third = {"defensive_third": "defensive third", "middle_third": "middle third", "final_third": "final third"}
    fa_steps, en_steps = [], []
    for index, event in enumerate(chain, start=1):
        items = event.get("items", []) if isinstance(event, dict) else event
        fa_parts, en_parts = [], []
        for raw_item in items if isinstance(items, list) else []:
            item = _pattern_item(raw_item)
            name = str(item.get("predicate") or item.get("name") or "tactical event")
            try:
                phrase_fa = parse_body(f"seq_{name}(S)")[0].phrase
            except (KeyError, ValueError):
                phrase_fa = name.replace("_", " ")
            phrase_en = ENGLISH_TERMS.get(f"seq_{name}", name.replace("_", " "))
            action_type = str(item.get("action_type") or "")
            lane, third = str(item.get("lane") or ""), str(item.get("third") or "")
            fa_location = " و ".join(v for v in (fa_lane.get(lane), fa_third.get(third)) if v)
            en_location = ", ".join(v for v in (en_lane.get(lane), en_third.get(third)) if v)
            fa_action_type = fa_action.get(action_type)
            en_action_type = en_action.get(action_type)
            fa_parts.append(phrase_fa + (f" ({fa_action_type})" if fa_action_type else "") + (f" در {fa_location}" if fa_location else ""))
            en_parts.append(phrase_en + (f" ({en_action_type})" if en_action_type else "") + (f" in {en_location}" if en_location else ""))
        if fa_parts:
            fa_steps.append(f"گام {index}: " + " و ".join(fa_parts))
            en_steps.append(f"Step {index}: " + " and ".join(en_parts))
    if not fa_steps:
        return None
    gap = max((int(item.get("max_intervening_actions", 0)) for item in feature.get("step_distance_constraints", []) if isinstance(item, dict)), default=0)
    number = re.search(r"(\d+)", predicate).group(1)
    fa_title = f"جزئیات الگوی ترتیبی {number}"
    en_title = f"Ordered pattern {number} — expanded steps"
    if len(fa_steps) > 1 and gap:
        fa_steps[-1] += f" (با حداکثر {gap} اکشن میانجی بین گام‌ها)"
        en_steps[-1] += f" (up to {gap} intervening actions between steps)"
    return {"title": fa_title, "steps": fa_steps, "title_en": en_title, "steps_en": en_steps}


def _pattern_details(rule: dict) -> list[dict]:
    """Expand final-model or fold-local ordered patterns into FA/EN steps."""
    details = []
    metadata = rule.get("pattern_metadata", {})
    metadata = metadata if isinstance(metadata, dict) else {}
    for predicate in _literal_names(rule):
        if not re.fullmatch(r"seq_pattern_\d+", predicate):
            continue
        feature = metadata.get(predicate)
        detail = _pattern_details_from_feature(predicate, feature) if isinstance(feature, dict) else None
        if detail is None:
            try:
                literal = parse_body(f"{predicate}(S)")[0]
            except (KeyError, ValueError):
                continue
            phrase = literal.phrase.removeprefix("توالی تاکتیکی:").strip()
            phrase = re.split(r"؛\s*فاصلهٔ هر گام", phrase, maxsplit=1)[0]
            number = re.search(r"(\d+)", predicate).group(1)
            detail = {"title": f"جزئیات الگوی ترتیبی {number}",
                      "steps": [step.strip() for step in phrase.split("؛ سپس ") if step.strip()],
                      "title_en": f"Ordered pattern {number} — expanded steps",
                      "steps_en": _pattern_steps_en(rule, predicate)}
        details.append(detail)
    return details


def _pattern_steps_en(rule: dict, predicate: str) -> list[str]:
    """Translate pattern items from the report's stored pattern metadata."""
    report_path = REPORTS_DIR / "phase5_sequential_training.json"
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        feature = next(row for row in report.get("pattern_facts", {}).get("features", [])
                       if row.get("predicate") == predicate)
    except (OSError, ValueError, StopIteration, AttributeError):
        return ["Ordered tactical event"]
    result = []
    for event in feature.get("pattern_chain", []):
        for item in event.get("items", []):
            name = item.get("predicate", "")
            phrase = ENGLISH_TERMS.get(f"seq_{name}", name.replace("_", " "))
            lane = item.get("lane")
            third = item.get("third")
            location = " — ".join(part.replace("_", " ") for part in (third, lane) if part and part != "unknown")
            result.append(f"{phrase} ({location})" if location else phrase)
    return result or ["Ordered tactical event"]


def _default_status(rule: dict) -> str:
    metrics = rule.get("metrics", {})
    holdout = metrics.get("holdout", {})
    drop = metrics.get("holdout_drop_ratio")
    # No firing means no estimate of holdout confidence; do not turn an
    # unobserved rare rule into an automatic rejection.
    if holdout.get("support") == 0:
        prior = rule.get("validation_status")
        return prior if prior in {"approved", "rejected"} else ""
    if (isinstance(drop, (int, float)) and drop >= 1.0) or holdout.get("confidence") == 0:
        return "rejected"
    prior = rule.get("validation_status")
    return prior if prior in {"approved", "rejected"} else ""


def build_html(document: dict) -> str:
    """Build the review form with a persistent Persian/English view switch."""
    payload = json.dumps(document, ensure_ascii=False).replace("</", "<\\/")
    return r'''<!doctype html><html lang="fa" dir="rtl"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Coach review</title><link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;600;700;800&display=swap" rel="stylesheet"><style>
:root{--ink:#eafcf1;--muted:#9fc4ad;--line:rgba(167,243,208,.14);--green:#22c55e;--green2:#4ade80}*{box-sizing:border-box}body{margin:0;padding:28px 18px 56px;font-family:Vazirmatn,Tahoma,Arial,sans-serif;line-height:1.9;color:var(--ink);background:linear-gradient(180deg,#050f0b,#0a1c13 55%,#081510);background-attachment:fixed}body.en{font-family:Arial,sans-serif}main{max-width:960px;margin:auto}.hero,.intro,.rule{border:1px solid var(--line);border-radius:18px;padding:20px;margin:18px 0;background:#ffffff0a}.hero{padding:30px}.badge{display:inline-block;border-radius:999px;padding:4px 12px;margin:4px 0 12px;font-size:.88rem;font-weight:700;border:1px solid}.badge.primary{color:#bbf7d0;background:#14532d66;border-color:#4ade8066}.badge.candidate{color:#fde68a;background:#78350f66;border-color:#fbbf2466}.topline{display:flex;justify-content:space-between;align-items:center}h1{font-size:clamp(1.5rem,4vw,2.1rem)}.muted,small{color:var(--muted)}.chips{display:flex;gap:12px;flex-wrap:wrap}.chip{border:1px solid #ffffff22;border-radius:14px;padding:8px 18px;min-width:150px}.chip small{display:block}.body{direction:ltr;text-align:left;background:#03110bcc;padding:12px;border:1px dashed #86efac55;border-radius:12px;overflow-wrap:anywhere;color:#a7f3d0}.metrics{display:block;background:#02140c80;border-radius:14px;padding:12px 16px;margin:14px 0}fieldset{border:1px dashed #a7f3ac38;border-radius:14px;padding:10px 14px}label.choice{margin-inline-end:12px;font-weight:600}.choice input{position:absolute;opacity:0}.choice span{display:inline-block;padding:8px 24px;border-radius:999px;border:1px solid #ffffff35;cursor:pointer}.choice.approve input:checked+span{background:linear-gradient(135deg,var(--green2),#16a34a);color:#032010}.choice.reject input:checked+span{background:linear-gradient(135deg,#f87171,#dc2626);color:white}textarea{display:block;width:min(100%,900px);min-height:76px;margin-top:8px;padding:12px;font:inherit;color:var(--ink);background:#03110bcc;border:1px solid #a7f3ac33;border-radius:12px}button{background:linear-gradient(135deg,var(--green2),#16a34a);color:#04220f;border:0;border-radius:14px;padding:12px 24px;font:inherit;font-weight:800;cursor:pointer}#language{background:#ffffff12;color:var(--ink);border:1px solid var(--line);padding:6px 14px}#status{margin:12px 0;font-weight:bold;color:#a7f3d0}details{margin:12px 0;background:#ffffff08;border:1px solid var(--line);border-radius:12px;padding:10px 16px}summary{cursor:pointer;color:#a7f3d0;font-weight:700}li{padding:4px 0;color:#d5e8dc}</style></head><body><main><header class="hero"><div class="topline"><b id="phase"></b><button id="language" type="button">EN</button></div><h1 id="title"></h1><p id="tagline" class="muted"></p><div class="chips"><span class="chip"><small id="method-label"></small><b id="method"></b></span><span class="chip"><small id="count-label"></small><b id="count"></b></span></div></header><section class="intro"><p id="intro"></p><details><summary id="guide-title"></summary><p id="instructions"></p></details><p id="warning"></p></section><div id="rules"></div><button id="save"></button><div id="status" role="status"></div><footer id="footer"></footer></main><script>
const doc=__PAYLOAD__;
const T={
fa:{phase:'🏆 فاز ۶',title:'فرم بازبینی مربی — فاز ۶',tagline:'بازبینی تاکتیکی قانون‌های کشف‌شده و ثبت تصمیم نهایی مربی',method:'روش استخراج',count:'تعداد قانون‌ها',intro:'قانون‌ها را از نظر تاکتیکی بررسی کنید. یادداشت اختیاری است. تصمیم نهایی با مربی است.',guide:'📘 راهنمای تصمیم‌گیری',save:'💾 ذخیرهٔ تصمیم‌ها در فایل YAML',footer:'سامانهٔ اعتبارسنجی دانش تاکتیکی • فاز ۶ 🏟️',approve:'✅ تأیید',reject:'🟥 رد',legend:'تصمیم مربی',note:'یادداشت (اختیاری)',train:'🎯 آموزش',holdout:'🧪 نگه‌داشته‌شده',support:'حمایت',confidence:'دقت',primary:'قانون برنامهٔ نهایی',intermediate:'نامزد میانی CV',candidateEvidence:'شواهد آزمون CV',fold:'فولد',missing:'توضیح تاکتیکی ثبت نشده است.',error:'برای قانون NAME تصمیم تأیید/رد را انتخاب کنید.',saved:'تصمیم‌ها ثبت شد.',downloaded:'فایل YAML دانلود شد؛ آن را در مرحلهٔ بعد بارگذاری کنید.',cvrare:'نامزد نادر — شواهد کافی نیست',cvunobserved:'در CV فعال نشد؛ دقت قابل محاسبه نیست',cvnotfound:'در فولدهای CV دوباره کشف نشد',cvpromising:'نامزد امیدوارکننده — بررسی شود',cvreview:'شواهد CV — نیازمند بازبینی',cvnote:'این مورد رد یا تأیید خودکار نشده است. برای قضاوت، بازی‌های تازه لازم است.'},
en:{phase:'🏆 Phase 6',title:'Coach review form — Phase 6',tagline:'Review discovered rules tactically and record the coach decision',method:'Discovery method',count:'Number of rules',intro:'Review each rule in its tactical context. Notes are optional; the final decision belongs to the coach.',guide:'📘 Decision guide',save:'💾 Save decisions to YAML',footer:'Tactical knowledge validation • Phase 6 🏟️',approve:'✅ Approve',reject:'🟥 Reject',legend:'Coach decision',note:'Note (optional)',train:'🎯 Training',holdout:'🧪 Holdout',support:'Support',confidence:'Confidence',primary:'Final-program rule',intermediate:'Intermediate CV candidate',candidateEvidence:'CV test evidence',fold:'Fold',missing:'No reviewed tactical description is available.',error:'Choose approve or reject for rule NAME.',saved:'Decisions saved.',downloaded:'The YAML file was downloaded. Upload it in the next step.',cvrare:'Rare candidate — insufficient evidence',cvunobserved:'No CV firings; precision cannot be estimated',cvnotfound:'Not rediscovered in CV folds',cvpromising:'Promising CV candidate — review',cvreview:'CV evidence — review required',cvnote:'This rule was not auto-approved or rejected. Fresh matches are needed to assess it.'}
};
const root=document.getElementById('rules');const savedLanguage=localStorage.getItem('tacticalIlpCoachLanguage');let lang=savedLanguage==='en'?'en':'fa';
function render(){if(root.childElementCount){(doc.rules||[]).forEach((r,i)=>{const selected=document.querySelector(`input[name="status-${i}"]:checked`),note=document.querySelector(`textarea[name="note-${i}"]`);if(selected)r.validation_status=selected.value;if(note)r.validation_note=note.value})}const t=T[lang];document.documentElement.lang=lang==='fa'?'fa':'en';document.documentElement.dir=lang==='fa'?'rtl':'ltr';document.body.classList.toggle('en',lang==='en');for(const [id,key] of [['phase','phase'],['title','title'],['tagline','tagline'],['method-label','method'],['count-label','count'],['intro','intro'],['guide-title','guide'],['save','save'],['footer','footer']])document.getElementById(id).textContent=t[key];document.getElementById('method').textContent=doc.source_method||'—';document.getElementById('count').textContent=(doc.rules||[]).length;document.getElementById('language').textContent=lang==='fa'?'EN':'FA';document.getElementById('instructions').textContent=lang==='fa'?(doc.instructions||''):(doc.instructions_en||'Review each candidate using holdout evidence and tactical context. Record approve/reject and an optional note.');document.getElementById('warning').textContent=lang==='fa'?(doc.candidate_selection_warning||''):(doc.candidate_selection_warning_en||'');root.innerHTML='';(doc.rules||[]).forEach((r,i)=>{const c=document.createElement('section');c.className='rule';const h=document.createElement('h2');h.textContent=`${i+1}. ${r.name||(lang==='fa'?'قانون':'Rule')}`;c.append(h);const badge=document.createElement('span');badge.className='badge '+(r.category==='intermediate_candidate'?'candidate':'primary');badge.textContent=r.category==='intermediate_candidate'?t.intermediate:t.primary;c.append(badge);const cvStatus=r.cv_candidate_status||'';if(cvStatus){const status=document.createElement('span');status.className='badge candidate';status.textContent=({rare_unobserved:t.cvunobserved,not_rediscovered_in_cv:t.cvnotfound,rare_low_support:t.cvrare,promising_single_fold:t.cvpromising,review_candidate:t.cvreview}[cvStatus]||t.cvrare);c.append(status);const evidence=document.createElement('p');evidence.className='muted';evidence.textContent=lang==='fa'?(r.cv_evidence_note||t.cvnote):({rare_unobserved:`No out-of-fold firings across ${r.cross_validation?.evaluated_match_count??0} held-out matches; precision and lift are undefined.`,rare_low_support:`Only ${r.cross_validation?.support??0} out-of-fold firings across ${r.cross_validation?.triggered_match_count??0} distinct matches; evidence is too sparse for a decision.`,not_rediscovered_in_cv:t.cvnote}[cvStatus]||r.cv_evidence_note||t.cvnote);c.append(evidence)}const p=document.createElement('p');p.textContent=lang==='fa'?(r.human_readable||t.missing):(r.human_readable_en||t.missing);c.append(p);const details=lang==='fa'?(r.pattern_details||[]):(r.pattern_details||[]).map(d=>({title:d.title_en||d.title,steps_en:d.steps_en||d.steps}));if(details.length){for(const d of details){const box=document.createElement('details');box.open=true;const s=document.createElement('summary');s.textContent=d.title;box.append(s);const ol=document.createElement('ol');for(const step of d.steps||d.steps_en||[]){const li=document.createElement('li');li.textContent=step;ol.append(li)}box.append(ol);c.append(box)}}const logic=document.createElement('div');logic.className='body';logic.textContent=lang==='fa'?(r.display_rule||r.prolog_body||''):(r.display_rule_en||r.prolog_body||'');c.append(logic);const a=r.metrics?.train||{},b=r.metrics?.holdout||{},pct=x=>x==null?'—':(100*x).toFixed(1)+(lang==='fa'?'٪':'%');const m=document.createElement('div');m.className='metrics';m.innerHTML=`${t.train}: ${t.support} ${a.support??'—'} | ${t.confidence} ${pct(a.confidence)} | Lift ${a.lift??'—'}<br>${t.holdout}: ${t.support} ${b.support??'—'} | ${t.confidence} ${pct(b.confidence)} | Lift ${b.lift??'—'}`;c.append(m);if(r.cross_validation){const cv=document.createElement('div');cv.className='metrics';const e=r.cross_validation;cv.textContent=`${lang==='fa'?'CV خارج‌ازنمونه':'Match-grouped CV'}: ${t.support} ${e.support??'—'} | ${t.confidence} ${pct(e.precision)} | Lift ${e.lift??'—'} | ${lang==='fa'?'فولدهای کامل / بازی‌های ارزیابی‌شده':'complete folds / matches evaluated'} ${e.complete_folds_found??0} / ${e.evaluated_match_count??0} | ${lang==='fa'?'بازی‌های دارای firing':'matches with rule firings'} ${e.triggered_match_count??0}`;c.append(cv)}if(r.candidate_metrics){const cm=document.createElement('div');cm.className='metrics';const ev=r.candidate_metrics;cm.textContent=`${t.candidateEvidence}: ${t.support} ${ev.support??'—'} | Lift ${ev.lift??'—'}${r.discovery_fold?` | ${t.fold} ${r.discovery_fold}`:''}`;c.append(cm)}const fs=document.createElement('fieldset'),legend=document.createElement('legend');legend.textContent=t.legend;fs.append(legend);for(const [value,label] of [['approved',t.approve],['rejected',t.reject]]){const l=document.createElement('label');l.className='choice '+(value==='approved'?'approve':'reject');const input=document.createElement('input');input.type='radio';input.name=`status-${i}`;input.value=value;input.required=true;input.checked=r.validation_status===value;const span=document.createElement('span');span.textContent=label;l.append(input,span);fs.append(l)}c.append(fs);const nl=document.createElement('label');nl.textContent=t.note;const ta=document.createElement('textarea');ta.name=`note-${i}`;ta.value=r.validation_note||'';nl.append(ta);c.append(nl);root.append(c)})}
document.getElementById('language').addEventListener('click',()=>{lang=lang==='fa'?'en':'fa';localStorage.setItem('tacticalIlpCoachLanguage',lang);render()});render();
document.getElementById('save').addEventListener('click',async()=>{const t=T[lang],status=document.getElementById('status');for(let i=0;i<doc.rules.length;i++){const selected=document.querySelector(`input[name="status-${i}"]:checked`);if(!selected){status.textContent=t.error.replace('NAME',doc.rules[i].name);return}doc.rules[i].validation_status=selected.value;doc.rules[i].validation_note=document.querySelector(`textarea[name="note-${i}"]`).value.trim()}const output=JSON.parse(JSON.stringify(doc));for(const r of output.rules){delete r.pattern_details;delete r.display_rule;delete r.display_rule_en;delete r.pattern_details_en;delete r.human_readable_en;delete r.pattern_metadata}try{const response=await fetch('/save',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(output)});if(!response.ok)throw new Error(await response.text());status.textContent=t.saved}catch(_e){const blob=new Blob([JSON.stringify(output,null,2)+'\n'],{type:'application/yaml;charset=utf-8'}),a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='phase6_coach_validation.yaml';a.click();URL.revokeObjectURL(a.href);status.textContent=t.downloaded}});
</script></body></html>'''.replace("__PAYLOAD__", payload)


def write_ui(yaml_path: Path, html_path: Path) -> Path:
    if not yaml_path.is_file():
        raise FileNotFoundError(f"{yaml_path} not found; run scripts/run_phase6.py first")
    document = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    document.setdefault("rules", [])
    for rule in document["rules"]:
        rule["human_readable"] = _description(rule)
        rule["human_readable_en"] = _english_description(rule)
        literals = _non_pattern_literals(rule)
        pattern_names = [name for name in _literal_names(rule) if re.fullmatch(r"seq_pattern_\d+", name)]
        english_parts = [ENGLISH_TERMS.get(literal.base, literal.base.replace("_", " ")) for literal in literals]
        persian_parts = [literal.phrase for literal in literals]
        if pattern_names:
            english_parts.insert(0, "the ordered pattern expanded above")
            persian_parts.insert(0, "الگوی ترتیبیِ شرح‌داده‌شده در بالا")
        rule["display_rule_en"] = " and ".join(english_parts) or rule.get("prolog_body", "")
        rule["display_rule"] = " و ".join(persian_parts) or rule.get("prolog_body", "")
        rule["pattern_details"] = _pattern_details(rule)
        rule["validation_status"] = _default_status(rule)
        rule.setdefault("validation_note", "")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_html(document), encoding="utf-8")
    return html_path


def serve(html_path: Path, host: str, port: int, yaml_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            if urlparse(self.path).path != "/":
                self.send_error(404)
                return
            content = html_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_POST(self):  # noqa: N802
            if urlparse(self.path).path != "/save":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if size > 2_000_000:
                    raise ValueError("Form payload is too large")
                doc = json.loads(self.rfile.read(size))
                if not isinstance(doc, dict) or not isinstance(doc.get("rules"), list):
                    raise ValueError("Invalid coach review document")
                for rule in doc["rules"]:
                    if rule.get("validation_status") not in {"approved", "rejected"}:
                        raise ValueError(f"Missing decision for {rule.get('name', 'unnamed rule')}")
                    if rule.get("validation_note") is None:
                        rule["validation_note"] = ""
                yaml_path.parent.mkdir(parents=True, exist_ok=True)
                yaml_path.write_text(yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=110), encoding="utf-8")
                response, code = b"Saved", 200
            except (ValueError, json.JSONDecodeError, OSError) as exc:
                response, code = str(exc).encode("utf-8"), 400
            self.send_response(code)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"Coach review UI: http://{host}:{port}/", flush=True)
    print(f"Decisions will be saved to: {yaml_path}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nCoach review server stopped.")
    finally:
        server.server_close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--yaml", type=Path, default=COACH_VALIDATION_PATH)
    parser.add_argument("--html", type=Path, default=DEFAULT_HTML)
    parser.add_argument("--serve", action="store_true", help="serve UI and save decisions directly to the YAML path")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    path = write_ui(args.yaml, args.html)
    print(f"Coach review UI written to: {path}")
    if args.serve:
        serve(path, args.host, args.port, args.yaml)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

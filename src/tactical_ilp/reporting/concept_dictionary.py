"""
Phase 7 -- the concept dictionary: one entry per body predicate.

WHY THIS FILE EXISTS
--------------------
The final report must regenerate itself whenever a rule is added, changed or
approved -- nobody should hand-edit report prose after a new Popper run.  The
only hand-written text in the whole reporting pipeline therefore lives *here*:
one short, reviewed Persian phrase per Prolog predicate.  Everything else
(rule sentences, statistics, "what changed since last run") is composed from
these phrases plus the numbers in the coach-validation YAML.

If Popper discovers a rule that uses a predicate not listed here, the
translator REFUSES to produce a sentence for it (raises
``UnknownPredicateError``) instead of guessing.  A guessed football sentence
that a coach then reads as fact is worse than a loud failure.

SCOPE -- the most important field
---------------------------------
Every ``seq_*`` predicate is an existential lift: "SOMEWHERE in this
sequence, an action satisfied P".  Two predicates in one rule are therefore
NOT guaranteed to describe the same player or the same moment.  Measured on
the real data for rule 1: only 97 of its 172 matching sequences (56%) have
"dropped between the lines" and "dead-end support" on the *same* action; in
the other 44% they occur at different moments of the sequence.  Every phrase
here is written so that the composed sentence says "in the sequence" and
never "the same player who ...".  The ``scope`` field is checked by a test.

Adding a predicate: add one ``Concept`` below and run the tests.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from tactical_ilp.config import REPORTS_DIR


class UnknownPredicateError(KeyError):
    """A rule uses a predicate with no reviewed dictionary entry."""


@dataclass(frozen=True)
class Concept:
    predicate: str            # exact Prolog predicate name, e.g. "seq_dead_end_support"
    fa_present: str           # phrase when the predicate HOLDS (fills a "..." clause)
    fa_absent: str | None     # phrase when the predicate is NEGATED (not_<pred>), if any
    glossary_ref: str         # section in docs/tactical_concepts_glossary.md
    proxy_note: str = ""      # honest caveat about how the concept is measured
    scope: str = "sequence"   # "sequence" = somewhere in the sequence, never "same action"
    short: str = ""           # short Latin legend label used inside figures


CONCEPTS: dict[str, Concept] = {c.predicate: c for c in [
    # Every phrase below was checked against the ACTUAL definition in bk.pl /
    # config.py, not against the glossary's informal wording -- the two differ
    # in places (see the proxy_note fields).
    Concept(
        "seq_dropped_between_lines",
        fa_present=(
            "حامل توپ در یکی از اکشن‌ها بین دو خط حریف قرار داشته "
            "(خط دفاعی و خط هافبکیِ تخمینی حریف)"
        ),
        fa_absent="حامل توپ در هیچ اکشنی بین دو خط حریف قرار نداشته است",
        glossary_ref="۲.۳",
        proxy_note=(
            "فقط بازیکنِ توپ‌دار قابل بررسی است (بازیکنان بدون توپ در فریم ۳۶۰ شناسه ندارند). "
            "دو خط حریف با رتبه‌بندی مختصات x بازیکنان دیده‌شده تخمین زده می‌شوند، نه با نقش واقعی."
        ),
    ),
    Concept(
        "seq_dead_end_support",
        fa_present=(
            "انسداد زوایای حمایت (نبود خط پاسِ بدون‌مانع به بازیکن خودی)"
        ),
        fa_absent="حامل توپ دست‌کم یک هم‌تیمیِ قابل‌مشاهده با مسیر پاسِ باز داشته است",
        glossary_ref="۴.۳",
        proxy_note=(
            "گزینه‌ها فقط از بازیکنانِ قابل‌مشاهده در فریم ۳۶۰ شمرده می‌شوند؛ "
            "مسیر باز یعنی هیچ حریفی در فاصلهٔ ۲ متری خط مستقیم پاس نباشد. "
            "فریم فاقد بازیکنِ مشخصِ حامل توپ به‌اشتباه «بی‌حمایت» برچسب نمی‌خورد. "
            "این مفهوم مستقل از رو‌به‌جلو بودن اکشن است."
        ),
    ),
    Concept(
        "seq_dwelt_long",
        fa_present="حامل توپ در یکی از اکشن‌ها آن را بیش از ۲.۵ ثانیه نگه داشته است",
        fa_absent="هیچ‌کس توپ را بیش از ۲.۵ ثانیه نگه نداشته است",
        glossary_ref="۶.۴",
        proxy_note="آستانهٔ ۲.۵ ثانیه ثابت است؛ به سطح فشار وابسته نیست (برخلاف مفهوم «تردید»).",
        short="long dwell",
    ),
    Concept(
        "seq_hesitated",
        fa_present="در سکانس، دست‌کم یک اکشن با مکث غیرعادیِ متناسب با سطح فشار ثبت شده است",
        fa_absent="در هیچ اکشنی از سکانس مکثِ غیرعادیِ متناسب با سطح فشار ثبت نشده است",
        glossary_ref="۶.۱",
        proxy_note=(
            "مکث با آستانهٔ میانگین به‌علاوهٔ یک انحراف معیارِ همان سطح فشار سنجیده می‌شود؛ "
            "این رخداد به‌تنهایی پیامد یا علت از دست‌رفتن توپ را نشان نمی‌دهد."
        ),
        short="hesitation",
    ),
    Concept(
        "seq_diagonal_pass",
        fa_present="حداقل یک پاس قطری داده شده است",
        fa_absent="هیچ پاس قطری داده نشده است",
        glossary_ref="۱.۲",
        proxy_note="قطری = جابه‌جایی حداقل ۸ متر روی هر دو محور طولی و عرضی.",
        short="diagonal pass",
    ),
    Concept(
        "fast_sequence",
        fa_present="ریتم سکانس در دهک سریع‌ترین سکانس‌ها قرار داشته است",
        fa_absent=None,
        glossary_ref="۶.۲",
        proxy_note="سریع = نرخ اکشن در صدک ۷۰ یا بالاتر (دهک‌های ۸ تا ۱۰)؛ آستانه از توزیع سکانس‌های معتبر همین داده محاسبه می‌شود.",
        short="fast tempo",
    ),
    Concept(
        "slow_sequence",
        fa_present="نرخ اکشن سکانس در بازهٔ کم‌تمپو قرار داشته است",
        fa_absent=None,
        glossary_ref="۶.۲",
        proxy_note="کندی با نرخ تعداد اکشن به مدت سکانس سنجیده می‌شود؛ این شاخص، سرعت حرکت توپ را مستقیماً اندازه نمی‌گیرد.",
        short="slow tempo",
    ),
    Concept(
        "seq_triangulation_opponent_trapped",
        fa_present="در یکی از اکشن‌های سکانس، مثلث پاس با محاصرهٔ مدافع حریف تشکیل شده است",
        fa_absent=None,
        glossary_ref="۳.۶ب",
        proxy_note="محاصره از موقعیت مدافعان در فریم ۳۶۰ و هندسهٔ مثلث تخمین زده می‌شود.",
        short="trapped opponent triangle",
    ),
    Concept(
        "seq_counter_attack_potential",
        fa_present=(
            "در زمین خودی، بارسا نسبت به حریفان نزدیک توپ برتری عددی داشته است "
            "(پتانسیل ضدحمله)"
        ),
        fa_absent=None,
        glossary_ref="۹.۱ب",
        proxy_note=(
            "فریم ۳۶۰ فقط بخشی از زمین را می‌بیند، پس شمارش حریفان ممکن است ناقص باشد. "
            "در این داده تقریباً معادل «برتری عددی در یک‌سوم دفاعی» است."
        ),
    ),
    Concept(
        "seq_first_time_play",
        fa_present="حداقل یک بار اکشن با نگه‌داری توپِ حداکثر ۰٫۵ ثانیه ثبت شده است",
        fa_absent=None,
        glossary_ref="۶.۳",
        proxy_note="StatsBomb تعداد لمس را ثبت نمی‌کند؛ یک‌لمسه بودن با نگه‌داری حداکثر ۰٫۵ ثانیه تقریب زده می‌شود.",
        short="one-touch",
    ),
    Concept(
        "opponent_visible_majority",
        fa_present="در دست‌کم یک فریم معتبر، بازیکنان قابل‌مشاهدهٔ حریف از بازیکنان قابل‌مشاهدهٔ خودی بیشتر بوده‌اند",
        fa_absent="در هیچ فریم معتبرِ سکانس، حریفانِ قابل‌مشاهده اکثریت نداشته‌اند",
        glossary_ref="۴.۶",
        proxy_note="فریم باید دست‌کم ۷ بازیکنِ قابل‌مشاهده در مجموع و بازیکنِ صاحب توپ داشته باشد. شمارش ۳۶۰ ناقص و وابسته به زاویهٔ ثبت است؛ تعداد فریم‌های معتبر و دارای اکثریت نیز در فکت‌ها ذخیره می‌شود.",
        short="visible opponent majority",
    ),
    Concept(
        "keeper_played_open_play",
        fa_present="در سکانس دست‌کم یک پاس موفق کوتاه یا متوسط توسط دروازه‌بان اجرا شده است",
        fa_absent="در سکانس پاس موفق کوتاه یا متوسطی توسط دروازه‌بان ثبت نشده است",
        glossary_ref="۱۲.۴",
        proxy_note="این ویژگی فقط پاس نوع pass را در صورت موفقیت و طبقه‌بندی طول short یا medium می‌شمارد. دریافت توپ توسط دروازه‌بان به‌تنهایی و گل‌کیک در آن دخیل نیست؛ این proxy دربارهٔ قصد تاکتیکی یا کیفیت تصمیم داوری نمی‌کند.",
        short="short/medium keeper pass",
    ),
    Concept(
        "seq_costly_hesitation",
        fa_present=(
            "در یک اکشن، مکث نسبت به سطح فشار غیرعادی بوده و اکشن بعدیِ همان سکانس ناموفق ثبت شده است"
        ),
        fa_absent=None,
        glossary_ref="۶.۱ب",
        proxy_note=(
            "تردید با آستانهٔ متناسب با فشار تخمین زده می‌شود؛ اکشن بعدی ناموفق است. "
            "چون ناموفق‌بودن اکشن پایانی در بخشی از تعریف برچسب هدف دخالت دارد، این ویژگی از واژگان "
            "جست‌وجوی کنترل‌شده از نظر نشت کنار گذاشته شده است؛ فقط به‌عنوان هم‌توالیِ پس‌نگرانه "
            "تفسیر شود و رابطهٔ علّی را ثابت نمی‌کند."
        ),
        short="costly hesitation",
    ),
    # --- additional concepts used by reviewed candidate rules --------------
    Concept(
        "seq_persistent_half_space_occupier",
        fa_present="یک بازیکن دست‌کم دو بار از نیم‌فضا توپ لمس کرده است",
        fa_absent=None,
        glossary_ref="۵.۱",
        short="half-space occupier",
    ),
    Concept(
        "seq_line_pass",
        fa_present="یک پاس موازی خط کناری داده شده است",
        fa_absent=None,
        glossary_ref="۱.۳",
        short="line pass",
    ),
    Concept(
        "seq_connecting_pass",
        fa_present=(
            "در سکانس، دست‌کم یک پاس موفقِ کوتاه و غیرپیش‌رونده برای پیوند دادن بازی ثبت شده است"
        ),
        fa_absent=None,
        glossary_ref="۱.۶",
        proxy_note=(
            "طبق bk.pl، پاس باید موفق و کوتاه باشد و شرط progressive برقرار نباشد. "
            "این نام‌گذاریِ برنامه است و لزوماً قصد ذهنیِ حفظ مالکیت را ثابت نمی‌کند."
        ),
        short="connecting pass",
    ),
    Concept(
        "seq_line_breaking_pass",
        fa_present="حداقل یک پاس موفق از خط دفاعی تخمینی حریف عبور کرده است",
        fa_absent="هیچ پاس خط‌شکن موفقی در سکانس ثبت نشده است",
        glossary_ref="۱.۱",
        proxy_note=(
            "خط دفاعی از جایگاه بازیکنان قابل‌مشاهده تخمین زده می‌شود. نبود این پاس به‌تنهایی "
            "به معنی عقیم‌بودن حمله نیست؛ پیشروی ممکن است با حمل، دریبل یا الگوی پاس دیگری رخ دهد."
        ),
        short="line-breaking pass",
    ),
    Concept(
        "seq_pass_in_behind",
        fa_present="دست‌کم یک پاس موفق پشت خط دفاعی تخمینی حریف فرود آمده و به یک‌سوم هجومی یا محوطهٔ جریمه رسیده است",
        fa_absent=None,
        glossary_ref="۱.۵",
        proxy_note="مقصد باید هم پشت خط دفاعی تخمینی و هم در یک‌سوم هجومی یا محوطهٔ جریمه باشد؛ برخلاف line_breaking_pass لازم نیست پاس از پشت خط شروع شود.",
        short="pass in behind",
    ),
    Concept(
        "seq_supported_action",
        fa_present="در دست‌کم یک اکشن، هم‌تیمیِ قابل‌مشاهده‌ای با مسیر پاسِ باز وجود داشته است",
        fa_absent=None,
        glossary_ref="۴.۱",
        proxy_note="مسیر باز با فاصلهٔ حریفان از خط مستقیم حامل توپ تا هم‌تیمی سنجیده می‌شود؛ فریم‌های فاقد حامل مشخص نادیده گرفته می‌شوند.",
        short="open support",
    ),
    Concept(
        "seq_no_open_support",
        fa_present="در هیچ اکشنِ دارای فریم معتبر، هم‌تیمیِ قابل‌مشاهده‌ای با مسیر پاسِ باز وجود نداشته است",
        fa_absent=None,
        glossary_ref="۴.۳",
        proxy_note="این alias برای فکت‌های no_open_support است؛ فقدان فریم معتبر را به‌عنوان نبود حمایت تفسیر نمی‌کند.",
        short="no open support",
    ),
    Concept(
        "seq_rondo_cage",
        fa_present="در یک اکشن، دست‌کم دو مثلث هم‌زمان مدافعی از حریف را محاصره کرده‌اند",
        fa_absent=None,
        glossary_ref="۳.۶ب",
        proxy_note="محاصره از آرایش مثلثی و بازیکنان قابل‌مشاهده در فریم ۳۶۰ تخمین زده می‌شود؛ عمداً زیرمجموعهٔ مفهوم trapping است.",
        short="rondo cage",
    ),
    Concept(
        "seq_press_resistance",
        fa_present="در دست‌کم یک اکشن، مالکیت زیر فشار سنگین حفظ شده است",
        fa_absent=None,
        glossary_ref="۸.۴",
        proxy_note="فشار سنگین بر مبنای دادهٔ فریم/پرچم فشار و موفقیت اکشن تعریف می‌شود؛ این نام، مقاومت تحت فشار را به موفقیت همان اکشن محدود می‌کند.",
        short="press resistance",
    ),
    Concept(
        "seq_counter_pressed",
        fa_present="پرچم counterpress در دادهٔ StatsBomb برای دست‌کم یک اکشن ثبت شده است",
        fa_absent=None,
        glossary_ref="۸.۳",
        proxy_note="پرچم منبع مستقیماً مصرف می‌شود؛ تعریف مستقل از فاصلهٔ زمانی یا موفقیت ضدپرس نیست.",
        short="counter-press action",
    ),
    Concept(
        "seq_open_support",
        fa_present="در سکانس دست‌کم یک اکشن با یارِ دارای خط پاس باز وجود داشته است",
        fa_absent="در هیچ اکشنی از سکانس یارِ دارای خط پاس باز دیده نشده است",
        glossary_ref="۴.۱",
        proxy_note=(
            "این نام مستعار همان seq_supported_action است: گزینهٔ حمایتی با فاصلهٔ نقطه تا مسیر پاس "
            "و بررسی حریفان قابل‌مشاهده سنجیده می‌شود؛ نبود آن در کل سکانس از نبود حمایت در یک لحظه خاص گسترده‌تر است."
        ),
        short="open support",
    ),
    Concept(
        "seq_rest_defence_protected",
        fa_present="در دست‌کم یک اکشن، دو هم‌تیمیِ قابل‌مشاهده پشت توپ بوده‌اند",
        fa_absent="در هیچ اکشنی شرط پروکسیِ پوشش دفاعیِ پشت توپ برقرار نبوده است",
        glossary_ref="۴.۴",
        proxy_note=(
            "پروکسیِ فعلی در هر فریم تعداد هم‌تیمی‌های قابل‌مشاهده با مختصات طولی دست‌کم ۲ متر عقب‌تر از حامل توپ را می‌شمارد؛ "
            "پوشش کامل تیم یا نقش واقعی بازیکنان را تضمین نمی‌کند."
        ),
        short="rest defence protected",
    ),
    Concept(
        "seq_rest_defence_exposed",
        fa_present="آسیب‌پذیری ساختار دفاعی پشت توپ (در این اکشن کمتر از دو هم‌تیمیِ غیرحارس دست‌کم ۵ متر عقب‌تر از توپ بوده‌اند)",
        fa_absent="در اکشن‌های بررسی‌شده، پروکسیِ آسیب‌پذیری ساختار دفاعی پشت توپ ثبت نشده است",
        glossary_ref="۴.۴",
        proxy_note=(
            "این ویژگی دروازه‌بان را کنار می‌گذارد و هم‌تیمی‌های قابل‌مشاهده را با فاصلهٔ طولی حداقل ۵ متر عقب‌تر از حامل توپ می‌شمارد؛ "
            "پوشش کامل تیم یا نقش واقعی بازیکنان را تضمین نمی‌کند."
        ),
        short="rest defence exposed",
    ),
    Concept(
        "seq_keeper_build_up_start",
        fa_present="بازیسازی با دروازه‌بان (گل‌کیک یا گرفتن توپ) شروع شده است",
        fa_absent=None,
        glossary_ref="۱۲.۱",
        short="keeper start",
    ),
    Concept(
        "seq_many_passing_options",
        fa_present=(
            "در دست‌کم یک اکشن، برای حامل توپ پنج یا بیش از پنج هم‌تیمیِ قابل‌مشاهده "
            "با مسیر پاسِ باز وجود داشته است"
        ),
        fa_absent="در هیچ اکشنی از سکانس پنج گزینهٔ پاسِ باز یا بیشتر ثبت نشده است",
        glossary_ref="۴.۵",
        proxy_note=(
            "فقط بازیکنانِ دیده‌شده در فریم ۳۶۰ شمرده می‌شوند؛ پنج گزینه، آستانهٔ bucket فعلی کد است. "
            "این ویژگی به معنی انتخاب یا استفاده از آن گزینه‌ها نیست."
        ),
        short="many passing options",
    ),
    Concept(
        "seq_switch_of_play",
        fa_present=(
            "در سکانس، دست‌کم یک پاس موفق با جابه‌جایی عرضیِ حداقل ۲۵ متر "
            "و تغییر کانال عرضی زمین ثبت شده است"
        ),
        fa_absent="هیچ پاس موفقی با جابه‌جایی عرضیِ حداقل ۲۵ متر و تغییر کانال ثبت نشده است",
        glossary_ref="۱.۴",
        proxy_note=(
            "پروکسی هندسی کد برای تغییر جناح است؛ لزوماً قصد عبور از تراکم حریف را ثابت نمی‌کند. "
            "وجود آن در هر نقطهٔ سکانس به‌صورت ویژگی وجودی ثبت می‌شود."
        ),
        short="switch of play",
    ),
    Concept(
        "seq_switch_to_right",
        "در سکانس پاس موفقی با تغییر عرضی حداقل ۲۵ متر به نیم‌فضای راست یا بال راست رسیده است",
        None,
        "۱.۴",
        "مقصد با lane انتهایی در چارچوب حملهٔ بارسلونا تعیین می‌شود؛ قصد تاکتیکی از دادهٔ هندسی به‌تنهایی اثبات نمی‌شود.",
        short="switch to right",
    ),
    Concept(
        "seq_switch_to_left",
        "در سکانس پاس موفقی با تغییر عرضی حداقل ۲۵ متر به نیم‌فضای چپ یا بال چپ رسیده است",
        None,
        "۱.۴",
        "مقصد با lane انتهایی در چارچوب حملهٔ بارسلونا تعیین می‌شود؛ قصد تاکتیکی از دادهٔ هندسی به‌تنهایی اثبات نمی‌شود.",
        short="switch to left",
    ),
    Concept(
        "seq_aerial_pass",
        "دست‌کم یک پاس با برچسب منبع StatsBomb برابر High Pass ثبت شده است",
        None,
        "۱.۸",
        "برچسب ارتفاع از extra رویداد اصلی StatsBomb حفظ می‌شود؛ اکشن مصنوعی فاقد شناسهٔ رویداد چنین برچسبی ندارد.",
        short="aerial pass",
    ),
    Concept(
        "seq_long_pass",
        "دست‌کم یک پاس با طول اقلیدسی ۳۰ متر یا بیشتر ثبت شده است",
        None,
        "۱.۹",
        "فاصله از مختصات ابتدا و انتهای SPADL روی هر دو محور محاسبه می‌شود؛ این ویژگی مستقل از ارتفاع پاس است.",
        short="long pass",
    ),
    Concept(
        "seq_under_pressure",
        "دست‌کم یک اکشن زیر فشار حریف انجام شده است",
        None,
        "۸.۴",
        "فشار از پرچم StatsBomb و تعداد حریفانِ دیده‌شده در شعاع‌های ۳ و ۵ متر تعیین می‌شود.",
        short="under pressure",
    ),
    Concept(
        "seq_flank_dribble",
        "دست‌کم یک دریبل از یکی از دو کانال کناری آغاز شده است",
        None,
        "۵.۴",
        "کانال کناری با laneهای left_wing و right_wing در مختصات SPADL تشخیص داده می‌شود.",
        short="flank dribble",
    ),
    Concept(
        "seq_flank_line_carry",
        "دست‌کم یک دریبل در همان کانال کناری دست‌کم ۵ متر رو به دروازه پیشروی کرده است",
        None,
        "۵.۴",
        "برابری کانال ابتدا و انتها و جابه‌جایی طولی مثبتِ حداقل ۵ متر بررسی می‌شود؛ عرضی‌بودن مسیر سنجیده نمی‌شود.",
        short="flank line carry",
    ),
    Concept(
        "seq_flank_cut_inside",
        "دست‌کم یک دریبل از بال کناری به نیم‌فضا یا کانال مرکزی وارد شده است",
        None,
        "۵.۴",
        "ویژگی از lane ابتدا و انتهای اکشن دریبل ساخته می‌شود و مسیر دقیق حرکت بین دو مختصات را مدل نمی‌کند.",
        short="flank cut inside",
    ),
    Concept(
        "seq_central_long_carry",
        "حمل توپ از یکی از سه lane غیرکناری با طول مسیر حداقل ۳٫۵ متر ثبت شده است",
        None, "۵.۴", "طول مسیر اقلیدسی اکشن SPADL سنجیده می‌شود و جهت حرکت شرط نیست.",
        short="central carry",
    ),
    Concept(
        "seq_central_take_on",
        "اقدام یک‌به‌یک برای عبور از حریف در یکی از سه lane غیرکناری ثبت شده است",
        None, "۵.۴", "برچسب take_on و lane مبدأ استفاده می‌شود؛ موفقیت دوئل شرط تعریف نیست.",
        short="central take-on",
    ),
    Concept(
        "seq_central_backward_pass",
        "پاسی از یکی از سه lane غیرکناری با طول حداقل ۷ متر و جابه‌جایی رو به عقب حداقل ۵ متر ثبت شده است",
        None, "۵.۴", "طول اقلیدسی حداقل ۷ متر و اختلاف طولی x حداکثر منفی ۵ متر لازم است.",
        short="central backward pass",
    ),
    Concept(
        "seq_flank_backward_carry",
        "حمل توپ در همان lane کناری دست‌کم ۴ متر رو به عقب انجام شده است",
        None, "۵.۴", "آستانهٔ اجرایی از شرط صریح Δx ≤ −۴ متر پیروی می‌کند؛ شرح درخواست ۳ متر را نیز ذکر می‌کرد.",
        short="backward flank carry",
    ),
    Concept(
        "seq_flank_to_center_progression",
        "پاسی از بال کناری به نیم‌فضا یا کانال مرکزی با طول حداقل ۱۰ متر و پیشروی طولی حداقل ۵ متر ثبت شده است",
        None, "۵.۴", "ویژگی کنشی چپ و راست را ادغام می‌کند؛ lane مبدأ در نمادهای ماینر همچنان سمت را نشان می‌دهد.",
        short="flank to center",
    ),
    Concept(
        "seq_flank_to_center",
        "در سکانس پاسی از بال کناری به نیم‌فضا یا کانال مرکزی با پیشروی معنادار رخ داده است",
        None, "۵.۴", "نام عمومی Popper است و به همان ویژگی کنشی flank_to_center_progression نگاشت می‌شود.",
        short="flank to center",
    ),
    Concept(
        "seq_pass_exchange",
        "در سکانس دست‌کم یک تبادل رفت‌وبرگشت پاس میان دو بازیکن متفاوت رخ داده است",
        None, "۵.۵", "گیرنده از فرادادهٔ StatsBomb خوانده می‌شود؛ فقط گزارهٔ بولی سکانس ذخیره می‌شود و شناسه‌ها وارد فکت‌های Popper نمی‌شوند.",
        short="pass exchange",
    ),
    Concept(
        "seq_central_switch",
        "دو پاس موفقِ پیاپی از یک جناح به کانال مرکزی و سپس به جناح مقابل ثبت شده است",
        None,
        "۵.۵",
        "پروکسی مکانی بر اساس دو ردیف پیاپی SPADL است؛ شناسهٔ عمومی گیرنده در دادهٔ فعال وجود ندارد، پس تداوم همان هافبک تأیید نمی‌شود.",
        short="central switch",
    ),
    Concept(
        "seq_recycle_to_same_flank",
        "دو پاس موفقِ پیاپی از یک جناح به مرکز و از مرکز به همان جناح ثبت شده است",
        None,
        "۵.۵",
        "پروکسی مکانیِ دو پاس پیاپی است؛ گیرنده و هویت بازیکنِ ادامه‌دهنده قابل تأیید نیست.",
        short="same-flank recycle",
    ),
    Concept(
        "seq_center_cycling",
        "دو پاس موفقِ پیاپی در کانال مرکزی آغاز و در همان کانال تمام شده‌اند",
        None,
        "۵.۵",
        "فقط دو ردیف پیاپی SPADL بررسی می‌شوند؛ الگوهای طولانی‌تر یا دریبل‌های میانجی در این تعریف نیستند.",
        short="center cycling",
    ),
]}


# Latin legend labels for figures.  Kept apart from the Persian phrases on purpose:
# they are a display concern, and a separate table cannot corrupt a phrase.
SHORT_LABELS: dict[str, str] = {
    "seq_dropped_between_lines": "between lines",
    "seq_dead_end_support": "dead-end support",
    "seq_dwelt_long": "long dwell",
    "seq_diagonal_pass": "diagonal pass",
    "fast_sequence": "fast tempo",
    "seq_counter_attack_potential": "numerical superiority",
    "seq_first_time_play": "one-touch",
    "seq_persistent_half_space_occupier": "half-space occupier",
    "seq_line_pass": "line pass",
    "seq_keeper_build_up_start": "keeper start",
    "seq_costly_hesitation": "costly hesitation",
    "seq_hesitated": "hesitation",
    "seq_connecting_pass": "connecting pass",
    "opponent_visible_majority": "visible opponent majority",
    "keeper_played_open_play": "short/medium keeper pass",
    "seq_switch_of_play": "switch of play",
    "seq_switch_to_right": "switch to right",
    "seq_switch_to_left": "switch to left",
    "seq_aerial_pass": "aerial pass",
    "seq_long_pass": "long pass",
    "seq_line_breaking_pass": "line-breaking pass",
    "seq_open_support": "open support",
    "seq_rest_defence_protected": "protected rest defence",
    "seq_many_passing_options": "many passing options",
    "seq_under_pressure": "under pressure",
    "seq_flank_dribble": "flank dribble",
    "seq_flank_line_carry": "flank line carry",
    "seq_flank_cut_inside": "flank cut inside",
    "seq_central_long_carry": "central long carry",
    "seq_central_take_on": "central take-on",
    "seq_central_backward_pass": "central backward pass",
    "seq_flank_backward_carry": "backward flank carry",
    "seq_flank_to_center_progression": "flank-to-centre progression",
    "seq_flank_to_center": "flank-to-centre progression",
    "seq_pass_exchange": "bidirectional pass exchange",
    "seq_central_switch": "central switch",
    "seq_recycle_to_same_flank": "same-flank recycle",
    "seq_center_cycling": "center cycling",
}


def short_label(predicate: str) -> str:
    """Latin legend text for a predicate; falls back to the raw name (never fails)."""
    pattern = re.fullmatch(r"seq_pattern_(\d+)", predicate)
    if pattern:
        return f"sequence pattern {pattern.group(1)}"
    return SHORT_LABELS.get(predicate, predicate)


def _pattern_concept(predicate: str) -> Concept | None:
    """Build reviewed wording from the active run's saved pattern metadata."""
    match = re.fullmatch(r"seq_pattern_\d+", predicate)
    if not match:
        return None
    metadata_path = REPORTS_DIR / "phase5_sequential_training.json"
    if not metadata_path.exists():
        return None
    try:
        report = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = report.get("pattern_facts", {}).get("features", [])
        feature = next((row for row in features if row.get("predicate") == predicate), None)
    except (OSError, json.JSONDecodeError, AttributeError):
        return None
    if feature is None:
        return None

    lanes = {
        "left_wing": "بال چپ", "left_half_space": "نیم‌فضای چپ",
        "central_channel": "کانال مرکزی", "right_half_space": "نیم‌فضای راست",
        "right_wing": "بال راست", "unknown": "کانال نامشخص",
        "left_channel": "کانال چپ", "right_channel": "کانال راست",
    }
    thirds = {
        "defensive_third": "یک‌سوم دفاعی", "middle_third": "یک‌سوم میانی",
        "final_third": "یک‌سوم هجومی", "unknown": "یک‌سوم نامشخص",
    }
    predicate_phrases = {
        "line_breaking_pass": "پاس خط‌شکن",
        "diagonal_pass": "پاس قطری",
        "switch_to_right": "تعویض جناح به راست",
        "switch_to_left": "تعویض جناح به چپ",
        "aerial_pass": "پاس هواییِ برچسب‌خورده به‌عنوان High Pass",
        "long_pass": "پاس بلند",
        "pass_in_behind": "پاس پشت خط دفاع",
        "line_pass": "پاس در همان کانال طولی",
        "connecting_pass": "پاس ارتباطی",
        "dropped_between_lines": "دریافت/حمل توپ بین خطوط",
        "first_time_play": "بازی یک‌ضرب",
        "dwelt_long": "نگه‌داشتن توپ بیش از ۲٫۵ ثانیه",
        "supported_action": "اکشن دارای حمایت هم‌تیمی",
        "dead_end_support": "انسداد زوایای حمایت (نبود خط پاس بدون‌مانع به بازیکن خودی)",
        "triangulation": "ساختار مثلثی",
        "triangulation_opponent_trapped": "تشکیل مثلث پاس با محاصرهٔ مدافع حریف",
        "rondo_cage": "محاصرهٔ بازیکن در ساختار روندو",
        "press_resistance": "حفظ توپ زیر فشار",
        "counter_pressed": "اکشن پس از ضدپرس",
        "rest_defence_exposed": "آسیب‌پذیری ساختار دفاعی پشت توپ (کمتر از دو هم‌تیمیِ غیرحارس دست‌کم ۵ متر عقب‌تر از توپ)",
        "under_pressure": "اکشن زیر فشار",
        "flank_line_carry": "پیشروی طولی در کناره",
        "flank_cut_inside": "حرکت از کناره به داخل",
        "flank_dribble": "دریبل در کناره",
        "central_long_carry": "حمل توپ بلند در کانال مرکزی",
        "central_take_on": "اقدام به دریبل یک‌به‌یک در کانال مرکزی",
        "central_backward_pass": "پاس بلند رو به عقب از کانال مرکزی",
        "flank_backward_carry": "حمل توپ رو به عقب در کناره",
        "flank_to_center_progression": "پاس پیش‌رونده از کناره به مرکز",
        "central_switch": "تعویض جناح از مسیر مرکزی",
        "recycle_to_same_flank": "بازگرداندن توپ به همان جناح از مرکز",
        "center_cycling": "گردش متوالی توپ در مرکز",
    }
    fa_digits = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    steps = []
    for row in feature.get("pattern_chain", []):
        item_phrases = []
        for event in row.get("items", []):
            event_predicate = event.get("predicate")
            reviewed = CONCEPTS.get(f"seq_{event_predicate}")
            base = predicate_phrases.get(
                event_predicate,
                reviewed.fa_present if reviewed else event_predicate or "رخداد تاکتیکی",
            )
            action_types = {
                "pass": "پاس", "cross": "ارسال", "carry": "حمل توپ",
                "dribble": "دریبل", "shot": "شوت", "unknown": "اکشن نامشخص",
            }
            action_type = action_types.get(event.get("action_type"), "اکشن")
            lane = lanes.get(event.get("lane"), "کانال نامشخص")
            third = thirds.get(event.get("third"), "یک‌سوم نامشخص")
            item_phrases.append(f"{base} ({action_type}) در {lane} و {third}")
        if item_phrases:
            step_number = str(row.get("step")).translate(fa_digits)
            steps.append(f"گام {step_number}: " + " و ".join(item_phrases))
    if not steps:
        return None
    gap = max(
        (int(item.get("max_intervening_actions", 0))
         for item in feature.get("step_distance_constraints", [])),
        default=0,
    )
    text = "؛ سپس ".join(steps)
    phrase = f"توالی تاکتیکی: {text}"
    if len(steps) > 1:
        phrase += f"؛ فاصلهٔ هر گام می‌تواند حداکثر {str(gap).translate(fa_digits)} اکشنِ میانجی داشته باشد"
    return Concept(
        predicate=predicate,
        fa_present=phrase,
        fa_absent=f"الگوی ترتیبی «{text}» در سکانس ثبت نشده است",
        glossary_ref="فاز ۵، metadata الگوهای ترتیبی",
        proxy_note=(
            f"این الگو در دادهٔ آموزش با حمایت هدف حداقل ۵ و growth هموارشده انتخاب شده است؛ "
            f"growth={feature.get('growth_rate', 0):.2f}x، "
            f"حمایت هدف {feature.get('training_target_support', 0)} و حمایت کنترل "
            f"{feature.get('training_control_support', 0)}. هم‌رخدادی/ترتیب، رابطهٔ علّی را اثبات نمی‌کند."
        ),
        short=f"sequence pattern {predicate.rsplit('_', 1)[-1]}",
    )


def lookup(predicate: str) -> Concept:
    """Return the reviewed entry for a predicate, or fail loudly."""
    try:
        return CONCEPTS[predicate]
    except KeyError:
        dynamic = _pattern_concept(predicate)
        if dynamic is not None:
            return dynamic
        raise UnknownPredicateError(
            f"predicate {predicate!r} has no entry in concept_dictionary.CONCEPTS -- "
            f"add a reviewed Persian phrase before this rule can appear in a report"
        ) from None


def split_negation(literal: str) -> tuple[str, bool]:
    """``not_seq_dwelt_long`` -> (``seq_dwelt_long``, True); ``fast_sequence`` -> (.., False).

    Complement predicates come from phase5_vocabulary.COMPLEMENT_PREDS.
    """
    if literal.startswith("not_"):
        return literal[len("not_"):], True
    return literal, False

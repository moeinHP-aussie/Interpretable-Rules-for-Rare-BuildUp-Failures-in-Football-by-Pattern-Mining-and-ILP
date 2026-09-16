# Neuro-Symbolic Tactical Analysis — FC Barcelona Build-up (MVP)

کشف الگوهای رفتاری تکرارشونده و ناخودآگاه در فاز بازیسازی (Build-up) با
ترکیب یک لایه استخراج ویژگی قاعده‌محور و یک موتور **Inductive Logic
Programming** (Popper).

* **دیتاست:** StatsBomb Open Data — La Liga 2020/21، FC Barcelona
  (`competition_id=11`, `season_id=90`)، ۳۵ بازی، همگی با داده ۳۶۰
* **زنجیره:** StatsBomb → SPADL → سکانس Build-up → Prolog Facts → Popper → گزارش مربی
* **وضعیت:** فازهای ۰–۳ کامل. فاز ۴ (`bk.pl`) در انتظار ورودی تاکتیکی.

> داده از StatsBomb Open Data است. هنگام انتشار هر نتیجه‌ای، منبع را ذکر کنید.

---

## اجرای سریع

```bash
uv sync --extra dev
uv run python scripts/run_phase0_check.py     # چک محیط
uv run python scripts/run_phase1.py           # دانلود + SPADL + 360  (اجرای اول: ۱۰-۲۵ دقیقه)
uv run python scripts/run_phase2.py           # سکانس + برچسب + نمونه‌گیری
uv run python scripts/run_phase3.py           # تولید Prolog facts
uv run pytest -m "not network"                # ۵۸ تست
```

**ویندوز:** راهنمای کامل گام‌به‌گام در [`docs/SETUP_WINDOWS.md`](docs/SETUP_WINDOWS.md).

---

## پیش‌نیازها

| ابزار | نسخه | چرا |
|---|---|---|
| Python | ۳.۱۱ یا ۳.۱۲ | `socceraction` روی ۳.۱۳+ کار نمی‌کند |
| uv | هر نسخه اخیر | مدیریت پکیج و محیط |
| SWI-Prolog | **≥ ۹.۱.۱۲** | `janus_swi` (وابسته Popper) نسخه پایین‌تر را رد می‌کند |
| Popper | 5.x | موتور ILP — در محیط پایتون ۳.۱۴ **مجزای خودش** اجرا می‌شود |

---

## خروجی اصلی

```
data/prolog/facts_all.pl          86,496 ground facts  (35 matches, 1066 sequences)
data/prolog/facts_manifest.json   شمارش هر predicate
data/processed/ilp_examples.parquet   191 positive + 382 negative examples
```

نمونه واقعی از factها (سکانس `s_3773377_0002`، برچسب منفی — عیناً از `facts_all.pl`):

```prolog
player_identity(p_5203).
player_name(p_5203,sergio_busquets).
has_role(p_5203,defensive_midfielder).
plays_for(p_5203,barcelona).

sequence(s_3773377_0002).
sequence_label(s_3773377_0002,negative).
sequence_length(s_3773377_0002,5).
sequence_start_zone(s_3773377_0002,defensive_third_left_half_space).

involved_in(p_30486,s_3773377_0002).
involved_in(p_43728,s_3773377_0002).
involved_in(p_5211,s_3773377_0002).
involved_in(p_5203,s_3773377_0002).

action(a_3773377_00326).
in_sequence(a_3773377_00326,s_3773377_0002).
action_order(a_3773377_00326,4).
actor(a_3773377_00326,p_5203).
action_type(a_3773377_00326,dribble).
action_result(a_3773377_00326,success).
at_third(a_3773377_00326,defensive_third).
at_lane(a_3773377_00326,left_half_space).
under_pressure(a_3773377_00326).
pressure_level(a_3773377_00326,light_pressure).
opponents_within(a_3773377_00326,1).
pass_direction(a_3773377_00326,backward).
pass_length(a_3773377_00326,short).

passes_to(p_30486,p_43728,s_3773377_0002).
passes_to(p_43728,p_5211,s_3773377_0002).
passes_to(p_5211,p_5203,s_3773377_0002).
```

خوانش انسانی: پدری → مینگزا → آلبا → بوسکتس، همه در نیمه چپ یک‌سوم دفاعی؛
بوسکتس زیر فشار سبک توپ را به عقب می‌برد و سکانس به پیامد منفی ختم می‌شود.

**مهم:** هر predicate در `facts_all.pl` یک **Fact ثابت از پیش محاسبه‌شده** است،
نه یک rule زنده Prolog. هندسه در پایتون محاسبه و اینجا فریز می‌شود.

---

## چارچوب مرجع

کل پروژه در یک چارچوب واحد کار می‌کند:

```
x ∈ [0, 105]   x=0 دروازه بارسا،  x=105 دروازه حریف   (بارسا همیشه به +x حمله می‌کند)
y ∈ [0, 68]    y=0 خط طولی سمت راست از دید جهت حمله بارسا
```

اکشن‌های حریف هم به همین چارچوب آینه می‌شوند.

---

## مستندات

| سند | محتوا |
|---|---|
| [`report.md`](report.md) | گزارش کامل کارهای انجام‌شده، تصمیم‌ها، مشکلات و انحراف‌ها |
| [`docs/SETUP_WINDOWS.md`](docs/SETUP_WINDOWS.md) | نصب و تست روی ویندوز |
| [`docs/negative_outcome_definition.md`](docs/negative_outcome_definition.md) | تعریف عملیاتی «پیامد منفی» (تسک ۲.۲) |
| [`docs/negative_sampling_strategy.md`](docs/negative_sampling_strategy.md) | استراتژی نمونه‌گیری منفی (تسک ۲.۴) |

# گزارش کار پروژه آنالیز تاکتیکی نورو-سمبلیک

**پروژه:** کشف الگوهای ناخودآگاه در فاز Build-up بارسلونا (La Liga 2020/21)
**وضعیت فعلی:** فازهای ۰ تا ۳ **تکمیل شد** — فاز ۴ در انتظار تأیید شما
**آخرین به‌روزرسانی:** دور ۱ (فازهای ۰–۳)
**نتیجه تست‌ها:** ۵۸ تست سبز، ۰ قرمز (`uv run pytest -m "not network"`)

---

## ۰. خلاصه یک‌نگاهی

| فاز | تسک‌ها | وضعیت | خروجی کلیدی |
|---|---|---|---|
| ۰ | ۰.۱–۰.۴ | ✅ کامل | محیط uv + SWI-Prolog 9.2.9 + Popper کارکرد‌تأییدشده |
| ۱ | ۱.۱–۱.۴ | ✅ کامل | ۸۲٬۰۴۸ اکشن SPADL، پوشش ۳۶۰ = ۹۲.۶۵٪، ۲۵ بازیکن مرجع |
| ۲ | ۲.۱–۲.۴ | ✅ کامل | ۱۰۶۶ سکانس برچسب‌خورده، ۱۹۱ pos / ۳۸۲ neg |
| ۳ | ۳.۱–۳.۳ | ✅ کامل | `facts_all.pl` با ۸۶٬۴۹۶ fact، لود تمیز در SWI-Prolog |
| ۴ | — | ⏸ متوقف | نیازمند تأیید و مشارکت تاکتیکی شما |

---

## ۱. فاز ۰ — راه‌اندازی محیط و زیرساخت

### تسک ۰.۱ — پروژه Python و مدیریت پکیج ✅

پروژه با `uv` و `pyproject.toml` ساخته شد. وابسته‌های اصلی: `statsbombpy`،
`socceraction`، `kloppy`، `pandas`، `numpy`، `pyarrow`، `mplsoccer`،
`matplotlib`، `tqdm` + `pytest` به‌عنوان extra گروه `dev`.

**مشکلی که خوردم و حلش کردم:**
`socceraction 1.5.3` وابسته به `pandera 0.17.2` است. `pandera 0.17` با
`multimethod >= 2.0` سازگار نیست و در زمان import می‌شکند:

```
ImportError: cannot import name 'overload' from 'multimethod'
```

از آنجا که هیچ‌کدام از این دو پکیج این را در metadata خود declare نکرده‌اند،
`uv` آخرین نسخه `multimethod` (۲.۱) را نصب می‌کرد و کل زنجیره socceraction
از کار می‌افتاد. راه‌حل: پین صریح در `pyproject.toml`:

```toml
"multimethod<2.0",
```

یک تست رگرسیون (`test_socceraction_pandera_stack_is_healthy`) هم اضافه شد تا اگر
روزی این پین برداشته شد، بلافاصله قرمز شود.

### تسک ۰.۲ — زنجیره Prolog / ILP ✅

**مشکل اصلی این تسک:** Popper 5.0.0 به `janus_swi` وابسته است و `janus_swi`
در زمان build چک می‌کند که SWI-Prolog نصب‌شده حداقل نسخه **9.1.12** باشد.
نسخه‌ای که مخزن رسمی Ubuntu 24.04 می‌دهد **9.0.4** است، پس نصب شکست خورد:

```
RuntimeError: At least SWI-Prolog version 9.1.12 is required
```

**راه‌حل در محیط من:** SWI-Prolog **9.2.9** را از سورس build و نصب کردم
(cmake + ninja، بدون packages اضافی).

**راه‌حل روی ویندوز شما (خیلی ساده‌تر):** فقط installer رسمی نسخه ۹.۲ به بالا را
نصب کنید. علاوه بر این، `janus_swi` روی PyPI برای ویندوز wheel آماده دارد
(`janus_swi-1.5.3-cp314-cp314-win_amd64.whl`)، بنابراین **نیازی به Visual
Studio Build Tools ندارید**. جزئیات کامل در `docs/SETUP_WINDOWS.md`.

**نتیجه تست:** مثال رسمی Popper بدون خطا اجرا شد:

```
2.1s ********** SOLUTION **********
2.1s Precision:1.00 Recall:1.00 TP:108 FN:0 TN:356 FP:0 Size:16
```

> **نکته اصلاحی نسبت به سند فازبندی:** Popper خروجی خود را روی **stderr**
> می‌نویسد نه stdout. تست تسک ۰.۲ در نسخه اولیه من روی stdout دنبال کلمه
> `SOLUTION` می‌گشت و قرمز می‌شد. تست اصلاح شد تا هر دو stream را بررسی کند.

> **نکته درباره Clingo:** سند فنی Clingo را به‌عنوان وابستگی مجزا ذکر کرده بود.
> در عمل Popper 5 آن را به‌صورت پکیج پایتون (`clingo`) در محیط خودش نصب می‌کند و
> نیاز به نصب سیستمی جداگانه نیست. همچنین Popper 5 به‌صورت پیش‌فرض از solver
> **CP-SAT** (از ortools) برای مرحله combine استفاده می‌کند، نه از Clingo تنها.

> **معماری محیط‌ها:** Popper به Python **3.14** نیاز دارد، ولی `socceraction`
> هنوز روی ۳.۱۴ کار نمی‌کند. بنابراین دو محیط مجزا داریم: محیط پروژه (۳.۱۲) و
> محیط خود Popper که `uv` داخل پوشه Popper مدیریت می‌کند. پروژه، Popper را
> به‌صورت subprocess صدا می‌زند. این یک تصمیم آگاهانه است، نه دور زدن مشکل.

### تسک ۰.۳ — دسترسی به دیتاست ✅

`competition_id=11, season_id=90` → **۳۵ بازی**، همگی با حضور بارسلونا
(معیار پذیرش: ≥ ۲۰).

### تسک ۰.۴ — دسترسی به داده ۳۶۰ ✅

**هر ۳۵ بازی** داده ۳۶۰ کامل دارند (۳۵/۳۵). این بهتر از انتظار سند فنی بود که
گفته بود «بخشی از بازی‌های فصل ۲۰۲۰/۲۱».

---

## ۲. فاز ۱ — بارگذاری و آماده‌سازی داده خام

### تسک ۱.۱ — دانلود و کش ✅

ماژول: `src/tactical_ilp/data/load_raw_data.py`

سه artefact برای هر بازی در `data/raw/` کش می‌شود:

| فایل | محتوا |
|---|---|
| `events_<id>.parquet` | جریان Event خام (schema socceraction) |
| `lineup_<id>.parquet` | ترکیب + پست شروع + دقایق بازی |
| `frames_<id>.parquet` | فریم‌های ۳۶۰ (یک ردیف به ازای هر بازیکن قابل‌مشاهده در هر Event) |

جزئیات پیاده‌سازی:
- ستون‌های `object` (dict/list) پیش از نوشتن در parquet به JSON تبدیل و هنگام
  خواندن برگردانده می‌شوند.
- ستون `visible_area` فریم‌های ۳۶۰ حذف شد (حجیم و بلااستفاده). فقط
  `event_id, teammate, actor, keeper, x, y` نگه داشته می‌شود.
- اگر بازی داده ۳۶۰ نداشته باشد، یک DataFrame خالی کش می‌شود تا دوباره تلاش
  دانلود انجام نشود.

تست `test_second_run_reads_from_cache` با monkeypatch کردن `StatsBombLoader`
اثبات می‌کند اجرای دوم اصلاً به شبکه نمی‌رود.

**نتیجه:** ۳۵/۳۵ بازی کش شد؛ مجموع ردیف‌های ۳۶۰ ≈ ۲.۲ میلیون.

### تسک ۱.۲ — تبدیل به SPADL ✅

ماژول: `src/tactical_ilp/data/spadl_converter.py`

**تصمیم معماری مهم — چارچوب مرجع واحد:**
تابع `convert_to_actions(events, home_team_id)` در socceraction به‌صورت داخلی
`_fix_direction_of_play` را صدا می‌زند. با پاس دادن **team_id بارسلونا** به‌جای
تیم میزبان، یک چارچوب مرجع ثابت برای کل پروژه قفل شد:

```
x ∈ [0, 105]   x=0 دروازه خودی، x=105 دروازه حریف
y ∈ [0, 68]    y=0 خط طولی سمت راست از دید جهت حمله بارسا
```

اکشن‌های حریف هم به همین چارچوب آینه می‌شوند (نه چارچوب خودشان). این باعث
می‌شود جمله‌ای مثل «حریف به یک‌سوم دفاعی ما رسید» مستقیماً با `x <= 35` بیان شود.

**اعتبارسنجی این تصمیم** (تست `test_reference_frame_barcelona_attacks_positive_x`):

| تیم | میانگین x شوت‌ها |
|---|---|
| بارسلونا | **۹۲.۵** |
| حریف | **۱۳.۴** |

**نتیجه:** ۸۲٬۰۴۸ اکشن SPADL، ۶۴.۲٪ متعلق به بارسا. توزیع نوع اکشن:
pass ۳۶٬۸۰۰، dribble ۳۴٬۵۹۴، take_on ۱۲۰۶، tackle ۱۰۹۳، ...

علاوه بر schema استاندارد SPADL، چهار ستون از داده خام حمل شد:
`under_pressure`، `counterpress` (برای مدل فشار فاز ۳) و
`possession`، `possession_team_id` (برای تفکیک سکانس در فاز ۲).

### تسک ۱.۳ — پیوند فریم‌های ۳۶۰ ✅

ماژول: `src/tactical_ilp/data/freeze_frames.py`

مراحل تبدیل مختصات:
1. مقیاس‌دهی StatsBomb (120×80) → SPADL (105×68) با معکوس‌کردن محور y
   (دقیقاً همان فرمولی که socceraction برای Eventها به کار می‌برد)
2. آینه‌کردن فریم‌های اکشن‌های حریف به چارچوب بارسا

هفت ویژگی زمینه‌ای از هر فریم استخراج شد:

| ویژگی | معنا |
|---|---|
| `ff_n_visible` | تعداد بازیکنان داخل کادر |
| `ff_opponents_5m` | حریفان در شعاع ۵ متری توپ |
| `ff_opponents_3m` | حریفان در شعاع ۳ متری توپ |
| `ff_teammates_5m` | هم‌تیمی‌ها در شعاع ۵ متری |
| `ff_nearest_opponent_dist` | فاصله نزدیک‌ترین حریف |
| `ff_opponents_ahead` | حریفانی که جلوتر از توپ‌اند |
| `ff_teammates_ahead` | هم‌تیمی‌هایی که جلوتر از توپ‌اند |

**معیار پذیرش:** ≥ ۷۰٪ پوشش روی اکشن‌های Build-up.
**نتیجه: ۹۲.۶۵٪** (پوشش کل: ۹۱.۷۰٪).

### تسک ۱.۴ — هویت و نقش بازیکنان ✅

ماژول: `src/tactical_ilp/data/players_reference.py`

طبق بخش ۲.۶ سند فنی، هویت و نقش دو بُعد مستقل‌اند. نقش اصلی هر بازیکن =
نقشی که بیشترین **دقیقه بازی** را در آن داشته (نه بیشترین تعداد حضور)، از روی
پست شروع در ترکیب. ۲۴ پست StatsBomb به ۹ نقش تاکتیکی نگاشت شد.

**نتیجه:** ۲۵ بازیکن، ۲۴ نفر با ≥ ۵ بازی. نمونه خروجی:

| atom | بازیکن | نقش | بازی | دقیقه |
|---|---|---|---|---|
| `p_5503` | Lionel Messi | forward | ۳۵ | ۳۱۹۱ |
| `p_5203` | Sergio Busquets | defensive_midfielder | ۳۳ | ۲۴۰۷ |
| `p_5213` | Gerard Piqué | central_defender | ۱۸ | ۱۵۵۷ |
| `p_20055` | Marc-André ter Stegen | goalkeeper | ۲۹ | ۲۷۴۶ |
| `p_30486` | Pedri | central_midfielder | ۳۵ | ۲۳۳۳ |

اسامی به atom امن Prolog تبدیل شدند (حذف دیاکریتیک، snake_case):
`Marc-André ter Stegen` → `marc_andre_ter_stegen`.

---

## ۳. فاز ۲ — تفکیک سکانس‌ها و طراحی Negative Sampling

### تسک ۲.۱ — قاعده شروع/پایان سکانس Build-up ✅

ماژول: `src/tactical_ilp/sequences/sequence_builder.py`

تعریف عملیاتی:

| مرحله | قاعده |
|---|---|
| **شروع** | ابتدای یک possession chain متعلق به بارسا که اولین اکشنش `start_x <= 35` باشد |
| **پایان** | اولین موردی که رخ دهد: `exit` (خروج موفق از یک‌سوم دفاعی) / `failed_action` / `shot` / پایان possession / سقف ۲۵ اکشن |
| **نگهداری** | فقط سکانس‌های ۲ تا ۲۵ اکشنی |

از `possession` خود StatsBomb برای زنجیره‌بندی استفاده شد (به‌جای تشخیص دستی
تغییر تیم)، که به‌مراتب مقاوم‌تر در برابر نویز است.

**نتیجه:** ۱۰۶۶ سکانس، ۳۰.۵ در هر بازی، طول min/median/max = ۲/۴/۲۱.

| دلیل پایان | تعداد |
|---|---|
| `exit` | ۷۲۹ |
| `failed_action` | ۱۹۶ |
| `possession_end` | ۱۴۱ |

### تسک ۲.۲ — تعریف عملیاتی «پیامد منفی» و برچسب‌گذاری ✅

سند مستقل: **`docs/negative_outcome_definition.md`**
ماژول: `src/tactical_ilp/sequences/outcome_labeler.py`

تعریف سه‌شرطی:

| شرط | معیار |
|---|---|
| A1 | از دست دادن توپ در طول سکانس یا تا ۸ ثانیه بعد |
| A2 | محل Turnover در نیمه خودی (`x <= 52.5`) |
| A3 | حداقل یکی از: شوت حریف تا ۱۲ ثانیه بعد، **یا** رسیدن حریف به یک‌سوم دفاعی ما تا ۱۲ ثانیه بعد |

**نتیجه:** ۱۹۱ `negative` (۱۷.۹۲٪) / ۸۷۵ `non_negative` (۸۲.۰۸٪).
معیار پذیرش (هیچ کلاسی زیر ۱۰٪) **بدون شل‌کردن تعریف** برقرار شد.

اعتبارسنجی منطقی تعریف (crosstab):

| دلیل پایان | negative | non_negative |
|---|---|---|
| `exit` | ۳۷ | ۶۹۲ (۹۵٪) |
| `failed_action` | ۱۰۰ | ۹۶ |
| `possession_end` | ۵۴ | ۸۷ |

این جدول نشان می‌دهد «اکشن ناموفق» ≠ «پیامد منفی» — که دقیقاً هدف شرط A3 بود.

**همگام‌سازی سند و کد:** ماژول `docs_sync.py` مقادیر عددی داخل سند markdown را
parse می‌کند و با `config.py` مقایسه می‌کند. تست
`test_negative_outcome_definition_documented` قرمز می‌شود اگر یکی از این دو
تغییر کند و دیگری نه. یعنی سند نمی‌تواند بی‌سروصدا از کد جدا بیفتد.

### تسک ۲.۳ — سکانس‌های چندبازیکنی نامزد ✅

ماژول: `src/tactical_ilp/sequences/player_subsets.py`

مدل تعامل: `passes_to(P1, P2, Seq)` وقتی اکشن *i* یک پاس موفق از P1 باشد و
اکشن *i+1* همان سکانس توسط بازیکن دیگری (P2) انجام شود. زنجیره سه‌نفره
(`chain/4`) از دو لینک پیاپی با سه بازیکن متمایز ساخته می‌شود.

**نتیجه:**

| معیار | مقدار |
|---|---|
| لینک‌های پاس | ۲۱۸۱ |
| جفت‌های متمایز بازیکن | ۳۲۹ |
| زنجیره‌های سه‌نفره | ۱۰۱۱ |
| سکانس‌های دارای ≥۱ جفت | **۸۷.۲۴٪** (معیار: ≥۵۰٪) |
| سکانس‌های دارای ≥۱ سه‌تایی | ۴۸.۳۱٪ |

پرتکرارترین لینک‌ها (یک سیگنال اعتبارسنجی خوب، چون دقیقاً الگوی بازیسازی
شناخته‌شده بارسا را نشان می‌دهد):

```
ter Stegen  -> Lenglet     84
Lenglet     -> Jordi Alba  57
Lenglet     -> ter Stegen  57
ter Stegen  -> Mingueza    52
ter Stegen  -> Jordi Alba  51
```

> **یادداشت دامنه:** روابط `supports` / `covers` عمداً در این تسک ساخته نشدند.
> آن‌ها به فریم ۳۶۰ و قضاوت تاکتیکی نیاز دارند، نه به زنجیره توپ — یعنی جای
> آن‌ها تسک ۴.۳ است، نه ۲.۳.

### تسک ۲.۴ — استراتژی Negative Sampling ✅

سند مستقل: **`docs/negative_sampling_strategy.md`**
ماژول: `src/tactical_ilp/sequences/negative_sampling.py`

روش: **نمونه‌گیری لایه‌ای هم‌زمینه**. هر نمونه منفی از همان لایه نمونه مثبت
متناظرش انتخاب می‌شود:

```
stratum = (start_zone, pressure_bucket, length_bucket)
```

نسبت: ۲ منفی به ازای هر مثبت. نردبان Fallback چهارسطحی با ثبت
`match_level` برای هر نمونه. seed ثابت = ۲۰۲۰۲۱ برای تکرارپذیری.

**نتیجه:**

| معیار | مقدار |
|---|---|
| نمونه مثبت | ۱۹۱ |
| نمونه منفی | ۳۸۲ (۱:۲.۰۰) |
| از لایه دقیق | ۳۶۷ (۹۶.۱٪) |
| با یک پله شل‌شدگی | ۱۵ (۳.۹٪) |
| از استخر سراسری | ۰ |
| **همپوشانی توزیع ناحیه‌ای** | **۱.۰۰۰** (معیار: ≥۰.۴۰) |

**پیامد مهم برای فاز ۵ (حتماً بخوانید):** همپوشانی ۱.۰۰۰ یعنی متغیر `zone`
عامدانه بی‌اطلاع شده است. پس هر قانونی که Popper تولید کند و فقط روی
`sequence_start_zone` یا `pressure_bucket` تکیه داشته باشد، **طبق ساختِ خود
نمونه‌ها** نمی‌تواند قدرت تفکیک داشته باشد. اگر چنین قانونی در خروجی دیدید،
نشانه باگ است نه یافته.

---

## ۴. فاز ۳ — لایه Rule-based Feature Extractor

### تسک ۳.۱ — Predicateهای هندسی پایه ✅

ماژول: `src/tactical_ilp/features/geometry.py` (توابع خالص، بدون side-effect)

| تابع | خروجی |
|---|---|
| `occupies_half_space(x, y)` / `lane_of(y)` | `right_wing` / `right_half_space` / `central_channel` / `left_half_space` / `left_wing` |
| `third_of(x)` | `defensive_third` / `middle_third` / `final_third` |
| `is_zone14(x, y)` | ناحیه مرکزی جلوی محوطه جریمه حریف |
| `under_pressure(...)` | مدل فشار مبتنی بر ۳۶۰ با fallback به فلگ StatsBomb |
| `pressure_level(...)` | `no_pressure` / `light_pressure` / `heavy_pressure` |
| `pass_direction(...)` | `forward` / `sideways` / `backward` |
| `pass_length_class(...)` | `short` / `medium` / `long` |
| `is_progressive(...)` | قاعده استاندارد کاهش ۲۵٪ فاصله تا دروازه |

مدل لاین‌ها: تقسیم ۵ لاینه عرض زمین (۱۳.۶ متری). `y=0` سمت راست از دید جهت
حمله است، بنابراین `occupies_half_space(70, 15) == "right_half_space"` و
`occupies_half_space(70, 34) == "central_channel"` — دقیقاً مطابق تست سند
فازبندی.

مدل فشار دو منبعی است: اگر فریم ۳۶۰ موجود باشد، تعداد حریفان در شعاع ۵ متر
تصمیم می‌گیرد؛ در غیر این صورت فلگ `under_pressure` خود StatsBomb. این باعث
می‌شود predicate **کامل (total)** باشد و در ۷.۳٪ اکشن‌های بدون فریم هم مقدار
معنادار برگرداند.

### تسک ۳.۲ — تولید Fact برای یک بازی نمونه ✅

ماژول: `src/tactical_ilp/prolog/prolog_facts.py`

**نتیجه:** `data/prolog/facts_match_3764440.pl` — ۲۱ سکانس، ۲۱۱۲ fact.
`swipl consult` تمیز (exit 0، بدون هیچ Warning).

**مشکلی که خوردم:** نسخه اول فایل شامل کاراکتر em-dash (`—`) در کامنت هدر بود
و SWI-Prolog هشدار `Illegal multibyte Sequence` می‌داد. هرچند exit code صفر
می‌ماند، این روی ویندوز با code page غیر-UTF-8 می‌تواند به خطای واقعی تبدیل
شود. کل خروجی به **ASCII خالص** تبدیل شد و یک تست اختصاصی
(`test_sample_fact_file_is_ascii`) اضافه شد.

ساختار فایل: factها بر اساس predicate **گروه‌بندی و مرتب** می‌شوند، بنابراین
نیازی به `:- discontiguous` نیست و فایل برای شما قابل خواندن است.

### تسک ۳.۳ — تعمیم به کل دیتاست ✅

**نتیجه:** `data/prolog/facts_all.pl` — ۳۵ بازی، ۱۰۶۶ سکانس، **۸۶٬۴۹۶ fact**،
لود تمیز در SWI-Prolog.

| Predicate | تعداد | دسته |
|---|---|---|
| `action/1`, `in_sequence/2`, `action_order/2`, `actor/2`, `action_type/2`, `action_result/2` | ۵۳۲۹ هرکدام | اسکلت اکشن |
| `at_third/2`, `at_lane/2`, `pressure_level/2` | ۵۳۲۹ هرکدام | هندسه و فشار |
| `successful/1` | ۵۱۳۱ | نتیجه |
| `pass_direction/2`, `pass_length/2` | ۵۰۴۶ هرکدام | شکل پاس |
| `has_freeze_frame/1`, `opponents_within/2` | ۴۵۴۲ هرکدام | ۳۶۰ |
| `involved_in/2` | ۲۹۱۵ | عضویت |
| `under_pressure/1` | ۲۲۴۳ | فشار |
| `passes_to/3` | ۲۱۸۱ | رابطه‌ای |
| `sequence/1`, `sequence_match/2`, `sequence_label/2`, `sequence_length/2`, `sequence_start_zone/2` | ۱۰۶۶ هرکدام | سکانس |
| `chain/4` | ۱۰۱۱ | رابطه‌ای سه‌نفره |
| `progressive/1` | ۲۵۰ | هندسه |
| `failed/1` | ۱۹۸ | نتیجه |
| `player_identity/1`, `player_name/2`, `has_role/2`, `plays_for/2` | ۲۵ هرکدام | هویت (§۲.۶) |

هدر هر فایل تولیدی صریحاً هشدار می‌دهد که **تمام** این predicateها
**Fact ثابت از پیش محاسبه‌شده** هستند و نه rule زنده Prolog — دقیقاً همان
روشن‌سازی‌ای که سند فازبندی نسخه ۲ برای فاز ۳ خواسته بود.

---

## ۵. انحراف‌ها از سند فازبندی (لطفاً بازبینی کنید)

| # | محل | سند فازبندی گفته بود | چه کردم | چرا |
|---|---|---|---|---|
| ۱ | تست ۳.۳ | `count_predicate_occurrences(file, "involved_in") == len(labeled_sequences_df)` | `count(sequence/1) == len(labeled)` **و** تعداد سکانس‌های **متمایز** در `involved_in` `== len(labeled)` | `involved_in/2` یک fact به ازای هر **بازیکن در هر سکانس** است (۲۹۱۵ ≠ ۱۰۶۶). برابری اصلی فقط برای سکانس‌های تک‌بازیکنی درست است و با طراحی چندبازیکنی بخش ۲.۵ سند فنی در تضاد است. هدف تست («factها دقیقاً همان سکانس‌های برچسب‌خورده را پوشش دهند») حفظ شد. |
| ۲ | تست ۰.۲ | `result = subprocess.run([...]); assert returncode == 0` | علاوه بر returncode، وجود `SOLUTION` در stdout **یا stderr** | Popper لاگش را روی stderr می‌نویسد. صرف returncode=0 اثبات نمی‌کند قانونی تولید شده. |
| ۳ | تست ۱.۴ | `spadl_all_matches["player_id"].value_counts() >= 5` | همان، ولی فیلترشده روی اکشن‌های **بارسا** | نسخه اصلی بازیکنان حریف را هم شامل می‌شد، در حالی که `players_reference` طبق متن تسک فقط بازیکنان بارساست. |
| ۴ | فاز ۰ | Clingo به‌عنوان نصب سیستمی مجزا | نصب نشد | Popper 5 آن را به‌صورت پکیج پایتون در محیط خودش می‌آورد و پیش‌فرض از CP-SAT استفاده می‌کند. |
| ۵ | تسک ۲.۳ | «پاس، پوشش، حمایت» | فقط پاس (`passes_to`, `chain`) | `supports`/`covers` نیازمند فریم ۳۶۰ + قضاوت تاکتیکی هستند؛ جای آن‌ها تسک ۴.۳ است. |

---

## ۶. نکاتی که روی فاز ۴ و ۵ اثر می‌گذارند

1. **`progressive/1` فقط ۲۵۰ بار رخ داده** (۴.۷٪ اکشن‌ها). قاعده استاندارد
   «کاهش ۲۵٪ فاصله تا دروازه» در یک‌سوم دفاعی بسیار سخت‌گیر است (از `x=20`
   نیاز به پاس ~۲۱ متری رو به جلو دارد). اگر در فاز ۴ می‌خواهید این را به‌عنوان
   `body_pred` استفاده کنید، احتمالاً باید آستانه‌اش بازتعریف شود.

2. **۵۳۲۹ اکشن در ۱۰۶۶ سکانس** یعنی میانگین ۵ اکشن در هر سکانس. فضای جستجوی
   Popper با `max_vars=6` و این حجم fact قابل مدیریت است، ولی `chain/4` (آریتی ۴)
   گران است. پیشنهاد: در `bias.pl` نسخه اول آن را کنار بگذارید.

3. **`opponents_within/2` مقدار عددی برمی‌گرداند.** برای Popper بهتر است یا
   type annotation بگیرد یا به نسخه گسسته (`pressure_level/2`) تکیه کنید.

4. **مثال‌های ILP آماده‌اند** ولی هنوز به `exs.pl` تبدیل نشده‌اند —
   آن تسک ۵.۱ است. جدول `data/processed/ilp_examples.parquet` شامل
   `sequence_id`, `example_class` (`pos`/`neg`), `match_level` است.

---

## ۷. ساختار فایل‌های پروژه

```
football-tactical-ilp/
├── pyproject.toml                    وابستگی‌ها (شامل پین multimethod<2.0)
├── uv.lock                           قفل دقیق نسخه‌ها
├── README.md                         معرفی و اجرای سریع
├── report.md                         ← همین فایل
├── .gitignore
│
├── docs/
│   ├── SETUP_WINDOWS.md              راهنمای کامل نصب و تست روی ویندوز
│   ├── negative_outcome_definition.md    تحویل‌دادنی تسک ۲.۲
│   └── negative_sampling_strategy.md     تحویل‌دادنی تسک ۲.۴
│
├── scripts/
│   ├── run_phase0_check.py           چک‌لیست محیط
│   ├── run_phase1.py                 دانلود + SPADL + ۳۶۰ + بازیکنان
│   ├── run_phase2.py                 سکانس + برچسب + جفت + نمونه‌گیری
│   └── run_phase3.py                 تولید Prolog facts
│
├── src/tactical_ilp/
│   ├── config.py                     تمام ثابت‌ها و مسیرها (منبع حقیقت)
│   ├── docs_sync.py                  همگام‌سازی سند markdown با config
│   ├── data/
│   │   ├── load_raw_data.py          تسک ۱.۱
│   │   ├── spadl_converter.py        تسک ۱.۲
│   │   ├── freeze_frames.py          تسک ۱.۳
│   │   └── players_reference.py      تسک ۱.۴
│   ├── sequences/
│   │   ├── sequence_builder.py       تسک ۲.۱
│   │   ├── outcome_labeler.py        تسک ۲.۲
│   │   ├── player_subsets.py         تسک ۲.۳
│   │   └── negative_sampling.py      تسک ۲.۴
│   ├── features/
│   │   └── geometry.py               تسک ۳.۱
│   └── prolog/
│       └── prolog_facts.py           تسک ۳.۲ و ۳.۳
│
├── tests/
│   ├── conftest.py                   fixtureها
│   ├── test_phase0_environment.py    ۹ تست
│   ├── test_phase1_data.py           ۱۲ تست
│   ├── test_phase2_sequences.py      ۲۰ تست
│   └── test_phase3_features.py       ۱۸ تست
│
└── data/            (در .gitignore — قابل بازتولید)
    ├── raw/         کش دانلود (~۵۰۰ MB)
    ├── processed/   جداول میانی parquet
    └── prolog/      facts_all.pl, facts_match_*.pl, facts_manifest.json
```

---

## ۸. خروجی داده‌ای تولیدشده

| فایل | محتوا |
|---|---|
| `data/processed/spadl_all_matches.parquet` | ۸۲٬۰۴۸ اکشن SPADL |
| `data/processed/spadl_with_360_all_matches.parquet` | همان + ۷ ویژگی ۳۶۰ |
| `data/processed/players_reference.parquet` | ۲۵ بازیکن با هویت و نقش |
| `data/processed/sequences.parquet` | ۱۰۶۶ سکانس Build-up |
| `data/processed/sequence_actions.parquet` | ۵۳۲۹ عضویت اکشن-سکانس |
| `data/processed/labeled_sequences.parquet` | سکانس‌ها + برچسب پیامد |
| `data/processed/sequence_pairs.parquet` | ۲۱۸۱ لینک پاس |
| `data/processed/sequence_triples.parquet` | ۱۰۱۱ زنجیره سه‌نفره |
| `data/processed/sequence_context.parquet` | سکانس‌ها + کلیدهای لایه‌بندی |
| `data/processed/ilp_examples.parquet` | ۵۷۳ نمونه ILP (۱۹۱ pos / ۳۸۲ neg) |
| `data/prolog/facts_all.pl` | ۸۶٬۴۹۶ fact |
| `data/prolog/facts_match_3764440.pl` | ۲۱۱۲ fact (بازی نمونه) |
| `data/prolog/facts_manifest.json` | شمارش هر predicate |

---

## ۹. گام بعدی — فاز ۴ (نیازمند تأیید شما)

فاز ۴ برخلاف فازهای قبل **خودکار نیست**. طبق سند فازبندی، روال هر تسک این است:
شما مفهوم را به زبان ساده توصیف می‌کنید → با هم به Predicate منطقی تبدیل
می‌کنیم → روی چند نمونه واقعی تست می‌زنیم → تأیید یا اصلاح.

پیش از شروع، این‌ها نیاز به تصمیم شما دارند:

1. **تأیید انحراف‌های بخش ۵** (به‌ویژه مورد ۱ درباره تست ۳.۳).
2. **تسک ۴.۰** می‌خواهد شما دستی برچسب بزنید: ≥۱۰ پاس برای Line-breaking،
   ≥۵ نمونه برای تردید، ≥۱۰ جفت برای supports/covers. من می‌توانم یک ابزار
   برچسب‌گذاری (با رسم mplsoccer از هر نمونه) بسازم تا این کار برایتان ساده شود —
   بگویید می‌خواهید یا نه.
3. **آیا `progressive` را با آستانه فعلی نگه داریم یا بازتعریف کنیم؟**

**تا زمانی که تأیید نکنید وارد فاز ۴ نمی‌شوم.**

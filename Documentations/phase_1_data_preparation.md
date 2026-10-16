# فاز ۱ — دریافت داده، تبدیل SPADL و آماده‌سازی مرجع مکانی و بازیکن

## هدف و محدوده

فاز ۱ داده‌ی بازی‌های FC Barcelona در La Liga فصل ۲۰۲۰/۲۱ را از StatsBomb Open Data می‌گیرد یا از cache بسته‌بندی‌شده می‌خواند، رویدادها را به جدول استاندارد SPADL تبدیل می‌کند، اطلاعات 360 را به اکشن‌ها پیوند می‌دهد و مرجع هویت/نقش بازیکنان را می‌سازد. خروجی این فاز ورودی فازهای بعدی برای تقسیم سکانس، استخراج ویژگی تاکتیکی و ساخت factهاست. هیچ یادگیری قانون ILP در فاز ۱ انجام نمی‌شود.

نقطه‌ی ورود [`scripts/run_phase1.py`](../scripts/run_phase1.py) چهار کار را به‌ترتیب اجرا می‌کند: ۱.۱ دریافت داده‌ی خام، ۱.۲ تبدیل SPADL، ۱.۳ پیوند فریم‌های 360، ۱.۴ ساخت مرجع بازیکنان. منطق جزئی در ماژول‌های زیر است:

- [`src/tactical_ilp/data/load_raw_data.py`](../src/tactical_ilp/data/load_raw_data.py)
- [`src/tactical_ilp/data/spadl_converter.py`](../src/tactical_ilp/data/spadl_converter.py)
- [`src/tactical_ilp/data/freeze_frames.py`](../src/tactical_ilp/data/freeze_frames.py)
- [`src/tactical_ilp/data/players_reference.py`](../src/tactical_ilp/data/players_reference.py)
- تنظیمات مشترک: [`src/tactical_ilp/config.py`](../src/tactical_ilp/config.py)

اجرای معمول و اجباری برای بازسازی:

```powershell
uv run python scripts/run_phase1.py
uv run python scripts/run_phase1.py --force
```

بدون `--force` هر تابع از cache موجود و معتبر استفاده می‌کند؛ `--force` cache را نادیده می‌گیرد و مراحل را از داده‌ی خام بازسازی می‌کند. اسکریپت فاز ۱ در انتها از `run_logging.py` برای آینه‌کردن خروجی در log زمان‌دار استفاده می‌کند؛ جزئیات در بخش «ثبت اجرا» آمده است.

## هویت دیتاست و قرارداد مختصات

در [`config.py`](../src/tactical_ilp/config.py) مجموعه‌ی هدف ثابت است: `COMPETITION_ID=11` (La Liga)، `SEASON_ID=90` (۲۰۲۰/۲۱)، `TEAM_NAME="Barcelona"` و `TEAM_ID=217`. ابعاد مرجع SPADL برابر ۱۰۵×۶۸ متر تعریف شده است. تمام اکشن‌های دو تیم در چارچوبی مشترک قرار می‌گیرند:

- Barcelona در جهت `+x` حمله می‌کند؛ دروازه‌ی حریف در `x=105` است.
- دروازه‌ی خودی Barcelona در `x=0` است، حتی برای اکشن‌های حریف.
- `y=0` سمت راست از دید تیمی است که در چارچوب Barcelona به سمت `+x` حمله می‌کند.

این جهت‌دهی عمدی است: ویژگی‌های هندسی فازهای بعد باید مقادیر `x/y` دو تیم را در یک چارچوب سازگار ببینند. converter نسخه‌ی قفل‌شده‌ی `socceraction 1.5.3` در `base._fix_direction_of_play` اکشن‌هایی را که `team_id != home_team_id` دارند، در هر دو محور حول ابعاد زمین mirror می‌کند؛ شرط آن team id است و در این نسخه بر اساس `period_id` تغییر نمی‌کند. در فراخوانی پروژه، `home_team_id=TEAM_ID` است. کنترل پایه در تست `test_reference_frame_barcelona_attacks_positive_x` انجام می‌شود؛ میانگین مختصات شوت‌های Barcelona باید بالاتر از ۸۰ و شوت‌های حریف پایین‌تر از ۲۵ باشد.

## ۱.۱ دریافت و cache داده‌ی خام

### فهرست بازی‌ها

`load_matches()` در `load_raw_data.py` ابتدا `data/raw/matches.parquet` را می‌خواند، مگر آن‌که `force=True` باشد یا فایل وجود نداشته باشد. در غیر این صورت `socceraction.data.statsbomb.StatsBombLoader` فهرست بازی‌های رقابت/فصل را می‌گیرد و `statsbombpy.sb.matches` برای نام میزبان و میهمان استفاده می‌شود. نام‌ها با `match_id` به جدول loader اضافه می‌شوند؛ اگر نام هر تیمی ناموجود باشد `RuntimeError` با شناسه‌ی بازی‌های ناقص صادر می‌شود. فهرست بازی‌ها در parquet ذخیره می‌شود.

`barca_match_ids()` بازی‌هایی را برمی‌گزیند که نام Barcelona در خانه یا میهمان است، شناسه‌ها را به `int` تبدیل و مرتب می‌کند. به این ترتیب همه‌ی پردازش‌ها یک ترتیب قطعی از بازی‌ها می‌گیرند.

### رویدادها و ترکیب‌ها

برای هر بازی، فایل‌های زیر زیر `data/raw/` نگهداری می‌شوند:

- `events_<match_id>.parquet`: رویدادهای StatsBomb در schema ارائه‌شده توسط socceraction؛
- `lineup_<match_id>.parquet`: فهرست بازیکنان و موقعیت شروع/دقایق؛
- `frames_<match_id>.parquet`: freeze-frameهای 360 به شکل long table؛
- `download_status.parquet`: تعداد رویدادها، بازیکنان، ردیف‌ها و رویدادهای دارای frame و پرچم وجود 360 برای هر مسابقه.

`load_events()` و `load_lineup()` cache را پیش از ساخت Loader بررسی می‌کنند؛ در نتیجه اجرای مجدد بدون `force` برای فایل موجود شبکه را لمس نمی‌کند. در رویدادها ستون‌های object حاوی dict/list برای ذخیره‌ی Parquet با JSON سریال می‌شوند؛ در زمان خواندن ستون‌های `extra`، `related_events` و `location` بازگشایی می‌شوند. این کار محدودیت Parquet برای اشیای دلخواه Python را دور می‌زند، درحالی‌که ساختار داده برای مصرف‌کنندگان حفظ می‌شود.

### فریم‌های 360 و مدیریت خطا

`_fetch_frames()` تا سه بار تلاش می‌کند و فاصله‌ی انتظار خطاهای موقت را به‌ترتیب ۲ و ۴ ثانیه می‌گذارد. فقط HTTP 404 نبود داده‌ی 360 برای بازی محسوب می‌شود؛ DataFrame خالی با schema ثابت `event_id, teammate, actor, keeper, x, y` بازگردانده و cache می‌شود تا همان بازی دوباره درخواست نشود. خطای اتصال/timeout و خطای HTTP غیر 404 دوباره تلاش می‌شوند و پس از اتمام retry به caller انتشار می‌یابند. این تمایز مهم است: خطای سرور یا شبکه نباید به‌اشتباه به‌عنوان «این بازی 360 ندارد» ثبت شود. خطاهای غیرقابل‌انتظار نیز بلعیده نمی‌شوند.

داده‌ی خام frame از ستون `location` به دو ستون عددی `x` و `y` بازآرایی می‌شود. flagهای `teammate`, `actor`, `keeper` به bool تبدیل می‌شوند. مختصات هنوز در مقیاس خام StatsBomb یعنی ۱۲۰×۸۰ است و در ماژول پیوند به مقیاس SPADL تبدیل می‌شود.

`download_all()` برای هر match هر سه loader را صدا می‌زند، آمار را چاپ می‌کند و status را ذخیره می‌کند. چون loaderها cache-aware هستند، این driver در اجرای بعدی فقط فایل محلی را می‌خواند. `cache_is_complete()` وجود cacheهای events/lineup/frames را برای فهرست بازی‌ها بررسی می‌کند.

## ۱.۲ تبدیل رویدادها به SPADL

### پاک‌سازی مکانی ورودی

`drop_locationless_events()` پیش از converter رویدادهایی را نگه می‌دارد که `location` معتبر دو مؤلفه‌ای و finite داشته باشند. رویدادهای اداری شناخته‌شده مانند `Starting XI`، `Half Start/End`، تعویض، ورود/خروج بازیکن، توقف مصدومیت و tactical shift که location ندارند، نادیده گرفته می‌شوند. سایر نوع‌های رویداد بدون location در خروجی جداگانه‌ی `dropped` نگه داشته می‌شوند تا تعداد و نوع‌شان در warning قابل مشاهده باشد؛ حذف کنترل‌شده مانع شکست converter روی candidateهای اکشن بدون مختصات می‌شود.

### تبدیل و enrichment

`build_spadl(match_id)` ابتدا cache مشتق‌شده‌ی `spadl_<match_id>.parquet` را می‌خواند، ولی فقط اگر نسخه‌ی schema با `PHASE1_VERSION` جاری باشد. سپس رویدادهای خام را به `socceraction.spadl.statsbomb.convert_to_actions(clean_events, home_team_id=TEAM_ID)` می‌دهد. converter با شناسه‌ی تیم منتخب جهت مختصات SPADL را تثبیت می‌کند؛ کد پروژه پس از آن چرخش اضافی برای بازی‌های خارج از خانه اعمال نمی‌کند. `assert_orientation()` در خروجی هر مسابقه‌ای که داده‌ی شوت/دروازه‌بان کافی دارد جهت را بررسی می‌کند و `run_phase1.py` نیز میانگین موقعیت شوت تیم‌ها را جداگانه به‌ازای مسابقه می‌سنجد. `spadl.add_names()` نام متنی type/result/body part را به ستون‌ها اضافه می‌کند.

از رویدادهای اصلی ستون‌های لازم برای مراحل بعد پیوند می‌شوند: `under_pressure`, `counterpress`, `possession`, `possession_team_id`, `index`, `duration`, `play_pattern_name` و در صورت وجود `shot_statsbomb_xg`. شناسه‌ی رویداد اصلی با `original_event_id` کلید join است. ستون `index` به `event_index` و `duration` به `event_duration` تغییر نام می‌یابد. ستون `pass_height_name` از `extra.pass.height.name` در metadata رخداد نگاشت می‌شود؛ برای اکشن مصنوعی فاقد شناسه‌ی رخداد اصلی، ارتفاع پاس نامشخص است. دو flag فشار به bool تبدیل می‌شوند؛ شناسه‌های possession با forward-fill و سپس backward-fill به ردیف‌های مصنوعی/بدون داده‌ی مستقیم منتقل و به nullable `Int64` تبدیل می‌شوند.

`time_seconds` اکشن‌های دارای event id از `timestamp` خام StatsBomb، با دقت میلی‌ثانیه و مبدأ همان دوره‌ی زمانی socceraction، بازسازی می‌شود. برای dribble مصنوعی زمان میانی دو همسایه به‌کار می‌رود؛ زمان‌های فاقد timestamp معتبر مقدار صحیح قبلی را نگه می‌دارند. ترتیب event داخل هر دوره غیرکاهشی حفظ می‌شود. این دقت برای tempo، فاصله‌ی زمانی اکشن‌ها و پنجره‌ی پنج‌ثانیه‌ای بازپس‌گیری استفاده می‌شود.

### منشأ اکشن و splitهای تبدیل

`action_source` برای ردیابی منشأ ردیف ساخته می‌شود:

- `event`: ردیف رویداد عادی با `original_event_id` یکتا؛
- `synthetic_dribble`: ردیف SPADL بدون `original_event_id` که converter به‌صورت اکشن carry/dribble مصنوعی ساخته؛
- `interception_split`: دو ردیفی که یک `original_event_id` مشترک دارند، معمولاً تبدیل interception به interception و pass.

این provenance برای تحلیل‌های بعدی لازم است تا ردیف مصنوعی با رویداد مستقل اشتباه نشود. برای نمونه، فریم 360 باید برای event یک‌بار شمرده شود، نه یک‌بار برای هر ردیف حاصل از split.

### زمان نگه‌داشتن توپ

تابع `compute_time_on_ball()` زمان event-based بین دریافت کنترل‌شده و release بعدی را از stream خام بازسازی می‌کند. رویدادها با `period_id` و `index` مرتب می‌شوند. فقط `Ball Receipt*` موفق holder را شروع می‌کند؛ outcome با نام `Incomplete` دریافت ناموفق محسوب می‌شود. holder شامل period، player، team، possession و زمان event است. تغییر دوره یا شماره‌ی possession، attribution را reset می‌کند.

releaseهای قابل انتساب عبارت‌اند از `Pass`, `Shot`, `Clearance`. اگر همان بازیکن یکی از آن‌ها را اجرا کند و فاصله‌ی زمانی از صفر تا ۱۰ ثانیه باشد، زمان در `time_on_ball_s` ثبت می‌شود. رخدادهای گسست‌زا (`Dispossessed`, `Miscontrol`, خطا، خروج توپ، offside، duel و موارد تعریف‌شده در مجموعه‌ی `DISRUPTIVE_TYPES`) انتساب را باطل می‌کنند. همچنین اگر بازیکن دیگری اکشن ball-taking انجام دهد، holder قبلی پاک می‌شود. این قاعده محافظه‌کارانه است: زمان نگه‌داشتن فقط وقتی نسبت داده می‌شود که دریافت قبلی و release بعدی بدون ابهام به یک بازیکن در همان possession متصل باشند. مستندات کد تصریح می‌کند که این شاخص زمان دقیق tracking نیست.

یک `Carry` که توسط همان holder اجرا شده، تایمر را reset نمی‌کند: شاخه‌ی `pid == holder["player"]` پیش از شاخه‌ی `BALL_TAKING_TYPES` بررسی می‌شود و فقط releaseهای تعریف‌شده مقدار را ثبت و holder را خالی می‌کنند. Carry یا اکشن ball-taking متعلق به بازیکن دیگری، holder قبلی را باطل می‌کند. سقف ۱۰ ثانیه مانع نسبت دادن فاصله‌های طولانی/مشکوک به یک مالکیت واحد است.

پس از بازسازی SPADL، تابع `_propagate_time_on_ball_to_dribbles()` بازه‌های receipt→release را نیز روی ردیف‌های `synthetic_dribble` منطبق اعمال می‌کند: بازیکن، تیم، possession و period باید برابر باشند و timestamp میانی synthetic dribble باید داخل بازه‌ی دریافت تا release باشد. این ردیف‌ها event id ندارند، بنابراین انتساب صرفاً از روی event id ممکن نبود. مقدار منتقل‌شده کل زمان نگه‌داشتن توپ برای آن بازه است و duration جداگانه‌ی حرکت مصنوعی یا اندازه‌گیری tracking نیست؛ بنابراین یک مقدار dwell ممکن است هم روی dribble میانی و هم روی اکشن release دیده شود. اکشن‌های synthetic که با بازه‌ی روشن تطبیق نکنند، `NaN` می‌مانند. زمان `interception` هم به‌طور صریح `NaN` است.

### ترتیب اکشن و schema نهایی

`order_actions()` اکشن‌ها را با `period_id`, `time_seconds`, `_order_key` به‌شکل stable مرتب می‌کند. برای event واقعی `event_index` کلید است. برای synthetic dribble بدون index، کلید برابر index اکشن قبلی به‌علاوه‌ی ۰٫۵ قرار می‌گیرد تا بلافاصله پس از event زمینه‌ساز بماند. الگوریتم stable است و ترتیب دو ردیف split با index یکسان را حفظ می‌کند.

ستون‌های استاندارد SPADL شامل شناسه‌ی بازیکن و تیم، نام نوع اکشن، مختصات شروع/پایان، نتیجه و نام اجزای SPADL هستند. ستون‌های افزوده‌ی پروژه شامل موارد provenance/timing/sequence نظیر `match_id`, `event_index`, `event_duration`, `action_source`, `time_on_ball_s`, `is_barca`, `action_id`, `phase1_version` و flagهای event است. schema اجباری در `REQUIRED_COLUMNS` ماژول آمده است: `player_id`, `team_id`, `type_name`, `start_x`, `start_y`, `end_x`, `end_y`, `result_name`.

`is_barca` با مقایسه‌ی `team_id == TEAM_ID` ساخته می‌شود. `action_id` پس از مرتب‌سازی از صفر در هر بازی افزایش می‌یابد؛ پس کلید سراسری برای جدول تجمیعی نیست و باید همراه `match_id` تفسیر شود.

### ارتفاع پاس

ارتفاع پاس از `extra.pass.height.name` رویداد StatsBomb استخراج و با `original_event_id` به اکشن SPADL پیوند داده می‌شود. برای اکشن‌های مصنوعی فاقد شناسهٔ رویداد متناظر، مقدار ارتفاع نامعلوم است.

## Cacheهای مشتق‌شده و نسخه‌بندی

`PHASE1_VERSION` در `config.py` اکنون `phase1-v7-orientation-fix-ms-time-360-linkage` است و شناسه‌ی نسخه‌ی داده‌ی مشتق‌شده محسوب می‌شود. هر جدول SPADL یا merged باید ستون `phase1_version` داشته باشد و تمام ردیف‌های آن با نسخه‌ی جاری برابر باشند. `read_if_current()` cache غایب، خالی، بدون ستون نسخه یا متعلق به نسخه‌ی دیگر را نامعتبر می‌داند؛ در صورت درخواست، مجموعه‌ی `match_id`های جدول نیز باید دقیقاً با مجموعه‌ی ورودی یکی باشد. raw cacheها مستقل از نسخه‌ی مشتق‌شده باقی می‌مانند. منطق این قرارداد در [`spadl_converter.py`](../src/tactical_ilp/data/spadl_converter.py) است.

مسیرهای مهم cache:

| داده | مسیر |
|---|---|
| رویداد، lineup، فریم خام و SPADL/merged هر بازی | `data/raw/<kind>_<match_id>.parquet` |
| فهرست بازی‌ها | `data/raw/matches.parquet` |
| خلاصه دانلود | `data/raw/download_status.parquet` |
| SPADL تجمیعی | `data/processed/spadl_all_matches.parquet` |
| اکشن‌ها با context 360 | `data/processed/spadl_with_360_all_matches.parquet` |
| مرجع بازیکنان | `data/processed/players_reference.parquet` |

نام cache هر بازی از تابع `cache_path()` در config می‌آید که به `RAW_DIR` متصل است. ریشه‌ی داده قابل تغییر با `TACTICAL_ILP_DATA` است؛ در غیر این صورت `data/` در ریشه‌ی پروژه استفاده می‌شود. دایرکتوری‌های raw/processed/prolog/reports هنگام import config ساخته می‌شوند. فاز ۱ ابتدا اگر هر سه خروجی اصلی processed وجود داشته باشند و دو جدول اکشن با نسخه‌ی جاری خوانده شوند، snapshot آماده را مستقیم استفاده می‌کند؛ در این مسیر، به cache raw یا فهرست آنلاین بازی نیاز نیست. فایل players در این شاخه parquet خوانده می‌شود. اجرای `--force` این راه میان‌بر را غیرفعال می‌کند.

## ۱.۳ پیوند 360 و ساخت ویژگی‌های فریم

### تبدیل مختصات

فریم‌ها در مختصات خام StatsBomb (طول ۱۲۰، عرض ۸۰) هستند. `frames_to_spadl_coordinates()` تبدیل یکنواخت زیر را انجام می‌دهد:

```text
fx = x / 120 * 105
fy = 68 - (y / 80 * 68)
```

پس از تبدیل پایه، فریم‌های event حریف در هر دو محور به مرجع تیم منتخب برگردانده می‌شوند؛ این تبدیل به home/away یا نیمه‌ی بازی وابسته نیست. چرخش اضافه‌ی مخصوص بازی‌های خارج از خانه حذف شده است. برای هر event، فریم با همان `event_id` به action وصل می‌شود و `_assert_actor_alignment()` فاصله‌ی نقطه‌ی actor تا مختصات شروع action را بررسی می‌کند.

برای جلوگیری از استفاده‌ی بی‌صدا از 360 فایل نامرتبط، `frame_linkage` سهم event idهای فریم را که در stream رویداد همان بازی پیدا می‌شوند می‌سنجد. اگر این سهم کمتر از `MIN_FRAME_LINKAGE=0.50` باشد، ویژگی‌های فریم برای آن بازی ساخته نمی‌شوند و بازی‌ها به‌طور پیش‌فرض از جدول merged کنار گذاشته و در `excluded_matches_360.json` ثبت می‌شوند. با `TACTICAL_ILP_EXCLUDE_UNLINKED_360=0` می‌توان حذف را خاموش کرد؛ در آن صورت ویژگی‌های 360 برای این بازی‌ها تهی می‌مانند.

### اتصال و جلوگیری از شمارش مضاعف

فریم با `event_id` و اکشن با `original_event_id` مرتبط است. پیش از merge، جدول مرجع action فقط event idهای معتبر را نگه می‌دارد و برای هر event تکراری یک ردیف (اولین ردیف) انتخاب می‌کند. این deduplication لازم است چون بعضی رویدادها، به‌ویژه splitهای interception، به چند ردیف SPADL تبدیل می‌شوند؛ تکثیر join باعث دوباره‌شماری بازیکنان حاضر و فشار می‌شد.

پس از تبدیل و mirror، فاصله‌ی اقلیدسی هر بازیکن قابل‌مشاهده از `start_x/start_y` اکشن محاسبه می‌شود. `teammate` در StatsBomb نسبت به عامل event است؛ پس `~teammate` برای event تیم بارسلونا همان حریف است. ویژگی‌های گروه‌شده با `event_id` عبارت‌اند از:

- `ff_n_visible`: تعداد تمام ردیف‌های بازیکن حاضر در frame؛
- `ff_opponents_5m`: تعداد بازیکنان حریف با فاصله‌ی حداکثر `PRESSURE_RADIUS_M=5` متر از محل شروع اکشن؛
- `ff_opponents_3m`: تعداد حریفان با فاصله‌ی حداکثر `TIGHT_PRESSURE_RADIUS_M=3` متر.

فاصله‌ها به مرز شامل هستند (`<=`). frame خالی خروجی aggregate خالی با schema ویژگی‌های ثابت دارد. برای eventهایی که frame ندارند، مقادیر شمارش `NaN` می‌مانند؛ صفر به معنی «frame موجود بوده و هیچ حریف نزدیک/بازیکن قابل‌شمارش نبوده» است و با فقدان مشاهده یکی نیست. `has_freeze_frame` از not-null بودن `ff_n_visible` ساخته می‌شود. ستون `freeze_frame` alias برابر همان تعداد قابل‌مشاهده است و برای قرارداد تست/مصرف‌کننده نگه داشته می‌شود.

### پوشش

`build_all()` برای هر بازی، نرخ frame را روی اکشن‌های `pass` و `dribble` چاپ می‌کند. `run_phase1.py` همین پوشش را به‌عنوان معیار اصلی گزارش می‌کند؛ سطح پذیرش اعلام‌شده در runner، حداقل ۷۰٪ از build-up actions است. این سنجه coverage است و نباید به‌عنوان تعداد دقیق frame کامل یا tracking پیوسته تفسیر شود: 360 داده‌ی snapshot بازیکنان حاضر در دید دوربین است و ممکن است بخشی از زمین/بازیکنان را نبیند.

## ۱.۴ مرجع هویت و نقش بازیکنان

### نگاشت موقعیت به نقش

`ROLE_MAP` در `players_reference.py` نقش‌های StatsBomb را به taxonomy تاکتیکی کوچک‌تر تبدیل می‌کند: goalkeeper، full_back، central_defender، defensive_midfielder، central_midfielder، wide_midfielder، attacking_midfielder، winger و forward. موقعیت‌های راست/چپ/مرکزی در نقش کلی ادغام می‌شوند؛ برای مثال Right/Left Back و Wing Back همگی `full_back` هستند. موقعیت‌های ناشناخته در map خروجی نقش معتبر ایجاد نمی‌کنند.

### تجمیع فصل و نقش اصلی

برای هر match id بارسلونا، lineup خوانده می‌شود و فقط ردیف‌های `team_id == 217` نگه داشته می‌شوند. نام موقعیت شروع به role تبدیل می‌شود؛ بازیکنان جانشین که starting position ندارند در رأی‌گیری نقش کنار گذاشته می‌شوند. مجموع `minutes_played` برای هر زوج بازیکن/نقش محاسبه و نقش دارای بیشترین دقیقه به‌عنوان نقش اصلی فصل انتخاب می‌شود. بازیکنی که هیچ شروعی/نقش قابل‌نگاشتی ندارد `unknown` می‌گیرد؛ حدس زدن نقش از شماره پیراهن عمداً انجام نمی‌شود.

آمار هر بازیکن به یک ردیف کاهش می‌یابد: نخستین `player_name`, `nickname`, `jersey_number`، تعداد مسابقات یکتا (`appearances`) و مجموع دقایق (`minutes_total`). `display_name` نام مستعار را در صورت وجود و در غیر این صورت نام رسمی می‌گیرد. `player_atom` برابر `p_<player_id>` است و اتصال بدون ابهام به شناسه‌ی StatsBomb را حفظ می‌کند. `name_atom` از نام نمایشی برای استفاده‌ی امن در Prolog ساخته می‌شود: Unicode به NFKD نرمال می‌شود، نشانه‌های ترکیبی حذف، متن lowercase و هر رشته‌ی غیر `[a-z0-9]` به underscore تبدیل می‌شود؛ underscoreهای ابتدا/انتها trim می‌شوند و رشته‌ی خالی به `unknown` می‌رسد. شناسه‌ی Prolog مبتنی بر id است و از برخورد احتمالی نام‌ها جلوگیری می‌کند.

ستون `team` برای همه‌ی ردیف‌ها `barcelona` است. `is_core` وقتی true است که تعداد حضور به `MIN_APPEARANCES_FOR_REFERENCE=5` برسد. خروجی بر اساس `is_core` نزولی و سپس دقایق نزولی مرتب می‌شود؛ همه‌ی بازیکنان حاضر در lineup در جدول باقی می‌مانند، نه فقط coreها. مرجع هویت و نقش را جدا نگه می‌دارد تا مصرف‌کننده بتواند بازیکن خاص یا نقش تاکتیکی را مستقل به‌کار گیرد.

فایل `data/processed/players_reference.parquet` یک ردیف به‌ازای هر بازیکن دارد. در runner تعداد کل/core و نمونه‌ی ۱۵ بازیکن اصلی چاپ می‌شود. مسیر cache در `PLAYERS_REFERENCE_PATH` است؛ تابع سازنده بدون force آن را مستقیم می‌خواند، و با force آن را از lineupها بازمی‌سازد.

## هماهنگ‌کننده‌ی فاز و پیام‌های پذیرش

`run_phase1.py` در ابتدای اجرا warningهای Python را فیلتر می‌کند و مسیر `src` را در `sys.path` می‌گذارد. پیش از دانلود، سه snapshot اصلی را بررسی می‌کند. اگر هر سه حاضر باشند و SPADL و merged از نظر version stamp جاری باشند، خروجی‌ها را reuse می‌کند، نمونه‌ی جهت شوت‌ها، پوشش 360 و شمار بازیکن‌ها را نشان می‌دهد و با کد صفر به مرحله‌ی فاز ۲ هدایت می‌شود.

در مسیر بازسازی، بخش‌های ۱.۱ تا ۱.۴ جداگانه banner می‌شوند. برای هر بازی وضعیت تعداد event و 360 row چاپ می‌شود؛ سپس شمار کل SPADL و میانگین `start_x` شوت برحسب `is_barca`، تعداد build-up actions، coverage 360 و نمونه‌ی مرجع بازیکنان چاپ می‌شود. جهت‌گیری به‌ازای مسابقه با guard سخت کنترل می‌شود؛ coverage 360 گزارش می‌شود و آستانه‌ی حداقل ۷۰٪ معیار تست داده‌ی فاز ۱ است.

`run_logged()` در [`scripts/run_logging.py`](../scripts/run_logging.py) خروجی stdout و stderr را هم‌زمان به کنسول و `data/reports/1_run_<UTC timestamp>.log` می‌فرستد. هر خط log timestamp UTC دارد؛ زمان اجرا و status نهایی درج می‌شوند. اگر استثنا رخ دهد، نوع و پیام ثبت و استثنا دوباره پرتاب می‌شود. این logging برای رصد notebook/اجرای طولانی است و منطق داده را تغییر نمی‌دهد.

## تست‌ها و کنترل کیفیت

تست‌های اصلی در [`tests/test_phase1_data.py`](../tests/test_phase1_data.py) و regressionهای مصنوعی آفلاین در [`tests/test_phase1_fixes.py`](../tests/test_phase1_fixes.py) هستند. قراردادهای مورد آزمون:

- وجود سه cache خام برای هر مسابقه و عدم فراخوانی شبکه در اجرای دوم؛
- وجود ستون‌های اجباری SPADL و معقول‌بودن نسبت تعداد action به event (۰٫۳ تا ۱ در نمونه‌ی بازی)؛
- قرارداشتن مختصات در محدوده‌ی ۱۰۵×۶۸؛
- جهت صحیح شوت‌های Barcelona و حریف؛
- پوشش حداقل ۷۰٪ freeze-frame برای pass/dribble و حدود منطقی شمارهای visible/opponent؛
- تکمیل مرجع بازیکنان دارای footprint معنادار، حداقل ۱۵ core، نقش یکتا برای هر player و جدایی identity از role؛
- عدم دوباره‌شماری فریم eventهایی که چند action row دارند؛
- رفتار فریم خالی؛
- attribution زمان توپ برای receipt→carry→pass، پاس one-touch، reset با loss/possession/player، incomplete receipt و سقف ۱۰ ثانیه؛
- حفظ تایمر بعد از Carry توسط همان بازیکن و انتقال زمان بازه به synthetic dribble متناظر، همراه با رد انتساب به synthetic dribble بازیکن دیگر؛
- حذف و گزارش locationlessهای غیراداری؛ ترتیب پایدار tieها و حفظ synthetic dribble بعد از event زمینه‌ای؛
- بررسی سازگاری cache با شناسه‌ی نسخه و مجموعه‌ی مسابقه؛
- cacheکردن پاسخ 404 به‌صورت خالی، اما cacheنکردن خطاهای موقت 503، 429، اتصال یا پاسخ خراب؛
- provenance، شمار frame، محدوده‌ی زمان نگه‌داشتن توپ و monotonic بودن `event_index` داخل هر دوره، وقتی جداول با کد جاری بازسازی شده‌اند.

تست‌های پایه داده به fixture/cache نیاز دارند؛ regressionهای `test_phase1_fixes.py` عمدتاً داده‌ی مصنوعی می‌سازند و آفلاین اجرا می‌شوند. آزمون‌ها به‌منزله‌ی قراردادند، ولی خروجی داده‌ی بسته‌بندی‌شده ممکن است با نسخه‌ی کد جاری هم‌نسخه نباشد؛ ستون `phase1_version` راه تشخیص این موضوع است.

## وابستگی‌ها و منابع بیرونی

وابستگی‌های Python در [`pyproject.toml`](../pyproject.toml) شامل `statsbombpy`, `socceraction`, `kloppy`, `pandas`, `numpy`, `pyarrow` و پکیج‌های تحلیل/نمایش است؛ `multimethod<2.0` برای سلامت stack پانداس/پاندرا محدود شده است. دریافت داده‌ی تازه به اینترنت و StatsBomb Open Data نیاز دارد. داده‌ی StatsBomb طبق توضیح loader باید هنگام انتشار هر نتیجه‌ی مشتق‌شده اعتباردهی شود.

## نقشه‌ی ارجاع به پیاده‌سازی

| جزء فنی | فایل |
|---|---|
| orchestration، cache snapshot و CLI `--force` | [`scripts/run_phase1.py`](../scripts/run_phase1.py) |
| dataset identity، پوشه‌ها، هندسه و آستانه‌ها/version | [`src/tactical_ilp/config.py`](../src/tactical_ilp/config.py) |
| فهرست مسابقه، دریافت/caching خام، retry فریم و status | [`src/tactical_ilp/data/load_raw_data.py`](../src/tactical_ilp/data/load_raw_data.py) |
| location filtering، converter SPADL، زمان توپ، ordering و version validation | [`src/tactical_ilp/data/spadl_converter.py`](../src/tactical_ilp/data/spadl_converter.py) |
| تبدیل مختصات، اتصال event به فریم، ویژگی‌های context و coverage | [`src/tactical_ilp/data/freeze_frames.py`](../src/tactical_ilp/data/freeze_frames.py) |
| taxonomy نقش، atom امن و مرجع فصل بازیکنان | [`src/tactical_ilp/data/players_reference.py`](../src/tactical_ilp/data/players_reference.py) |
| ثبت هم‌زمان کنسول و فایل log | [`scripts/run_logging.py`](../scripts/run_logging.py) |
| تست قراردادهای داده | [`tests/test_phase1_data.py`](../tests/test_phase1_data.py)، [`tests/test_phase1_fixes.py`](../tests/test_phase1_fixes.py)، [`tests/test_phase1_load_raw_data.py`](../tests/test_phase1_load_raw_data.py) |
| حدود نسخه Python و کتابخانه‌ها | [`pyproject.toml`](../pyproject.toml) |


# فاز ۳ — استخراج ویژگی‌های پایه و تولید فکت‌های Prolog

## هدف و مرز فاز

فاز ۳ اطلاعات ساخت‌یافته‌ی فازهای ۱ و ۲ را به واژگان نمادین قابل‌مصرف در Prolog تبدیل می‌کند. برای هر build-up انتخاب‌شده، اکشن‌های عضو و ویژگی‌های پایه‌ی مکانی، نتیجه، فشار، پیشروی و پاس به صورت **فکت‌های ground و ازپیش‌محاسبه‌شده** نوشته می‌شوند. فایل تولیدشده مجموعه‌ای از داده‌هاست؛ محاسبات Python این فاز را به قواعد Prolog زنده تبدیل نمی‌کند.

مراحل اصلی:

1. نگاشت ویژگی‌های پایه‌ی SPADL/360 به نمادهای گسسته؛
2. تولید فایل Prolog برای یک مسابقه‌ی نمونه؛
3. تولید فایل Prolog برای تمام مسابقات و ثبت manifest نسخه و شمار فکت‌ها.

در مسیر فعلی، ویژگی‌های هندسی پایه در [geometry.py](../src/tactical_ilp/features/geometry.py) هستند و [prolog_facts.py](../src/tactical_ilp/prolog/prolog_facts.py) فقط زیرمجموعه‌ی لازم برای واژگان جاری را صادر می‌کند. فاز ۴ نقاط منفرد فریم ۳۶۰ را جداگانه می‌خواند تا مفاهیم تاکتیکی مرکب را محاسبه کند؛ فاز ۳ فکت جداگانه‌ای برای هر نقطه‌ی بازیکن فریم تولید نمی‌کند. این جداسازی حجم فکت‌ها را محدود نگه می‌دارد و محاسباتی را که به شکل هندسی چندبازیکنی نیاز دارند در مرحله‌ی مصرف‌کننده‌ی آن‌ها انجام می‌دهد.

## قرارداد داده و وابستگی‌ها

### دستگاه مختصات

ویژگی‌ها از مختصات SPADL با زمین `105 × 68` متر استفاده می‌کنند:

- `x=0`: خط دروازه‌ی بارسلونا؛ `x=105`: خط دروازه‌ی حریف؛ حمله‌ی بارسلونا در جهت افزایش `x` است.
- `y=0`: خط طولی راست از دید حمله‌ی بارسلونا؛ `y=68`: خط طولی چپ.
- تمام مختصات اکشن از جدول SPADL فاز ۱ می‌آیند. تبدیل جهت حمله و هم‌ترازسازی 360 در همان فاز انجام شده است؛ ویژگی‌های فاز ۳ قرارداد مختصات را دوباره تبدیل نمی‌کنند.

ثابت‌های ابعاد و مرزها در [config.py](../src/tactical_ilp/config.py) هستند. داده‌ی مرجع اکشن و فریم‌های 360 در [freeze_frames.py](../src/tactical_ilp/data/freeze_frames.py) به `spadl_with_360_all_matches.parquet` می‌رسد. مستند جزئیات تبدیل داده در [فاز ۱](phase_1_data_preparation.md) آمده است.

### ورودی‌هایی که fact writer می‌خواند

تابع `generate_facts()` در [prolog_facts.py](../src/tactical_ilp/prolog/prolog_facts.py) داده‌های زیر را دریافت/می‌سازد:

| ورودی | داده‌ی مورد استفاده |
|---|---|
| `build_context()` | سکانس‌های برچسب‌خورده و زمینه‌ی هر سکانس؛ در فاز ۳ سطرهای `sequence` و `sequence_length` از آن صادر می‌شوند. |
| `build_sequences()` | جدول اعضای سکانس (`sequence_actions`/membership): شناسه‌ی مسابقه، اکشن، سکانس و ترتیب اکشن. |
| `build_merged_all()` | اکشن‌های SPADL و فیلدهای `player_id`, `type_name`, `result_name`, مختصات ابتدا/انتها، `under_pressure` و ستون‌های خلاصه‌ی فریم 360. |
| `build_players_reference()` | نقش بازیکنان برای `has_role/2`. |

اتصال membership به اکشن‌های merged با کلید `(match_id, action_id)` انجام می‌شود. فقط membershipهایی که `sequence_id` آن‌ها در context موجود است، صادر می‌شوند. اگر `match_ids` داده شود، پس از بارگذاری منابع، context، membership و اکشن‌ها به همان مسابقه‌ها محدود می‌شوند. در نتیجه فایل مسابقه‌ی نمونه همان schema و همان قواعد استخراج فایل کامل را دارد.

## وظیفه‌ی ۳.۱ — تعریف ویژگی‌های پایه

تابع‌های این بخش pure هستند: مقادیر عددی/تهی را می‌گیرند و برچسب یا مقدار Boolean برمی‌گردانند. تبدیل ویژگی‌های پایه به فکت در بخش بعد انجام می‌شود.

### تقسیم زمین به lane و third

`lane_of(y)` ابتدا `y` را به بازه‌ی `[0,68]` محدود می‌کند؛ سپس آن را به یکی از پنج lane می‌دهد. مرزها در `LANE_EDGES=(0,13.6,27.2,40.8,54.4,68)` و نام‌ها در `LANE_NAMES` پیکربندی شده‌اند:

| بازه‌ی y | lane |
|---|---|
| `0 <= y < 13.6` | `right_wing` |
| `13.6 <= y < 27.2` | `right_half_space` |
| `27.2 <= y < 40.8` | `central_channel` |
| `40.8 <= y < 54.4` | `left_half_space` |
| `54.4 <= y <= 68` | `left_wing` |

مرزهای میانی به lane بعدی تعلق دارند و `68` به آخرین lane تعلق می‌گیرد. این قرارداد با `y=0` سمت راست هماهنگ است.

`third_of(x)` از دید بارسلونا محاسبه می‌شود:

| شرط x | مقدار |
|---|---|
| `x <= 35` | `defensive_third` |
| `35 < x <= 70` | `middle_third` |
| `x > 70` | `final_third` |

مقادیر `35` و `70` از یک‌سوم‌های زمین (`PITCH_LENGTH/3` و `2*PITCH_LENGTH/3`) حاصل می‌شوند. `pitch_zone(x,y)` ترکیب این دو برچسب را با `|` می‌سازد، مانند `middle_third|left_half_space`. `pitch_zone()` برای context و ویژگی‌های فازهای دیگر هم استفاده می‌شود، اما در writer فاز ۳ فکت `pitch_zone/2` صادر نمی‌شود؛ دو مؤلفه‌ی آن جداگانه به شکل `at_third/2` و `at_lane/2` نوشته می‌شوند.

### فشار دودویی و سطح فشار

شمارش‌های `ff_opponents_5m` و `ff_opponents_3m` از خلاصه‌ی فریم‌های 360 فاز ۱ می‌آیند. شعاع‌ها در `config.py` برابر ۵ متر و ۳ متر هستند.

`under_pressure(n5, statsbomb_flag, min_opponents=1)`:

- اگر شمار ۵ متری موجود باشد، فقط همان شمار تعیین‌کننده است: `n5 >= 1` یعنی تحت فشار؛ `n5=0` یعنی بدون فشار. در این مسیر پرچم StatsBomb نتیجه را override نمی‌کند.
- اگر شمار ۵ متری تهی/NaN باشد، پرچم بومی StatsBomb fallback است.

`pressure_level(n3,n5,statsbomb_flag)` شدت را جداگانه و به‌ترتیب زیر bucket می‌کند:

1. `n3 >= 2` → `heavy_pressure`؛
2. در غیر این صورت `n3 >= 1` یا `n5 >= 2` → `light_pressure`؛
3. اگر `n5` موجود نباشد و پرچم StatsBomb روشن باشد → `light_pressure`؛
4. در بقیه‌ی حالات → `no_pressure`.

بنابراین فشار دودویی و شدت فشار معادل نیستند: یک حریف در فاصله‌ی ۳ تا ۵ متر `under_pressure` را روشن می‌کند ولی می‌تواند `pressure_level=no_pressure` بگیرد. در نبود فریم، پرچم StatsBomb می‌تواند فشار دودویی یا `light_pressure` بسازد. وقتی شمار فریم موجود و صفر است، شمار معتبر فریم بر پرچم StatsBomb اولویت دارد. برای intensity هیچ bucket جداگانه‌ی `unknown` صادر نمی‌شود.

### جهت و طول پاس

`pass_direction(start_x,start_y,end_x,end_y)` بردار جابه‌جایی را به `dx=end_x-start_x` و `dy=end_y-start_y` تبدیل می‌کند:

- `dx > 0` و `abs(dx) >= abs(dy)*0.5774` → `forward`؛
- `dx < 0` و همان شرط زاویه‌ای → `backward`؛
- همه‌ی حالت‌های دیگر، از جمله جابه‌جایی صفر یا مسیر عمدتاً عرضی → `sideways`.

شرط `0.5774` مخروطی با محور طولی `x` تعریف می‌کند؛ جابه‌جایی‌های مورب با نسبت عرضی به طولی تا حدود `sqrt(3)` همچنان forward/backward محسوب می‌شوند. جهت در دستگاه مختصات بارسلونا است.

`pass_length_class()` طول اقلیدسی ابتدا تا انتهای اکشن را محاسبه و bucket می‌کند: کمتر از ۱۰ متر `short`؛ از ۱۰ تا ۲۵ متر، شامل هر دو مرز، `medium`؛ بیش از ۲۵ متر `long`. واحد مختصات SPADL متر است. این دو تابع نام pass دارند ولی writer آن‌ها را برای مجموعه‌ی `PASS_LIKE` اعمال می‌کند: `pass`, `cross`, `freekick_short`, `goalkick`, `throw_in`, `corner_short`.

### پیشروی طولی

`is_progressive()` فقط افزایش مختصات طولی `x` را پیشروی می‌داند. تغییر `y` هیچ سهمی در معیار ندارد؛ پاس عرضی یا حرکتی که فقط به مرکز زمین/دروازه نزدیک‌تر می‌شود progressive نمی‌شود. اگر `start_x` یا `end_x` تهی/NaN باشد، پیشروی `False` است. همچنین جابه‌جایی طولی صفر/روبه‌عقب و شروع روی خط `x=105` پیشروی نمی‌شود.

آستانه بر مبنای **یک‌سوم محل آغاز اکشن** است:

| نقطه‌ی آغاز | شرط پیشروی طولی |
|---|---|
| `start_x <= 35` | `(end_x-start_x)/(105-start_x) >= 0.10` |
| `35 < start_x <= 70` | `(end_x-start_x)/(105-start_x) >= 0.25` |
| `start_x > 70` | `end_x-start_x >= 10 m` |

نسبت در دو زون اول بر طول باقی‌مانده تا خط دروازه‌ی حریف سنجیده می‌شود، نه بر طول ثابت هر یک‌سوم. بنابراین آستانه‌ی متری با `start_x` تغییر می‌کند. این سیاست عمداً حرکت طولی را می‌سنجد و مسیر مورب/فاصله تا مرکز دروازه را جایگزین آن نمی‌کند.

## وظیفه‌ی ۳.۲ و ۳.۳ — تولید فکت‌ها

### هویت نمادین و قالب فکت

در [prolog_facts.py](../src/tactical_ilp/prolog/prolog_facts.py):

- `_action_atom(match_id, action_id)` شناسه‌ی اکشن را به شکل `a_<match_id>_<action_id با حداقل ۵ رقم>` می‌سازد؛
- `_player_atom(player_id)` بازیکن را `p_<عدد>` می‌نامد؛
- `_atom()` متن را lowercase می‌کند، نویسه‌های غیرحرف/عدد/underscore را به `_` تبدیل می‌کند و atom معتبر می‌سازد؛
- `FactWriter.add()` سطرهایی از جنس `predicate(arg1,arg2).` می‌سازد؛ همه‌ی آرگومان‌ها ground هستند و متغیر Prolog ندارند.

`FactWriter.write()` فایل را با header توضیحی ASCII/UTF-8 می‌نویسد، فکت‌ها را بر اساس predicate گروه‌بندی می‌کند و در هر گروه رشته‌های یکتا را مرتب می‌نویسد. header هشدار می‌دهد که محتوا فکت ثابت است، مرجع مختصات بارسلونا را تکرار می‌کند و شعاع فشار ۵ متر را نشان می‌دهد. فکت تکراری یک predicate/آرگومان دوبار در فایل باقی نمی‌ماند.

### واژگان صادرشده در نسخه‌ی جاری

فهرست زیر خروجی فعال writer در نسخه‌ی فعلی است. `/n` تعداد آرگومان‌های predicate را نشان می‌دهد:

| Predicate | معنا و قاعده‌ی تولید |
|---|---|
| `sequence/1` | یک فکت برای هر sequence در context. |
| `sequence_length/2` | شناسه‌ی sequence و تعداد اکشن‌هایش (`n_actions`). |
| `has_role/2` | بازیکن عضو sequence و نقش او از players reference؛ اگر شناسه در مرجع پیدا نشود نقش `unknown` می‌گیرد. |
| `plays_for/2` | برای همان بازیکنان `barcelona`؛ sequenceهای این مجموعه build-up بارسلونا هستند. |
| `involved_in/2` | هر زوج یکتای `(player,sequence)`؛ برای هر اکشن/هر لمس تکراری فکت مجدد صادر نمی‌شود. |
| `in_sequence/2` | شناسه‌ی اکشن و sequence مربوط. |
| `action_order/2` | شماره‌ی ترتیب اکشن در sequence، همان `order_in_sequence`؛ order از صفر/مقدار membership فاز ۲ می‌آید. |
| `actor/2` | فقط اگر `player_id` اکشن تهی نباشد، اتصال اکشن به بازیکن. |
| `action_type/2` | نوع اکشن نرمال‌شده، مانند `pass` یا `dribble`. |
| `action_result/2` | نتیجه‌ی canonical اکشن نرمال‌شده. این فکت منبع نتیجه است. |
| `visible_player_count/2` | تعداد سطرهای بازیکن قابل مشاهده در فریم؛ فقط اگر شمار فریم موجود باشد صادر می‌شود. |
| `at_third/2` | یک‌سوم مختصات شروع اکشن (`third_of(start_x)`). |
| `at_lane/2` | lane مختصات شروع اکشن (`lane_of(start_y)`). |
| `progressive/1` | فقط برای اکشن‌هایی صادر می‌شود که معیار `is_progressive()` را پاس کنند. نبود فکت یعنی false در منطق مصرف‌کننده. |
| `under_pressure/1` | فقط وقتی تابع فشار دودویی True باشد صادر می‌شود؛ false با نبود فکت نمایش داده می‌شود. |
| `pressure_level/2` | برای هر اکشن، یکی از `heavy_pressure`, `light_pressure`, `no_pressure`. |
| `pass_direction/2` | برای اکشن‌های `PASS_LIKE` با هر چهار مختصات حاضر، یکی از `forward`, `sideways`, `backward`. |
| `pass_length/2` | برای همان اکشن‌های `PASS_LIKE` با مختصات کامل، یکی از `short`, `medium`, `long`. |
| `pass_end_lane/2` | برای اکشن `PASS_LIKE` با `end_y` موجود، lane نقطه‌ی پایان پاس؛ برای مقایسه‌ی lane مبدا/مقصد همان اکشن در `same_lane/1`. |

برای اکشن‌های عضو sequence، فکت‌های `in_sequence`, `action_order`, `action_type`, `action_result`, `at_third`, `at_lane` و `pressure_level` به‌صورت کامل ساخته می‌شوند. `actor` در صورت وجود شناسه ساخته می‌شود؛ `visible_player_count` و ویژگی‌های پاس ممکن است به‌علت داده‌ی غایب کمتر از تعداد اکشن باشند. `progressive` و `under_pressure` فکت‌های sparse هستند: نبود آن‌ها به معنی false است، نه مقدار گمشده.

برای جلوگیری از نشت آخرین اکشن، فاز ۳ `final_action/1` را برای عضو پایانی و `last_retained_action/1` را برای عضو بلافاصله پیش از آن صادر می‌کند. هر اکشن قبل از پایانی `in_prefix(Action, Sequence)` می‌گیرد. wrapperهای `seq_*` فاز ۵ فقط روی `in_prefix` تعریف می‌شوند؛ wrapperهای دارای look-ahead، عضو `last_retained_action` را هم کنار می‌گذارند تا اکشنی که به آخرین اکشنِ پنهان نگاه می‌کند وارد ویژگی نشود. فکت‌های پایه‌ی `in_sequence` و `action_result` همچنان برای روابط و گزارش‌ها کامل‌اند؛ محدودسازی روی wrapperهای ویژگی اعمال می‌شود.

`same_lane(A)` در [bk.pl](../src/tactical_ilp/prolog/bk.pl) با `at_lane(A,L)` و `pass_end_lane(A,L)` lane ابتدا و انتهای **خود اکشن پاس** را مقایسه می‌کند. در نتیجه تعریف `line_pass/1` به اکشن بعدی sequence وابسته نیست و synthetic dribble یا تغییر lane در اکشن بعدی نمی‌تواند نتیجه‌ی مقایسه‌ی پاس را عوض کند. نقطه‌ی مقصد با همان مرزهای `lane_of()` دسته‌بندی می‌شود؛ این predicate همچنان categorical است و نزدیک‌بودن به مرز را با تحمل متری ویژه‌ای مدل نمی‌کند.

### بازنمایی نتیجه و جداسازی label

`action_result/2` تنها fact نتیجه‌ی اکشن است. فکت‌های تکراری `successful/1` و `failed/1` تولید نمی‌شوند؛ [bk.pl](../src/tactical_ilp/prolog/bk.pl) آن‌ها را به صورت قواعد Prolog از نتیجه‌ی canonical مشتق می‌کند. این کار از دوباره‌نویسی یک حقیقت با چند predicate جلوگیری می‌کند.

برچسب دودویی `negative/non_negative` فاز ۲، کلاس نمونه‌ی Popper، نام بازیکن و metadata غیرلازم در facts پس‌زمینه صادر نمی‌شوند. هدف یادگیری باید از ورودی مثال‌ها بیاید، نه از predicate قابل‌استفاده به‌عنوان background knowledge؛ حذف label از `facts_all.pl` از نشت target به قواعد جلوگیری می‌کند. نام و marker هویت مستقل بازیکن هم صادر نمی‌شود؛ نقش و تیم برای قواعد نیازمند نقش (از جمله goalkeeper) باقی می‌ماند.

## فایل‌های خروجی

مسیرهای پیش‌فرض در [config.py](../src/tactical_ilp/config.py) هستند و اگر متغیر محیطی `TACTICAL_ILP_DATA` تنظیم شده باشد، ریشه‌ی داده از آن گرفته می‌شود. خروجی‌های فاز ۳ زیر `<DATA_DIR>/prolog/` نوشته می‌شوند:

| فایل | محتوا |
|---|---|
| `facts_match_<match_id>.pl` | فکت‌های یک مسابقه‌ی نمونه برای task 3.2؛ شناسه از فهرست مرتب مسابقات انتخاب می‌شود. |
| `facts_all.pl` | فکت‌های همه‌ی مسابقات انتخاب‌شده برای task 3.3؛ منبع مشترک پس‌زمینه‌ی مراحل بعد است. |
| `facts_manifest.json` | شناسه‌ی نمونه، version فاز ۲ و فاز ۳، نام فایل‌ها، تعداد مسابقه/sequence، مجموع factها و شمار هر predicate. |

در manifest فیلدهای جاری شامل `sample_match_id`, `phase2_sequence_cache_version`, `phase3_feature_version`, `sample_file`, `all_file`, `n_matches`, `n_sequences`, `total_facts`, `predicate_counts` هستند. `FactWriter.counts()` و manifest تعداد دفعات `add()` برای هر predicate را می‌شمارند. هنگام نوشتن، سطرهای یکسان با `set()` deduplicate می‌شوند؛ بنابراین اگر یک fact یکسان چند بار به writer داده شود، شمار manifest/comment گروه می‌تواند از تعداد سطرهای یکتای فایل بیشتر باشد. کد generator برای روابط اصلی از sequence/player/action و جفت‌های یکتا استفاده می‌کند تا تکرار غیرضروری رخ ندهد؛ آزمون‌ها نیز در موارد کلیدی تعداد واقعی factهای فایل را بررسی می‌کنند.

تعداد `involved_in/2` الزاماً با تعداد sequenceها برابر نیست؛ هر sequence می‌تواند چند بازیکن متفاوت داشته باشد. آزمون پوشش به‌جای مقایسه‌ی تعداد خام این رابطه، تعداد sequenceهای متمایز در آرگومان دوم را با تعداد sequenceهای ورودی مقایسه می‌کند.

## اجرا، ترتیب مراحل و cache

اجرای معمول:

```powershell
uv run python scripts/run_phase3.py
```

بازسازی اجباری:

```powershell
uv run python scripts/run_phase3.py --force
```

نقطه‌ی ورود [run_phase3.py](../scripts/run_phase3.py) ابتدا version cache توالی فاز ۲ را بررسی می‌کند:

1. شناسه‌های مسابقه از `spadl_all_matches.parquet` و به ترتیب صعودی خوانده می‌شوند؛ اگر این فایل موجود نباشد، از `barca_match_ids()` استفاده می‌شود.
2. اگر `--force` داده شده باشد، SPADL، فریم‌ها، sequenceها، labelها و context با force بازساخته می‌شوند.
3. در اجرای عادی اگر sequence cache فاز ۲ stale باشد، sequence، label و context بازسازی اجباری می‌شوند.
4. اگر cache معتبر تشخیص داده شود و فایل نمونه، `facts_all.pl` و manifest حاضر باشند و manifest هر دو `phase2_sequence_cache_version` و `phase3_feature_version` جاری را داشته باشد، فکت‌ها reuse می‌شوند.
5. در غیر این صورت `generate_facts()` یک بار برای نخستین match مرتب‌شده و یک بار برای کل مجموعه اجرا می‌شود؛ سپس manifest نوشته می‌شود.

`PHASE3_FEATURE_VERSION` در `prolog_facts.py` در نسخه‌ی جاری `active-popper-vocabulary-v7-prefix-only-final-action-facts` است. شناسه‌ی نسخه، schema فکت‌های فاز ۳ را مشخص می‌کند و برای کنترل سازگاری در manifest ثبت می‌شود. marker فاز ۲ نیز در manifest ذخیره می‌شود، تا خروجی فکت‌ها با قرارداد sequence سازگار بماند. cache فکت‌ها بر مبنای حضور فایل‌ها و برابری versionهای ثبت‌شده ارزیابی می‌شود؛ محتوای parquet/`.pl` با hash کامل مقایسه نمی‌شود. تغییر معنی‌دار در feature/fact schema باید با افزایش version فاز ۳ همراه شود. برای تولید این schema، `run_phase3.py` را اجرا کنید تا facts و manifest بازتولید شوند؛ اجرای `--force` همه‌ی ورودی‌های مشتق‌شده را هم بازسازی می‌کند.

برآوردگر هزینه‌ی فاز ۴ در [`cost_report.py`](../src/tactical_ilp/estimation/cost_report.py) شمار کل facts را از manifest می‌خواند، مشروط بر آنکه version فاز ۳ با version جاری مطابق باشد؛ در صورت نبود manifest معتبر، از مقدار پشتیبان برآورد استفاده می‌شود. این وابستگی، denominator برآورد هزینه را با مجموعهٔ facts ثبت‌شده در manifest هم‌خوان نگه می‌دارد.

اسکریپت نمونه و فایل کامل را با SWI-Prolog consult می‌کند، اگر `swipl` روی `PATH` باشد. نبود SWI-Prolog فقط load check را skip می‌کند. خروجی مشکل consult گزارش می‌شود؛ خود runner نتیجه‌ی Boolean consult را به exit code تبدیل نمی‌کند. اجرای script از `run_logged(3, main)` عبور می‌کند و stdout/stderr را هم‌زمان در `<DATA_DIR>/reports/3_run_<UTC timestamp>.log` ثبت می‌کند.

## نقش فاز ۳ در زنجیره‌ی Prolog

فایل `facts_all.pl` داده‌های ثابت مرحله‌ی ۳ را در اختیار [قواعد بک‌گراند](../src/tactical_ilp/prolog/bk.pl) می‌گذارد. فاز ۴ `facts_tactical.pl` و قواعد/محاسبات تاکتیکی خود را اضافه می‌کند. سپس bias و ورودی Popper predicateهای مجاز را محدود می‌کنند. پس:

- فکت‌های فاز ۳ مشاهده‌ی محاسبه‌شده از داده‌اند؛
- clauseهای `bk.pl` تعریف رابطه/استنتاج روی آن مشاهده‌ها هستند؛
- مثال‌های هدف در facts فاز ۳ جاسازی نمی‌شوند؛
- وجود یک predicate در `.pl` لزوماً به معنی مجازبودن آن برای یادگیری نیست؛ bias/واژگان مرحله‌ی بعد مرز search را تعیین می‌کند.

برای مثال، Prolog می‌تواند از `sequence/1`, `in_sequence/2` و `under_pressure/1` تعداد اکشن‌های تحت فشار را بشمارد، اما فایل facts به‌تنهایی تعریف نمی‌کند که فشار «باعث شکست build-up شده است»؛ این رابطه باید از قواعد/مثال‌ها و فرآیند یادگیری حاصل شود.

## آزمون‌ها و کنترل کیفیت

آزمون‌های فاز در [test_phase3_features.py](../tests/test_phase3_features.py) و fixtureهای مشترک در [conftest.py](../tests/conftest.py) هستند:

- مرزهای lane و third و پوشش کل عرض زمین؛
- تقدم شمارش freeze-frame بر پرچم StatsBomb و fallback وقتی شمار تهی است؛
- مرزهای bucket فشار، جهت و طول پاس؛
- پیشروی در سه ناحیه، رد حرکت عرضی/عقب‌رو و رفتار مختصات NaN؛
- consult شدن فایل نمونه و فایل کامل توسط SWI-Prolog و نبود Warning/Error؛
- ASCII بودن فایل نمونه؛
- تطابق پوشش sequence و شناسه‌های متمایز `involved_in` با خروجی فاز ۲؛
- هم‌ترازی تعداد predicateهای پایه‌ی action-level با `in_sequence/2`، و sparse بودن `visible_player_count/2`؛
- عدم صدور label هدف، شناسه/نام بازیکن، aliasهای موفق/ناموفق و روابط پاس غیرفعال؛
- حضور `pass_end_lane/2` برای اکشن‌های پاس با مقصد معلوم؛
- وجود role/team factها؛
- پاسخ‌دادن query واقعی Prolog درباره‌ی تعداد اکشن‌های تحت فشار.

بخش‌هایی از آزمون‌ها به parquetهای ساخته‌شده‌ی فازهای ۱–۳ وابسته‌اند؛ اگر داده‌ی پردازش‌شده وجود نداشته باشد fixtureها توضیح می‌دهند که ابتدا runner مربوط اجرا شود. آزمون syntax نیز تنها وقتی SWI-Prolog در `PATH` باشد consult واقعی انجام می‌دهد.

## نقشه‌ی فایل‌های مسئول

| مسیر | مسئولیت در فاز ۳ |
|---|---|
| [`scripts/run_phase3.py`](../scripts/run_phase3.py) | ترتیب اجرا، تشخیص cache، تولید sample/full facts، consult و manifest. |
| [`src/tactical_ilp/features/geometry.py`](../src/tactical_ilp/features/geometry.py) | lane/third، فشار، جهت/طول پاس، پیشروی و helperهای هندسی پایه. |
| [`src/tactical_ilp/prolog/prolog_facts.py`](../src/tactical_ilp/prolog/prolog_facts.py) | واژگان fact فعال، شناسه‌های atom، writer، join ورودی‌ها و serialization. |
| [`src/tactical_ilp/data/freeze_frames.py`](../src/tactical_ilp/data/freeze_frames.py) | اتصال frame به event و شمارش بازیکنان/حریفان در شعاع فشار برای ورودی featureها. |
| [`src/tactical_ilp/config.py`](../src/tactical_ilp/config.py) | مسیرها، ابعاد/مرزهای pitch و thresholdهای فشار. |
| [`src/tactical_ilp/prolog/bk.pl`](../src/tactical_ilp/prolog/bk.pl) | قواعد زنده‌ی Prolog؛ از جمله مشتق‌سازی موفق/ناموفق از `action_result/2`. |
| [`src/tactical_ilp/estimation/cost_report.py`](../src/tactical_ilp/estimation/cost_report.py) | baseline شمار فکت‌ها را از manifest فاز ۳ با version جاری می‌گیرد. |
| [`tests/test_phase3_features.py`](../tests/test_phase3_features.py) | آزمون قواعد feature و fact output. |


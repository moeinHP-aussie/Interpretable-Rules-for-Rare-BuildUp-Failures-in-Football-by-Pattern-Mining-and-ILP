# فاز ۲ — ساخت توالی‌ها، برچسب‌گذاری پیامد، زیرمجموعه‌های بازیکنان و نمونه‌گیری

## هدف و جایگاه فاز

فاز ۲ جدول اکشن‌های آماده‌شده در فاز ۱ را به واحدهای تحلیل تاکتیکی تبدیل می‌کند. چهار وظیفه‌ی اصلی آن عبارت‌اند از:

1. استخراج توالی‌های build-up بارسلونا و ثبت اکشن‌های عضو هر توالی؛
2. برچسب‌گذاری پیامد توالی با تعریف عملیاتی سه‌ناحیه‌ای؛
3. استخراج پیوندهای پاس بین بازیکنان و زنجیره‌های سه‌نفره؛
4. ساخت مجموعه‌ی نمونه‌های ILP با منفی‌های همسان‌شده از نظر زمینه.

نقطه‌ی ورود [`scripts/run_phase2.py`](../scripts/run_phase2.py) است. منطق وظایف در این ماژول‌ها پیاده‌سازی شده است:

- [`src/tactical_ilp/sequences/sequence_builder.py`](../src/tactical_ilp/sequences/sequence_builder.py)
- [`src/tactical_ilp/sequences/outcome_labeler.py`](../src/tactical_ilp/sequences/outcome_labeler.py)
- [`src/tactical_ilp/sequences/player_subsets.py`](../src/tactical_ilp/sequences/player_subsets.py)
- [`src/tactical_ilp/sequences/negative_sampling.py`](../src/tactical_ilp/sequences/negative_sampling.py)
- ثابت‌های مرزی و نمونه‌گیری: [`src/tactical_ilp/config.py`](../src/tactical_ilp/config.py)
- تقسیم‌بندی زمین و شدت فشار: [`src/tactical_ilp/features/geometry.py`](../src/tactical_ilp/features/geometry.py)

اجرا از ریشه‌ی بسته:

```powershell
uv run python scripts/run_phase2.py
uv run python scripts/run_phase2.py --force
```

بدون `--force` خروجی‌های cacheشده در صورت معتبر تشخیص داده‌شدن استفاده می‌شوند. گزینه‌ی `--force` همه‌ی محصولات فاز ۲ را بازسازی می‌کند. فاز ۲ به داده‌ی ادغام‌شده‌ی فاز ۱ (`spadl_with_360_all_matches.parquet`) وابسته است؛ توالی‌ها، برچسب‌ها و مثال‌های ILP را تولید می‌کند، اما factهای Prolog یا قوانین Popper را نمی‌سازد.

## ترتیب اجرا و وابستگی داده‌ای

Runner ابتدا مقدار `rebuild` را از `--force` یا کهنه‌بودن cache توالی تعیین می‌کند. سپس مراحل را با همین flag به‌ترتیب اجرا می‌کند: segmentation، برچسب‌گذاری، زیرمجموعه‌ی بازیکنان، ساخت context، و نمونه‌گیری. این ترتیب عمدی است: برچسب و membership باید پیش از context و نمونه‌های ILP وجود داشته باشند؛ زیرمجموعه‌های بازیکن نیز از همان عضویت توالی استفاده می‌کنند.

در پایان هر مرحله runner شمارها و توزیع‌ها را چاپ می‌کند و در انتها اجرای فاز ۳ را پیشنهاد می‌دهد. `run_logged(2, main)` خروجی کنسول و خطاها را در فایل زمان‌دار `data/reports/2_run_<UTC timestamp>.log` نیز ثبت می‌کند؛ پیاده‌سازی ثبت اجرا در [`scripts/run_logging.py`](../scripts/run_logging.py) است.

## تعریف ساختاری build-up در وظیفه‌ی ۲.۱

### واحد پردازش و ترتیب اکشن‌ها

`build_sequences_for_match()` ورودی یک مسابقه را می‌گیرد و اکشن‌ها را بر اساس `period_id`, `time_seconds`, `action_id` مرتب می‌کند. سپس ستون `row` را با شماره‌ی ترتیبی صفرمبنا در جدول اکشن همان مسابقه می‌سازد. داده بر اساس زوج `(period_id, possession)` گروه‌بندی می‌شود؛ بنابراین یک توالی از یک دوره یا یک شناسه‌ی مالکیت عبور نمی‌کند.

گروه فقط وقتی بررسی می‌شود که `possession_team_id` نخستین ردیف آن برابر `TEAM_ID=217` باشد. سپس اکشن‌های `is_barca=True` جدا می‌شوند و فقط همین اکشن‌ها در توالی قرار می‌گیرند. اکشن‌های حریف در همان گروه برای خود segmentation عضو نیستند؛ مرز possession در داده‌ی فاز ۱ و شناسه‌های StatsBomb آن اعمال می‌شود.

### شرایط آغاز

اولین اکشن بارسلونا در زنجیره باید هر دو شرط را داشته باشد:

- نقطه‌ی شروع آن در نیمه‌ی خودی باشد: `start_x <= BUILD_UP_START_MAX_X = 52.5` متر؛
- نوع اکشن در `VALID_START_TYPES` باشد.

به‌علاوه، توالی‌های آغازشده با `throw_in`, `corner_short`, `corner_crossed`,
`freekick_short` یا `freekick_crossed` از مجموعهٔ build-up حذف می‌شوند؛ بنابراین
در برچسب‌گذاری، نمونه‌گیری، آموزش و ارزیابی وارد نمی‌شوند. `goalkick` فقط وقتی
به‌عنوان شروع حفظ می‌شود که طول اقلیدسی آن حداکثر ۲۵ متر باشد (پاس کوتاه/متوسط
در شروع بازی‌سازی دروازه‌بان). شروع‌های بلند دروازه حذف می‌شوند. این فیلتر فقط
به اکشن آغازین مربوط است و پاس بلندِ جریان بازی در ادامهٔ یک توالی واجدشرایط
همچنان می‌تواند ویژگی تاکتیکی `long_pass` را داشته باشد.

انواع مجاز عبارت‌اند از `pass`, `dribble`, `goalkick`, `freekick_short`, `throw_in`, `keeper_pick_up`, `keeper_claim`, `keeper_save`, `interception`, `tackle`, `clearance`, `bad_touch`, `take_on`, `cross`, `freekick_crossed`, `corner_short`, `corner_crossed`, `foul`, `shot`, `shot_freekick`, `shot_penalty`.

این شرط یعنی مبنا محل آغاز نخستین اکشن بارسلونا در مالکیت است؛ صرف حضور بعدی توپ در نیمه‌ی خودی یک توالی جدید نمی‌سازد. آغاز در `x=52.5` پذیرفته می‌شود چون مرز نیمه inclusive است.

### نواحی ورود کنترل‌شده

تابع `_entry_regions(x, y)` مجموعه‌ی نواحی مجاز را برمی‌گرداند؛ یک نقطه می‌تواند در بیش از یک ناحیه قرار گیرد:

| شناسه‌ی ناحیه | شرط مکانی در متر |
|---|---|
| `deep_quarter` | `x >= 78`، در تمام عرض زمین |
| `zone_14` | `70 <= x <= 87.5` و `27.2 <= y <= 40.8` |
| `left_deep_half_space` | `x >= 70` و `40.8 <= y <= 54.4` |
| `right_deep_half_space` | `x >= 70` و `13.6 <= y <= 27.2` |

مرزها از `config.py` می‌آیند. در قرارداد مختصات پروژه، `y=0` سمت راست از دید حمله‌ی بارسلونا است؛ عنوان‌های left/right مطابق همین چارچوب هستند. اگر هر یک از مختصات مفقود باشد، هیچ regionای برگردانده نمی‌شود.

### اثبات کنترل و پذیرش در ناحیه

تابع `_controlled_entry(own, index, action)` بین کنترل و صرف رسیدن توپ به نقطه فرق می‌گذارد:

- اگر نقطه‌ی شروع اکشن در ناحیه‌ی مجاز باشد و اکشن قبلی بارسلونا **پاس** نباشد، ورود کنترل‌شده پذیرفته می‌شود.
- اگر اکشن قبلی بارسلونا پاس بوده، شروع اکشن جاری به‌تنهایی کافی نیست. پاس قبلی باید `success` باشد و نقطه‌ی پایان همان پاس نیز در حداقل یکی از همان regionهایی باشد که نقطه‌ی شروع فعلی در آن قرار دارد. بنابراین یک endpoint پاس به‌تنهایی کنترل توپ تلقی نمی‌شود؛ اکشن بعدی بارسلونا در همان ناحیه مدرک دریافت است.
- اگر اکشن جاری پاس باشد و نقطه‌ی شروعش region مجاز نباشد، صرف قرارداشتن مقصد پاس در آن ناحیه ورود نیست؛ دریافت بعدی باید کنترل را اثبات کند.
- اگر اکشن `dribble`, `take_on` یا `carry` باشد، در صورتی که اکشن مقصد نباشد (`dribble` و `take_on`) نتیجه باید `success` باشد. برای `carry` این فیلتر نتیجه اعمال نمی‌شود. نقطه‌ی پایان اکشن باید در region مجاز قرار گیرد.
- اکشن‌هایی غیر از این carryهای کنترل‌شده، با مقصد داخل zone به‌تنهایی ورود را اثبات نمی‌کنند.

برای پاس، اشتراک regionها مهم است؛ مثلاً موفقیت پاس به یک region و شروع اکشن بعدی در region متفاوت، حتی اگر هر دو zone دیگری را نیز لمس کنند، فقط در صورت اشتراک واقعی مجموعه پذیرفته می‌شود.

### شرایط پایان و اولویت آن‌ها

اکشن‌ها به‌ترتیب پیمایش می‌شوند و ابتدا به فهرست اعضا افزوده می‌شوند. سپس اولین شرط زیر که برقرار شود توالی را می‌بندد:

1. نوع اکشن یکی از `shot`, `shot_freekick`, `shot_penalty` باشد: `end_reason="shot"`.
2. نتیجه `fail` یا `owngoal` باشد: `end_reason="failed_action"`.
3. ورود کنترل‌شده به یکی از نواحی بالا تأیید شود: `end_reason="attacking_zone_entry"`.
4. اگر هیچ‌یک برقرار نباشد و طول به ۲۵ اکشن برسد: `end_reason="length_cap"`.
5. اگر پیمایش اکشن‌های مالکیت تمام شود: `end_reason="possession_end"`.

ترتیب بالا مهم است: شوت پیش از بررسی `fail` بررسی می‌شود؛ بنابراین شوت پایانی با نتیجه‌ی ناموفق هم از دید segmentation با `end_reason="shot"` ثبت می‌شود. سه پایان طبیعی (شوت، شکست اکشن، ورود) بر سقف فنی طول اولویت دارند.

پس از پایان، فقط توالی‌های دارای ۲ تا ۲۵ اکشن نگه داشته می‌شوند (`MIN_SEQUENCE_LENGTH=2`, `MAX_SEQUENCE_LENGTH=25`). اکشن تک‌افتاده حذف می‌شود. سقف ۲۵ از واحدهای بسیار بلند و غیرمتمرکز جلوگیری می‌کند و در عین حال سقف محاسباتی ثابتی به مراحل بعد می‌دهد.

`end_reason` علت پایان segmentation است و عموماً با label پیامد یکی نیست؛ استثنا دو نوع تکمیل موفق build-up هستند: `shot` و `attacking_zone_entry`. در labeler این دو علت، معیارهای ناحیه‌ای turnover را از تبدیل sequence به `negative` محافظت می‌کنند. `possession_end` و `length_cap` به‌تنهایی نتیجه‌ی outcome را تعیین نمی‌کنند و با رخدادهای پس از مرز ارزیابی می‌شوند.

### شناسه و آمار توالی

برای هر مسابقه، شمارنده‌ی `sequence_id` از صفر شروع می‌شود و شناسه به قالب `s_<match_id>_<index:04d>` ساخته می‌شود. چون شماره‌گذاری برای هر مسابقه جداگانه است، کلید آن به‌تنهایی معمولاً یکتا است، ولی ستون `match_id` همچنان بخشی از هویت امن داده محسوب می‌شود.

هر ردیف `sequences.parquet` این فیلدها را دارد:

- هویت و segmentation: `sequence_id`, `match_id`, `period_id`, `possession`, `n_actions`, `end_reason`؛
- زمان: `start_time`, `end_time`, `duration` (اختلاف زمان اکشن آخر و اول؛ ثانیه‌ی بازی، نه لزوماً زمان خالص کنترل توپ)؛
- مختصات: `start_x`, `start_y` از نخستین اکشن، `end_x`, `end_y` از endpoint آخرین اکشن؛
- پیشروی: `max_x` از همه‌ی `start_x`های اعضا به‌علاوه‌ی `end_x` اکشن‌های carry کنترل‌شده؛
- اتصال به جدول اکشن: `last_action_id`, `last_action_row`؛ `last_action_row` همان شماره‌ی ترتیبی در جدول اکشن کل مسابقه است و برای جست‌وجوی پس از توالی در برچسب‌گذاری مصرف می‌شود؛
- بازیکنان/پاس: `n_players`, `n_passes`, `involved_players` (شناسه‌های بازیکن یکتا، مرتب‌شده).

`max_x` محاسبه‌ی پیشروی کنترل‌شده‌ی کامل نیست: شروع همه‌ی اکشن‌ها در آن شمرده می‌شود، ولی پایان فقط برای اکشن‌های `dribble`, `take_on`, `carry` که carry کنترل‌شده محسوب می‌شوند افزوده می‌شود. `involved_players` لیست integer بازیکنان بدون تکرار است؛ `n_players` تعداد یکتای غیرمفقود آن‌هاست.

### جدول membership اکشن‌ها

هر ردیف `sequence_actions.parquet` اتصال یک اکشن به یک sequence را ثبت می‌کند: `sequence_id`, `match_id`, `action_id`, `order_in_sequence`. `order_in_sequence` از صفر آغاز می‌شود و برای بازیابی ترتیب عضوها استفاده می‌شود. جدول به‌جای کپی‌کردن همه‌ی ستون‌های action، کلیدها را نگه می‌دارد؛ مصرف‌کننده با زوج `(match_id, action_id)` به جدول merged فاز ۱ می‌پیوندد. تست فاز ۲ شمار اعضای هر sequence را با `n_actions` مقایسه می‌کند.

## وظیفه‌ی ۲.۲ — تعریف و محاسبه‌ی پیامد

### ورودی‌ها و ترتیب پس از توالی

`label_sequences()` از sequenceهای مرحله‌ی قبل و جدول merged فاز ۱ استفاده می‌کند. جدول اکشن برای هر مسابقه برحسب `match_id`, `period_id`, `time_seconds`, `action_id` مرتب می‌شود و `row` به‌صورت شمارنده‌ی داخل مسابقه افزوده می‌شود. برای هر sequence، `_after_sequence()` اکشن‌های همان دوره را با `row > last_action_row` می‌گیرد. جست‌وجوی transition به possession محدود نمی‌شود، چون هدف بررسی اتفاق بلافاصله بعد از پایان sequence، از جمله تغییر مالکیت، است.

`time_seconds` در نسخه‌ی جاری از timestamp دوره‌ای StatsBomb با دقت میلی‌ثانیه می‌آید، در همان مبنای زمانی SPADL/socceraction. بازپس‌گیری پنج‌ثانیه‌ای با همین ساعت سنجیده می‌شود؛ ترتیب ردیف‌ها نیز برای tieها حفظ می‌شود.

نتیجه دودویی است: `negative` برای از دست‌دادن توپ در نیمهٔ خودی بدون بازپس‌گیری در پنج ثانیه و بدون foul حریف؛ در غیر این صورت `non_negative`. این برچسب به معنی موفقیت قطعی فوتبال نیست.

در نام‌گذاری نمونه‌های ILP، `example_class="pos"` برای sequence با label=`negative` و `example_class="neg"` برای label=`non_negative` است؛ بنابراین `pos/neg` در فایل نمونه به کلاس هدف یادگیری اشاره می‌کند، نه خوب/بد بودن فوتبال.

### تشخیص اینکه مالکیت از دست رفته است

ابتدا آخرین اکشن sequence از جدول اکشن بازی پیدا می‌شود. اکشن پایانی ناموفق بارسلونا (`result_name` برابر `fail` یا `owngoal`) مدرک مستقیم loss است. در غیر این صورت فقط وقتی possession lost فرض می‌شود که **اولین** اکشن بعد از sequence متعلق به حریف باشد. اگر اولین اکشن بعدی هنوز از بارسلونا باشد، برای این مرز loss ثبت نمی‌شود، حتی اگر رویدادهای بعدی حریف باشند.

اگر اولین اکشن حریف `foul` باشد، رویداد توقف بازی تلقی می‌شود و بر loss فرض‌شده اولویت دارد؛ `lost_possession=False` و turnover مختصات/time نامعین می‌شود. foul حریف به‌تنهایی هیچ‌گاه منفی نیست. پیامدهای `shot` و `attacking_zone_entry` دیگر label را محافظت یا تغییر نمی‌دهند.

نقطه و زمان turnover از منبع متفاوتی می‌آیند:

- اگر اکشن پایانی بارسلونا fail/owngoal است: `turnover_x=end_x` همان اکشن و `turnover_time=time_seconds` همان اکشن است؛
- اگر انتقال به حریف با اولین اکشن بعدی شناسایی شود: `turnover_x=start_x` و زمان از همان اکشن نخست حریف می‌آید؛
- در نبود loss یا توقف ناشی از foul: `turnover_x/time` مقدار نامعین و `turnover_row=-1` می‌گیرند.

ناحیه‌ی turnover با محور `x` بارسلونا تعیین می‌شود:

| ناحیه | محدوده‌ی turnover |
|---|---|
| `defensive_third` | `x <= 35.0` |
| `own_half_midfield` | `35.0 < x <= 52.5` |
| `advanced_opponent_middle` | `52.5 < x < 70.0` |
| `outside_target_zones` | possession lost ولی بیرون سه محدوده‌ی بالا |
| `none` | turnover تأیید نشده |

مرز ۷۰ متر وارد ناحیه‌ی advanced loss نمی‌شود؛ آن مرز، ابتدای ناحیه‌ی ورود به بخش تهاجمی در قرارداد توالی است.

### بازپس‌گیری و پیامدهای توصیفی

تابع `_regained_within_window()` رویدادهای همان دوره را از ردیف turnover تا پنج ثانیه بعد بررسی می‌کند. فقط یک اکشن موفق و کنترل‌کننده‌ی مالکیت تیم منتخب موجب `regained_within_5s=True` می‌شود: پاس/ارسال، دریبل، take-on، carry، شروع مجدد کنترل‌شده یا مهار/دریافت مطمئن دروازه‌بان. تکل یا interception موفق به‌تنهایی کافی نیست.

`opponent_shot_after` و `opponent_entered_def_third` از اکشن‌های حریف در پنجره‌ی توصیفی ۱۲ثانیه‌ای پس از turnover ساخته می‌شوند. این دو ستون برای سنجش اینکه الگوها چند درصد به شوت یا ورود حریف ختم شده‌اند نگه داشته می‌شوند، اما در تعریف label هیچ نقشی ندارند. پیشروی مستقیم و تعداد اکشن‌های موفق حریف دیگر شرط label نیستند.

### تعریف فعلی برچسب منفی

یک سکانس `negative` است اگر توپ در مختصات تیم تحلیل‌شده در نیمه‌ی خودی از دست برود (`turnover_x <= OWN_HALF_MAX_X`, یعنی `x <= 52.5`)، تیم ظرف ۵ ثانیه کنترل توپ را دوباره به‌دست نیاورد، و نخستین اکشن حریف پس از مرز سکانس foul نباشد. این تعریف در تمام نواحی نیمه‌ی خودی یکسان است؛ پیشروی، شوت یا تعداد اکشن‌های موفق حریف در تصمیم برچسب دخالت ندارند.

تمام سکانس‌هایی که این تعریف را برآورده نمی‌کنند `non_negative` هستند. بنابراین foul حریف، بازپس‌گیری در پنجره، از دست‌دادن توپ خارج نیمه‌ی خودی، و نبودِ شواهد از دست‌رفتن توپ مثبت کاذب تولید نمی‌کنند. در این تعریف label دودویی است و `inconclusive` تولید نمی‌شود. «non_negative» یعنی شرط منفی عملیاتی برقرار نشده، نه اینکه build-up قطعاً موفق بوده است.

`opponent_shot_after` و `opponent_entered_def_third` فقط متغیرهای توصیفی‌اند و در برچسب دخالت ندارند. پارامترهای برچسب در `config.py`: `OWN_HALF_MAX_X=52.5` و `MIDFIELD_REGAIN_WINDOW_S=5`.

کدهای `negative_reason` عبارت‌اند از `own_half_loss_not_regained_within_5s`, `loss_regained_within_5s`, `loss_outside_own_half`, `loss_location_unknown`, `opponent_foul_stoppage` و `no_possession_loss`.

### ستون‌های فایل برچسب‌خورده

`labeled_sequences.parquet` همان ستون‌های sequence را با داده‌های برچسب ادغام می‌کند. ستون‌های افزوده:

- `lost_possession`, `turnover_x`, `turnover_time`, `turnover_row`, `turnover_zone`؛
- `final_action_failed`, `build_up_completed`؛
- `opponent_shot_after`, `opponent_entered_def_third` (descriptive only)؛
- `regained_within_5s`؛
- `negative_reason`, `label`؛
- `zone`: منطقه‌ی آغاز توالی به شکل `third|lane`، مثل `defensive_third|left_wing`.

### تعریف قابل‌مصرف ماشینی

`DEFINITION` در `outcome_labeler.py` پارامترهای برچسب را به‌صورت دیکشنری expose می‌کند. جزئیات بالا از منطق فعال `_label_one()` و `_regained_within_window()` استخراج شده است.

## وظیفه‌ی ۲.۳ — پیوندهای دونفره و سه‌نفره

`build_subsets()` membership را به action table فاز ۱ وصل می‌کند و عضوها را بر حسب sequence و `order_in_sequence` مرتب می‌کند. نوع‌های pass-like شامل `pass`, `cross`, `freekick_short`, `goalkick`, `throw_in`, `corner_short` است.

### پیوند دونفره

برای هر دو اکشن متوالی در sequence، یک `passes_to(P1,P2,Seq)` ثبت می‌شود اگر اکشن اول از نوع pass-like، نتیجه‌اش `success`، شناسه‌ی بازیکن هر دو اکشن موجود، و دو بازیکن متفاوت باشند. اکشن دومِ متوالی نقش receiver را دارد. داده‌ی pair شامل `sequence_id`, `match_id`, `order_in_sequence` اکشن پاس‌دهنده، `passer_id`, `receiver_id`, `pass_type` و مختصات شروع/پایان پاس است. inverse رابطه‌ی `receives_from` در این مرحله به‌عنوان جدول جدا ذخیره نمی‌شود.

### زنجیره‌ی سه‌نفره

زنجیره‌ی `p1 -> p2 -> p3` از دو pair مرتب‌شده در یک sequence ساخته می‌شود، وقتی receiver پیوند اول با passer پیوند دوم یکی باشد و مجموعه‌ی سه شناسه دقیقاً سه بازیکن متمایز داشته باشد. هر سطر triple خروجی `sequence_id`, `match_id`, `p1`, `p2`, `p3`, `order_in_sequence` پیوند نخست را نگه می‌دارد. پیوندها بر حسب order مرتب می‌شوند؛ الگوریتم پس از ساخت زوج‌ها میان فهرست زوج‌های واجد شرایط پیمایش می‌کند.

### خلاصه‌ی در سطح توالی و cacheها

`sequences_with_subsets.parquet` sequenceها را با `player_pairs`, `n_pairs`, `n_triples` غنی می‌کند. `player_pairs` تعداد **زوج‌های یکتای مرتب‌شده** در یک توالی است، نه تعداد کل تکرار رخداد پاس؛ برای ذخیره در Parquet هر زوج به رشته‌ی `passer>receiver` تبدیل می‌شود. `n_triples` تعداد سطرهای زنجیره‌ی سه‌نفره است. در هر دو جدول زوج/سه‌تایی، تکرارهای رخداد ممکن است جداگانه باقی بمانند.

اگر هیچ pair ساخته نشود، پیاده‌سازی `pairs` را از فهرست خالی به DataFrame خالی می‌سازد؛ schema جدول را باید از مسیر موفقی که ستون‌ها تولید شده‌اند یا مصرف‌کننده‌ی بالادست دانست، نه از وجود اجباری ستون در فایل صفرردیفی.

هیچ ترکیب همه‌به‌همه‌ای از ۱۱ بازیکن تولید نمی‌شود؛ فقط interaction مشاهده‌شده از توالی اکشن‌های متوالی وارد می‌شود. حمایت/پوشش مکانی از 360 در فاز ۴ تعریف می‌شود و بخشی از extraction این جدول نیست.

فایل‌ها: `sequence_pairs.parquet`, `sequence_triples.parquet`, `sequences_with_subsets.parquet` زیر `data/processed/`.

## وظیفه‌ی ۲.۴ — context و نمونه‌گیری منفی

### ساخت stratum

`build_context()` برای اکشن نخست هر sequence، از `sequence_actions` ردیف `order_in_sequence == 0` را انتخاب می‌کند و اطلاعات آن را از merged table می‌گیرد. سه متغیر تطبیق ساخته می‌شود:

1. **`start_zone`**: `pitch_zone(start_x,start_y)`، یعنی یک‌سوم افقی (`defensive_third`, `middle_third`, `final_third`) به‌علاوه‌ی یکی از پنج lane (`right_wing`, `right_half_space`, `central_channel`, `left_half_space`, `left_wing`)؛ در کل ۱۵ ترکیب مکانی ممکن.
2. **`pressure_bucket`**: از `ff_opponents_3m`, `ff_opponents_5m`, و fallback پرچم StatsBomb ساخته می‌شود. دو حریف یا بیشتر در ۳ متر = `heavy_pressure`; حداقل یک حریف در ۳ متر یا دو حریف در ۵ متر = `light_pressure`; در نبود frame اگر `under_pressure=True` نیز `light_pressure`; در بقیه‌ی موارد `no_pressure`. کمبود داده‌ی 360 صفر قطعی تلقی نمی‌شود و پرچم StatsBomb برای bucket fallback به‌کار می‌رود.
3. **`length_bucket`**: ۲–۳ اکشن `short`، ۴–۶ اکشن `medium`، ۷ اکشن یا بیشتر `long`.

رشته‌ی `stratum` با چسباندن سه مقدار و جداکننده‌ی `#` ساخته می‌شود: `start_zone#pressure_bucket#length_bucket`. `zone` در داده‌ی برچسب‌خورده با `start_zone` از نظر تعریف زمین هم‌معناست، اما در context از فیلد جداگانه برای تطبیق استفاده می‌شود.

فایل `sequence_context.parquet` تمام ستون‌های labeled sequence را نگه می‌دارد و `pressure_bucket`, `start_zone`, `first_player_id`, `first_action_type`, `length_bucket`, `stratum` را اضافه می‌کند. `first_player_id` و `first_action_type` برای شرح زمینه نگه‌داری می‌شوند، ولی جزء stratum نیستند؛ تطبیق هویت بازیکن یا نوع اکشن انجام نمی‌شود.

### تعریف کلاس‌ها و نسبت نمونه

نمونه‌های مثبت از همه‌ی sequenceهایی می‌آیند که `label == "negative"` دارند. pool کنترل از sequenceهای `label == "non_negative"` انتخاب می‌شود. هدف `NEG_SAMPLES_PER_POSITIVE=2` است؛ در صورت وجود کنترل کافی، به‌ازای هر مثبت دو کنترل بدون تکرار انتخاب می‌شود. این نسبت کلاس نمونه را تقریباً به یک‌سوم positive می‌رساند؛ اما کمبود pool می‌تواند تعداد واقعی کنترل‌ها را از هدف کمتر کند.

### نردبان تطبیق و بدون‌جایگذاری

برای هر positive، ابتدا دو منفی لازم درخواست می‌شود و این نردبان به‌ترتیب بررسی می‌شود:

| `match_level` | معیار candidate |
|---:|---|
| 3 | هر سه مقدار zone + pressure + length برابر (stratum دقیق) |
| 2 | `start_zone` و `length_bucket` برابر |
| 1 | فقط `start_zone` برابر |
| 0 | تمام pool `non_negative` بدون تطبیق زمینه؛ سهم آن حداکثر ۵٪ از منفی‌های نهایی است |

اگر سطح فعلی candidate کافی نداشته باشد، نمونه‌های موجود برداشته می‌شوند و کمبود به سطح بعد می‌رود. ابتدا تمام سطح‌های ۳ تا ۱ بدون جایگذاری انتخاب می‌شوند؛ سپس تابع `_global_fallback_budget()` حداکثر fallbackهای level 0 را از تعداد کنترل‌های هم‌زمینه محاسبه می‌کند. سقف با `MAX_GLOBAL_MATCH_SHARE=0.05` در `config.py` تعیین می‌شود: بودجه برابر `floor(0.05 / 0.95 × تعداد کنترل‌های level 1–3)` است تا سهم نهایی level 0 از کل منفی‌ها از ۵٪ بیشتر نشود. اگر این بودجه یا pool مناسب کافی نباشد، نسبت واقعی پایین‌تر از ۲:۱ می‌ماند و candidate سراسری اضافه نمی‌شود. `used` در تمام positives مشترک است؛ بنابراین یک sequence منفی دوباره انتخاب نمی‌شود. انتخاب با `numpy.random.default_rng(NEG_SAMPLING_RANDOM_SEED)` و seed برابر `202021` قطعی و تکرارپذیر است؛ ترتیب positives از DataFrame ورودی در ترتیب sampling اثر دارد.

برای هر negative ذخیره می‌شود: `sequence_id`, `example_class="neg"`, `match_level`, `matched_to` (شناسه‌ی positive هدفی که نمونه ابتدا برایش انتخاب شده). Positiveهای اصلی با `example_class="pos"`, `match_level=3`, `matched_to=خود sequence` ثبت می‌شوند؛ مقدار ۳ برای positiveها به معنی انجام تطبیق کنترل نیست، بلکه مقدار sentinel ثابت پیاده‌سازی است. سپس تمام ستون‌های context با sequence id به جدول مثال‌ها join می‌شوند. خروجی `ilp_examples.parquet` باید به‌ازای هر sequence حداکثر یک سطر داشته باشد.

### شاخص شباهت توزیع zone

`zone_distribution_overlap(pos, neg)` توزیع نسبی zone آغاز را در هر کلاس محاسبه و برای zoneهای اجتماع، کمینه‌ی دو نسبت را جمع می‌کند. مقدار بین صفر و یک است: یک یعنی توزیع‌ها یکسان، صفر یعنی هیچ zone مشترکی از نظر جرم توزیع ندارند. runner همپوشانی را همراه با سهم exact stratum و global fallback چاپ می‌کند؛ آستانه‌ی تشخیصی zone overlap حداقل `0.40` است، و سهم level 0 حداکثر `MAX_GLOBAL_MATCH_SHARE=0.05` است. اگر نمونه‌های هم‌زمینه کافی نباشند، نسبت واقعی از ۲:۱ کمتر می‌شود تا سقف پنج‌درصد رعایت شود.

## خروجی‌های فاز ۲

همه‌ی خروجی‌های داده زیر `data/processed/` ذخیره می‌شوند:

| فایل | سطح/کاربرد |
|---|---|
| `sequences.parquet` | یک ردیف به‌ازای هر build-up پذیرفته‌شده |
| `sequence_actions.parquet` | اتصال هر اکشن به sequence و ترتیبش |
| `sequence_cache_version.txt` | نسخه‌ی segmentation/label protocol و نسخه‌ی فاز ۱ |
| `labeled_sequences.parquet` | sequenceها با label، turnover و شواهد انتقال |
| `label_cache_version.txt` | شناسه‌ی تعریف label مورد استفاده |
| `sequence_pairs.parquet` | interaction دو بازیکن از پاس موفق و اکشن بعدی |
| `sequence_triples.parquet` | زنجیره‌های سه‌نفره‌ی پیوندی |
| `sequences_with_subsets.parquet` | خلاصه‌ی sequence با pair/triple |
| `sequence_context.parquet` | sequence برچسب‌خورده با bucketهای تطبیق |
| `ilp_examples.parquet` | کلاس‌های `pos/neg` و زمینه‌ی هر نمونه برای ILP |

جداول sequence و label marker نسخه‌ی مستقل دارند. جداول زوج/سه‌تایی، context و examples marker محتوایی ندارند؛ اگر فایل حاضر باشد بدون اعتبارسنجی داخلی خوانده می‌شود، مگر `force=True` باشد یا runner به‌دلیل sequence version کهنه، کل rebuild را فعال کند.

## نسخه‌بندی cache و اجرای مجدد

`SEQUENCE_CACHE_VERSION` در `sequence_builder.py` شناسه‌ی `controlled-entry-halfspace-orientation-v10-restart-start-filter-v10` به‌علاوه‌ی `PHASE1_VERSION` است. `sequence_cache_is_current()` وجود parquetهای sequence و membership و برابر بودن `sequence_cache_version.txt` با شناسه‌ی جاری را کنترل می‌کند؛ محتوای سطرها یا فهرست مسابقه جداگانه hash نمی‌شود.

`LABEL_CACHE_VERSION` در `outcome_labeler.py` نسخه‌ی `own-half-loss-no-regain-5s-no-opponent-foul-v6` است؛ `label_sequences()` فقط وقتی parquet و marker حاضر و برابر باشند labeled cache را reuse می‌کند. markerهای context و sample مثال version مستقلی ندارند. برای اعمال تعریف جدید، اجرای `uv run python scripts/run_phase2.py --force` همه‌ی محصولات وابسته را بازسازی می‌کند.

با `--force` مراحل از ابتدا ساخته می‌شوند؛ builder توالی از merged data فاز ۱ استفاده می‌کند و cache خام/processed فاز ۱ را بر اساس قواعد خود مدیریت می‌کند. اگر فایل‌های ورودی فاز ۱ در دسترس نباشند، runner فاز ۲ نمی‌تواند داده‌ی مسابقه را از خودش بسازد.

## گزارش runner و معیارهای پذیرش چاپ‌شده

runner موارد زیر را به‌ترتیب گزارش می‌کند:

- تعداد توالی، تعداد بازی، حداقل/میانه/حداکثر تعداد اکشن، و فراوانی `end_reason`؛
- شمار labelهای دودویی، سهم کلاس اقلیت، علت‌های negative، جدول turnover zone × label، end reason × label و درصد regain/شوت حریف/ورود حریف به یک‌سوم دفاعی؛
- تعداد پیوند پاس و triple و سهم sequenceهای دارای pair؛
- شمار positive و negative، نسبت واقعی negative:positive، سهم تطبیق exact (`match_level==3`)، سهم fallback سراسری (`match_level==0`) و zone overlap.

آستانه‌های تشخیصی عبارت‌اند از طول توالی ۲–۲۵، سهم کلاس اقلیت حداقل ۱۰٪، sequence دارای pair حداقل ۵۰٪، zone overlap حداقل ۰٫۴ و سهم level 0 حداکثر ۵٪. الگوریتم sampling سقف level 0 را اعمال می‌کند و در صورت نیاز نسبت واقعی را کمتر از هدف ۲:۱ می‌گذارد. سایر آستانه‌ها در runner فقط پیام/آمار گزارش هستند و آن‌ها را به کد خروج غیرصفر تبدیل نمی‌کنند؛ آزمون داده‌ای سهم exact stratum حداقل ۸۰٪ را نیز انتظار دارد.

## آزمون‌ها و قراردادهای کنترل کیفیت

[`tests/test_phase2_sequences.py`](../tests/test_phase2_sequences.py) موارد زیر را کنترل می‌کند:

- حداقل یک توالی و طول مجاز تمام نمونه‌ها؛
- پوشش sequence در همه‌ی مسابقات و حداقل پنج sequence در هر match؛
- آغاز همه‌ی sequenceها در نیمه‌ی خودی؛
- برابر بودن تعداد membershipها با `n_actions` و یکتایی sequence id؛
- تعادل labelهای دودویی، تک label بودن هر sequence و مجموعه‌ی مجاز دو label؛
- سازگاری معیار منفی با turnover در نیمهٔ خودی، پنجرهٔ پنج‌ثانیه‌ای بازپس‌گیری و استثنای foul حریف؛
- مستقل‌بودن label از شوت و ورود حریف به یک‌سوم دفاعی؛
- regain مبتنی بر اکشن کنترل‌شده؛
- تطابق نام نیم‌فضاهای عمیق با قرارداد `y=0` سمت راست؛
- حداقل پوشش زوج‌ها، distinct بودن بازیکنان زوج و سه‌تایی؛
- حداقل ۰٫۴ overlap، نسبت نمونه‌گیری حداکثر ۲:۱، یکتایی sequence در مثال‌ها و هم‌خوانی کلاس با label؛
- حداقل ۸۰٪ تطبیق exact stratum طبق آزمون داده؛
- سهم `match_level=0` حداکثر ۵٪؛
- قطعی‌بودن sampling با seed ثابت.

بسیاری از تست‌ها با fixtureهای parquet واقعی اجرا می‌شوند؛ `tests/conftest.py` در صورت نبود خروجی‌های فاز ۲ خطای توضیحی می‌دهد و اجرای فاز ۲ را لازم می‌داند. تست‌های rule-level مانند transition پنج‌متری و موارد مرزی label دودویی را با داده‌ی مصنوعی می‌سازند.

## نکات تفسیر برای توسعه‌دهنده‌ی بعدی

- Sequence یک مالکیت/دوره‌ی build-up بارسلوناست و فقط اکشن‌های بارسلونا را به‌عنوان member نگه می‌دارد؛ labeler برای دیدن اکشن‌های حریف به جدول کامل match برمی‌گردد.
- `non_negative` ادعای موفقیت تاکتیکی نیست؛ فقط منفی‌بودن در معیار عملیاتی حاضر را تأیید نمی‌کند.
- تمام sequenceها یکی از دو label را می‌گیرند؛ الگوهای شوت و ورود حریف فقط در گزارش توصیفی بررسی می‌شوند.
- مثبت/منفی در `ilp_examples` برچسب هدف ILP است و به‌ترتیب معادل outcome منفی و غیرمنفی است.
- نمونه‌گیری context-matched قصد دارد قواعد outcome را از تفاوت زمینه‌ی شروع (مکان، فشار، طول) جدا کند؛ fallback سطح ۰ حداکثر ۵٪ است و در صورت کمبود pool، نسبت نمونه‌گیری کاهش می‌یابد.
- `last_action_row` و `row` به ترتیب اکشن‌های merged کل match متکی‌اند؛ تغییر ترتیب فاز ۱ می‌تواند مرز تحلیل پس از sequence را عوض کند و باید cache فاز ۲ نیز بازسازی شود.
- labeler پنجره‌ی بررسی ۵ ثانیه‌ای transition را بر مبنای `time_seconds` و ترتیب ردیف‌های همان period اجرا می‌کند؛ این event-time است، نه tracking پیوسته.

## نقشه‌ی ارجاع

| موضوع | مسیر کد |
|---|---|
| اجرای ترتیبی، cache rebuild و آمار | [`scripts/run_phase2.py`](../scripts/run_phase2.py) |
| آغاز/پایان sequence، zone entry، schema و membership | [`src/tactical_ilp/sequences/sequence_builder.py`](../src/tactical_ilp/sequences/sequence_builder.py) |
| zone boundaries و آستانه‌های فاز ۲ | [`src/tactical_ilp/config.py`](../src/tactical_ilp/config.py) |
| loss detection، معیار سه‌ناحیه‌ای و reasonها | [`src/tactical_ilp/sequences/outcome_labeler.py`](../src/tactical_ilp/sequences/outcome_labeler.py) |
| pressure bucket و zone geometry | [`src/tactical_ilp/features/geometry.py`](../src/tactical_ilp/features/geometry.py) |
| استخراج pair/triple و خلاصه‌ی sequence | [`src/tactical_ilp/sequences/player_subsets.py`](../src/tactical_ilp/sequences/player_subsets.py) |
| context strata، seed و sampling ladder | [`src/tactical_ilp/sequences/negative_sampling.py`](../src/tactical_ilp/sequences/negative_sampling.py) |
| ثبت log اجرای طولانی | [`scripts/run_logging.py`](../scripts/run_logging.py) |
| fixtureها و تست‌های فاز | [`tests/conftest.py`](../tests/conftest.py)، [`tests/test_phase2_sequences.py`](../tests/test_phase2_sequences.py) |

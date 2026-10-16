# الگوریتم بازوی `hybrid_sequential` و انتقال patternها به Popper

این سند الگوریتم اجرایی **بازوی ۳ (هیبرید ماینر + Popper)** را از ساخت دنبالهٔ ورودی تا ساخت facts و bias قابل استفاده در Popper تشریح می‌کند. بازوی ۱، ماینر تنها، از همان patternهای منتخب هر fold استفاده می‌کند و با OR ساده آن‌ها پیش‌بینی می‌کند؛ بازوی ۲ درخت CART است؛ مسیر `full_tactical` بازوی ۰ و مرجع ثانویه است. پیاده‌سازی هیبرید در `sequential_patterns.py`، `patterns_only_inputs.py`، `phase5_vocabulary.py` و runnerهای `run_phase5_sequential.py` و `run_phase5_sequential_cv.py` قرار دارد.

## ۱. تعریف ورودی و دامنهٔ استخراج

واحد mining یک سکانس build-up برچسب‌خورده و توالی مرتب‌شدهٔ اکشن‌های عضو همان سکانس است. هر اکشن ابتدا با predicateهای تاکتیکی فازهای ۳ و ۴ توصیف می‌شود. بازوی تولیدی از `vocabulary="tactical"` استفاده می‌کند؛ modeهای `core` و `combined` در سازنده موجودند، اما ورودی run ترتیبی اصلی نیستند.

برچسب هدف یا رخدادهای پس از مرز سکانس به‌صورت item وارد pattern نمی‌شوند. الگوها فقط از سکانس‌های مسابقه‌های train هر اجرا/fold کشف می‌شوند. یک اکشن انتهایی پیش از ساخت دنبالهٔ mining کنار گذاشته می‌شود (`excluded_suffix_actions=1`). علاوه بر آن، روی آخرین اکشن نگه‌داشته‌شده کد فهرست مرزی `line_pass`, `first_time_play`, `dwelt_long`, `quick_line_break` را suppress می‌کند. `first_time_play` و `dwelt_long` به dwell مربوط‌اند؛ suppression برای `line_pass` نیز صریحاً در این فهرست اعمال می‌شود، هرچند تعریف فعال آن lane آغاز/پایان همان پاس را مقایسه می‌کند. `quick_line_break` در فهرست فعلی predicateهای اکشن‌محور صادرشده برای mining نیست، بنابراین در mode تولیدی itemی با این نام ساخته نمی‌شود. این mask فقط بر itemهای pattern اثر دارد و تعریف predicateها در BK را عوض نمی‌کند.

در اجرای CV، `fit_patterns()` فقط ردیف‌های مسابقه‌های train را برای تشکیل پایگاه mining و محاسبهٔ حمایت/رشد برمی‌دارد. frame ورودی patternها شامل همهٔ سکانس‌های برچسب‌خورده است؛ بنابراین انتخاب و امتیازدهی pattern در مسابقه‌های train بر همهٔ `negative`ها و `non_negative`های همان مسابقه‌ها انجام می‌شود، نه فقط شناسه‌های کنترل/نمونهٔ متوازن‌شده‌ای که به فایل‌های مثال Popper راه می‌یابند. محدودیت fold روی مسابقه اعمال می‌شود، درحالی‌که Popper با مثال‌های sampled آموزش می‌بیند. سپس facts الگوها برای frame ورودی نوشته می‌شوند که می‌تواند سکانس‌های train و test را دربرگیرد؛ این projection، الگوی ازپیش‌انتخاب‌شده را روی هر سکانس تطبیق می‌دهد و از برچسب test برای انتخاب یا امتیازدهی pattern استفاده نمی‌کند.

## ۲. ساخت itemset تاکتیکی هر اکشن

استخراج ویژگی با بازاستفاده از تعریف Prolog انجام می‌شود تا یک predicate تاکتیکی در Python نسخه‌برداری نشود. کد موقت SWI-Prolog فایل‌های `facts_all.pl`، `facts_tactical.pl` و `bk.pl` را consult می‌کند، برای هر عضو `WRAPPED_ACTION_PREDS` وجود predicate را روی اکشن‌های دارای `in_sequence/2` می‌سنجد و زوج‌های یکتای `(action, predicate)` را صادر می‌کند.

هر نماد اکشن با سه زمینهٔ مکانی/نوعی به یک atom تبدیل می‌شود:

```text
tactical:<predicate>@<spadl_type>@<start_channel>@<start_third>
```

برای نمونه، `tactical:line_pass@pass@left_channel@middle_third` یک پاس line-pass در کانال چپِ یک‌سوم میانی را نشان می‌دهد. برای کاهش پراکندگی الگوها، پنج lane پایه فقط در واژگان miner به سه کانال نگاشت می‌شوند: `left_wing` و `left_half_space` به `left_channel`، `central_channel` بدون تغییر، و `right_half_space` و `right_wing` به `right_channel`. این ادغام مختص بازوی ترتیبی است و قرارداد پنج‌lane در facts هندسی، نمونه‌گیری و سایر ویژگی‌های پروژه را تغییر نمی‌دهد. مختصات مفقود در نماد به `unknown` تبدیل می‌شود. اگر چند مفهوم تاکتیکی روی یک اکشن صدق کنند، آن اکشن چند item تاکتیکی دارد.

برای روشن‌بودن تفاوت modeها: mode تشخیصی `core` بیست نماد کوچک‌دامنه دارد (خانوادهٔ اکشن، نتیجه، سطح فشار، جهت، progressive، شکل diagonal/switch و timing). mode `combined` این نمادها را با نمادهای تاکتیکی ترکیب می‌کند. Runner تولیدی `tactical` است، پس آن ۲۰ نماد به‌طور پیش‌فرض وارد itemsetهای الگوی بازوی ۳ نمی‌شوند.

## ۳. استخراج الگوهای بستهٔ دارای gap محدود

الگوریتم `GapConstrainedClosedMiner` یک پیاده‌سازی محلی و bounded از prefix projection است. این پیاده‌سازی نباید به‌عنوان پیاده‌سازی canonical الگوریتم Gap-BIDE معرفی شود. هر سکانس به‌صورت فهرستی مرتب از itemsetها نمایش داده می‌شود؛ miner برای pattern، embeddingها را به شکل جایگاه‌های رخداد در هر `sequence_id` دنبال می‌کند.

### ۳.۱ بسط‌ها و پشتیبانی

- **شروع:** برای هر item، سکانس‌هایی که آن item را در دست‌کم یک اکشن دارند به‌عنوان مجموعهٔ حمایت singleton ساخته می‌شوند.
- **S-extension:** یک item در رخداد بعدی pattern افزوده می‌شود. با مقدار پیش‌فرض `max_gap=2` حداکثر دو اکشن میان دو گام pattern مجاز است؛ فاصلهٔ index دو گام حداکثر ۳ می‌شود.
- **I-extension:** در پیکربندی عمومی، می‌توان item دیگری را به همان رخداد pattern افزود تا conjunction درون رخداد ساخته شود. سقف پیش‌فرض `max_items_per_event=1` است، بنابراین patternهای انتخاب‌شدهٔ پیش‌فرض در هر گام فقط یک نماد تاکتیکی دارند؛ I-extension چندنمادی در این تنظیم رشد نمی‌کند.
- **حمایت:** support شمار sequenceهای متمایزی است که دست‌کم یک embedding پذیرفته‌شده دارند، نه شمار کل embeddingها. هر گسترش با حمایت کمتر از `min_support=5` دنبال نمی‌شود.

پس pattern دوگامی مانند `A → B` وقتی در یک سکانس وجود دارد که هر دو نماد با ترتیب حفظ شوند و فاصلهٔ بین‌ آن‌ها حداکثر سه جایگاه اکشن باشد. تکرارهای متعدد همان pattern در یک سکانس حمایت آن سکانس را چند برابر نمی‌کند.

### ۳.۲ بسته‌بودن و کنترل هزینهٔ جست‌وجو

الگوهای بازدیدشده بر اساس مجموعهٔ دقیق sequenceهای پشتیبان گروه‌بندی می‌شوند. patternی حفظ می‌شود که زیرساختار یک pattern بزرگ‌تر با همان مجموعهٔ حمایت نباشد. بررسی ساختاری اجازه می‌دهد اقلام گام‌های یک pattern در زیرمجموعهٔ گام‌های بزرگ‌تر embed شوند؛ بسته‌بودن در محدودهٔ جست‌وجو و gap پیکربندی‌شده معنا دارد، نه در کل فضای نامحدود الگوها.

پیکربندی پیش‌فرض mining چنین است: `min_support=5`، `min_target_support=5`، `min_growth_rate=1.5` و `smoothing=0.5`، طول pattern از ۲ تا ۳ event، `max_items_per_event=1`، سقف ۸ pattern انتخابی، حذف یک اکشن suffix و حداکثر ۵۰٬۰۰۰ candidate بازدیدشده. اگر شمار بازدید از سقف candidate بگذرد، جست‌وجو با خطا متوقف می‌شود؛ این سقف به معنی تعداد patternهای نهایی نیست.

## ۴. انتخاب تمایزبخش با توجه به کنترل هم‌زمینه

الگوهای بسته فقط روی sequenceهای منفیِ train استخراج می‌شوند تا بودجهٔ جست‌وجو صرف motifهای هدف شود؛ سپس حمایت هر الگو روی sequenceهای کنترل محاسبه می‌شود. کاندیدا باید دست‌کم ۵ حمایت هدف داشته باشد. نرخ هدف و کنترل با هموارسازی Laplace `alpha=0.5` محاسبه می‌شود و الگوهایی با growth کمتر از `1.5` حذف می‌شوند. این growth یک معیار heuristic برای رتبه‌بندی داخل train است، نه آزمون معناداری؛ عملکرد نهایی را باید با CV گروه‌بندی‌شده بر اساس مسابقه سنجید.

الگوهای قبول‌شده بر اساس growth هموارشده، حمایت هدف، طول و کلید قطعی مرتب می‌شوند. انتخاب فقط روی train هر fold انجام می‌شود و نتیجه‌ی نهایی همچنان باید روی مسابقات holdout ارزیابی شود.

## ۵. فشرده‌سازی انتخاب pattern و facts خروجی

پس از رتبه‌بندی، `select_compressing_patterns()` فقط patternی را می‌پذیرد که دست‌کم یک predicate تاکتیکیِ مؤلفه‌ای آن هنوز در مجموعهٔ مؤلفه‌های استفاده‌نشده باشد. مؤلفه‌ها برای جلوگیری از انتخاب چند pattern تکراری/هم‌پوشان در این مرحله علامت‌گذاری می‌شوند؛ این گزینش، حذف آن predicateها از bias نهایی نیست.

هر pattern پذیرفته‌شده نامی مانند `seq_pattern_01` می‌گیرد. `write_pattern_facts()` ترتیب نمادهای pattern، فاصلهٔ مجاز، شمار توصیفی فاصله‌های مشاهده‌شده در sequenceهای پشتیبان train، حمایت train در target/control، نرخ‌ها، نرخ هدف و کنترل، growth و log2-growth هموارشده و پوشش projection را در metadata می‌نویسد. برای هر سکانسی که pattern روی itemsetهایش match شود، fact بولی زیر نوشته می‌شود:

در cross-validation، این شماره‌ها فقط در همان fold معنا دارند؛ `seq_pattern_01` در دو fold ممکن است زنجیره‌های متفاوتی باشد. تکرار یا پایداری را باید با متن کامل pattern مقایسه کرد، نه پسوند عددی predicate.

```prolog
seq_pattern_01(s_..._....).
```

این fact وجود pattern مرتب را در سکانس می‌رساند؛ فاصلهٔ دقیق رخدادها یا هویت بازیکن را در آرگومان predicate رمز نمی‌کند. metadata برای توضیح زنجیرهٔ pattern و فاصله‌هاست و با fact بولی یکسان نیست.

## ۶. ساخت bias ترکیبی Popper

واژگان پایهٔ فعال ۳۴ predicate است (۲۷ اکشن، ۶ سکانسی و ۱ بازیکن-سکانس). bias روش hybrid تمام این predicateهای پایه را نگه می‌دارد و patternهای پذیرفته‌شده را به‌عنوان اتم‌های `seq_pattern_XX` به آن‌ها می‌افزاید؛ بنابراین با سقف ۸ pattern، bias حداکثر ۴۲ predicate بدنه دارد. در fallback بدون pattern همان ۳۴ predicate پایه باقی می‌مانند. قید bias اجازه نمی‌دهد pattern با یکی از مؤلفه‌های تاکتیکی خودش در همان clause ترکیب شود. همچنین `seq_flank_dribble` با `seq_flank_line_carry` و `seq_flank_cut_inside` در یک clause منع شده است. این قیود clause-level هستند.

هدف Popper `negative_outcome(Sequence)` است. bias نوع‌ها را به sequence محدود می‌کند و سقف جاری `max_vars=2`, `max_body=3`, `max_clauses=4` دارد. چند clause برای head یک OR منطقی می‌سازند. Popper از facts پایه، BK و facts الگوها همراه با فایل‌های مثال Prolog استفاده می‌کند تا برنامه‌ای نویزی/MDL بیاموزد. pattern یک literal اتمی است؛ Popper ترتیب داخلی pattern را دوباره یاد نمی‌گیرد، بلکه آن را از miner به‌عنوان یک feature آماده دریافت می‌کند.

اگر هیچ pattern از آستانه‌های mining و گزینش عبور نکند، `pattern_fallback` فعال می‌شود و bias بدون predicateهای pattern، با واژگان تاکتیکی پایه ساخته می‌شود. بنابراین نام runner روش ترکیبی است، اما وضعیت مؤثر هر fold باید از metadata همان اجرا خوانده شود؛ در fallback آن fold عملاً pattern ترتیبیِ منتخب ندارد.

## ۷. تفکیک سه مفهوم آماری

برای جلوگیری از تعبیر نادرست نتایج، این مقادیر از هم جدا هستند:

1. **Support miner:** تعداد sequenceهای train در یک کلاس که pattern را match می‌کنند.
2. **معیار انتخاب pattern:** حداقل حمایت هدف و growth هموارشده با alpha=0.5؛ این فیلتر heuristic است و ادعای معناداری آماری ندارد.
3. **معیار Popper روی test/CV:** سنجه‌هایی مانند precision، recall و lift برنامهٔ نهایی روی دادهٔ آزمون. این معیارها بعد از یادگیری محاسبه می‌شوند و با growth pattern برابر نیستند.

در مقاله باید تنظیمات هر run/fold، patternهای انتخاب‌شده و metadata، تعداد foldهایی که ruleها تکرار شده‌اند و سنجه‌های آزمون از گزارش CV همان اجرا نقل شوند. mining روی کل داده پیش از CV، یا درج growth به‌عنوان عملکرد پیش‌بینی، با پروتکل پیاده‌سازی فعلی سازگار نیست.

## منابع پیاده‌سازی

- [`sequential_patterns.py`](../src/tactical_ilp/sequences/sequential_patterns.py)
- [`patterns_only_inputs.py`](../src/tactical_ilp/prolog/patterns_only_inputs.py)
- [`phase5_vocabulary.py`](../src/tactical_ilp/prolog/phase5_vocabulary.py)
- [`run_phase5_sequential.py`](../scripts/run_phase5_sequential.py)
- [`run_phase5_sequential_cv.py`](../scripts/run_phase5_sequential_cv.py)

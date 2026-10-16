# فاز ۵ — یادگیری قواعد تاکتیکی با Popper و الگوهای ترتیبی

## هدف و جایگاه در خط لوله

فاز ۵ از توالی‌ها و برچسب‌های فاز ۲، فکت‌های پایه‌ی فاز ۳ و فکت‌ها/قواعد تاکتیکی فاز ۴ استفاده می‌کند تا فرضیه‌های منطقی قابل‌خواندن بسازد. هدف head در دو مسیر Popper (بازوی ۰ مرجع و بازوی ۳ هیبرید) چنین است:

```prolog
negative_outcome(Sequence)
```

مثبت برای Popper یعنی sequence با label برابر `negative`؛ منفی یعنی `non_negative`. پس «مثبت» در فایل `exs.pl` معادل outcome نامطلوب build-up است. قواعد آموخته‌شده ارتباط‌های آماری/نمادین را صورت‌بندی می‌کنند و به‌تنهایی علت‌ومعلول را اثبات نمی‌کنند؛ قواعد برای ارزیابی روی مسابقات کنارگذاشته‌شده و داوری تاکتیکی ارائه می‌شوند.

```text
Phase 1: SPADL + 360 + timing
 → Phase 2: sequences + outcomes + context-matched examples
 → Phase 3: ground facts + Phase 4: tactical facts/BK
 → Phase 5: ILP inputs → Popper search → match-held-out evaluation
```

## سه بازوی اصلی مقایسه و بازوی مرجع

سه روش اصلی عبارت‌اند از **بازوی ۱: ماینر تنها**، **بازوی ۲: درخت تصمیم** و **بازوی ۳: هیبرید ماینر + Popper**. مسیر تاریخی Popper با واژگان کامل (`full_tactical`) به‌عنوان **بازوی ۰ / مرجع ثانویه** حفظ می‌شود و در جدول اصلی مقایسه وارد نمی‌شود.

| بازو | روش | دادهٔ ورودی و قاعدهٔ پیش‌بینی |
|---|---|---|
| ۰، مرجع | Popper با واژگان کامل (`full_tactical`) | ویژگی‌های تاکتیکی سطح سکانس؛ گزارش تاریخی جداگانه |
| ۱، اصلی | ماینر تنها | در هر fold، OR الگوهای منتخبِ همان fold؛ اگر دست‌کم یک الگو match شود، توالی منفی پیش‌بینی می‌شود؛ بدون Popper و بدون predicateهای پایه به‌عنوان شرط پیش‌بینی |
| ۲، اصلی | درخت CART | درخت دودویی حداکثر عمق ۳ و حداقل ۱۰ نمونه در برگ، روی ۳۴ predicate بولی سطح سکانس fit می‌شود |
| ۳، اصلی | `hybrid_sequential` | Popper الگوهای منتخب همان fold را به‌همراه واژگان تاکتیکی پایه در clauseهای منطقی ترکیب می‌کند |

بازوی ۱ در `run_phase5_sequential_cv.py` در کنار بازوی ۳ سنجیده می‌شود و دقیقاً همان patternهای fold-local بعد از فشرده‌سازی را که به Popper عرضه شده‌اند، با قاعدهٔ OR ساده به‌کار می‌برد؛ اجرای جداگانه یا انتخاب الگو روی test ندارد. بنابراین زوج بازوهای ۱ و ۳ یک ablation مشخص می‌سازند: آیا Popper با ترکیب patternها و predicateهای پایه، نسبت به union خام patternها ارزش افزوده دارد؟ بازوی ۲ در `run_phase5_decision_tree.py` اجرا می‌شود. هر سه روش از seed و تقسیم‌های سه‌fold یکسان بر اساس مسابقه استفاده می‌کنند. در هر fold آموزش/انتخاب فقط با مسابقات train انجام می‌شود و سنجش روی همهٔ توالی‌های برچسب‌خوردهٔ مسابقات test با شیوع طبیعی است. خروجی‌ها به‌صورت pooled out-of-fold تجمیع می‌شوند.

این طراحی، نشت مسابقه بین train/test را می‌بندد و مقایسهٔ paired روی جمعیت آزمون مشترک می‌دهد. اما یک اجرای سه‌fold روی یک فصل، ارزیابی داخلی و توصیفی است؛ به‌تنهایی فاصلهٔ اطمینان، آزمون معناداری، یا تعمیم به فصل/تیم دیگر را ثابت نمی‌کند. افزون بر تفاوت الگوریتم‌ها، بازنمایی و شیوهٔ آموزش هم متفاوت است: ماینر الگوهای ترتیبی را روی همهٔ توالی‌های train می‌جوید، درحالی‌که CART و Popper از نمونه‌های هم‌زمینهٔ آموزشی استفاده می‌کنند. پس اختلاف معیارها را به‌عنوان مقایسهٔ روش‌های کامل پروژه تفسیر کنید، نه اثر خالص یک الگوریتم.

درخت همان ۳۴ predicate دودویی سطح سکانس bias بازوی `full_tactical` (۲۷ اکشن، ۶ سکانسی و ۱ بازیکن-سکانس؛ `aerial_pass` حذف است) را دریافت می‌کند. جزئیات طراحی و محدودیت‌های بازنمایی در [`decision_tree_baseline_and_hybrid_comparison.md`](decision_tree_baseline_and_hybrid_comparison.md) آمده است.

`fast_sequence` در واژگان فعال است و نرخ اکشن‌های صدک ۷۰ به بالا (دهک‌های ۸ تا ۱۰) را نشان می‌دهد. فاز ۴ آستانهٔ عددی محاسبه‌شده را در manifest ثبت می‌کند.

## قرارداد ورودی مشترک

- `labeled_sequences.parquet` مرجع label و match هر sequence است.
- `ilp_examples.parquet` نمونه‌های انتخاب‌شده‌ی فاز ۲ را تعیین می‌کند؛ negative controls از کل label table بدون قید انتخاب نمی‌شوند.
- sequenceهای انتخاب‌شده به `pos(negative_outcome(S)).` یا `neg(negative_outcome(S)).` تبدیل می‌شوند. `pos/1`, `neg/1` syntax مثال Popper است.
- sampling فاز ۲ حداکثر نسبت هدف ۲ کنترل به ۱ positive دارد و ابتدا contextهای همسان سطح ۳→۱ را مصرف می‌کند. fallback سراسری level 0 با `MAX_GLOBAL_MATCH_SHARE=0.05` سقف می‌خورد؛ کمبود کنترل مناسب باعث نسبت کمتر می‌شود، نه تزریق بی‌قید نمونه‌ی جهانی.
- مفهوم‌های action-level با wrapper existential به sequence-level تبدیل می‌شوند، مانند:

```prolog
seq_line_breaking_pass(S) :- in_prefix(A,S), line_breaking_pass(A).
```

معنا «وجود دست‌کم یک اکشن واجدشرط» است، نه تعداد وقوع‌ها یا ترتیب رخدادها. این wrapperها متغیرهای Action و Player را از بدنه‌ی قاعده دور نگه می‌دارند و فضای جست‌وجو را کوچک‌تر می‌کنند.

## بازوی ۰ / مرجع: `full_tactical`

### تولید فایل‌های Popper

ورودی‌های Popper شامل `exs.pl`, `bias.pl` و `bk.pl` هستند. BK تولیدی با مسیر نسبی `facts_all.pl`, `facts_tactical.pl` و قواعد پس‌زمینه را consult می‌کند و wrapperهای لازم را تعریف می‌کند.

Vocabulary فعال در `phase5_vocabulary.py` نگه‌داری می‌شود:

- **واژگان action-level جاری:** `WRAPPED_ACTION_PREDS` تعداد ۳۱ مفهوم دارد؛ ۲۷ مورد به bias Popper و miner ترتیبی تاکتیکی راه می‌یابند؛ `aerial_pass` از هر دو حذف شده و `long_pass` برای restart/set-piece برقرار نمی‌شود. `flank_to_center_progression` در bias مستقل اکشنی و واژگان miner نیست؛ Popper فقط alias سکانسی `seq_flank_to_center` را دریافت می‌کند. `triangulation_opponent_trapped` و `rondo_cage` نیز از جست‌وجو حذف شده‌اند.
- **۶ ویژگی sequence-level در Popper:** `opponent_visible_majority`, `slow_sequence`, `fast_sequence`, `keeper_played_open_play`, `seq_pass_exchange` و `seq_flank_to_center`. Aliasهای `seq_open_support` / `seq_no_open_support` در BK تولید می‌شوند، اما در bias فعال این تنظیم نیستند.
- **۱ player-sequence:** `persistent_half_space_occupier`. Wrapper می‌پرسد آیا دست‌کم یک بازیکن در همان توالی predicate را ارضا می‌کند.

مفهوم‌های زمانی `first_time_play` و `dwelt_long` از زمان نگه‌داری توپ به‌عنوان proxy استفاده می‌کنند: آستانه‌ها به‌ترتیب حداکثر ۰٫۵ ثانیه و بیش از ۲٫۵ ثانیه‌اند. `under_pressure` ویژگی مستقلی است و می‌تواند در قاعده با نشانه‌های زمانی ترکیب شود.

سه ویژگی flank از اکشن‌های SPADL نوع `dribble` و مختصات شروع/پایان ساخته می‌شوند: `flank_dribble` با شروع از `left_wing` یا `right_wing`؛ `flank_line_carry` با شروع و پایان در همان wing و پیشروی طولی حداقل ۵ متر؛ و `flank_cut_inside` با پایان در هر نیم‌فضا یا `central_channel`. این‌ها فکت ground هستند و به وجود فریم ۳۶۰ نیاز ندارند.

Bias جاری ترکیب `seq_flank_dribble` با هر یک از `seq_flank_line_carry` و `seq_flank_cut_inside` را در یک بند منع می‌کند. دو مفهوم مثلث/قفس (`triangulation_opponent_trapped` و `rondo_cage`) اصلاً به واژگان Popper عرضه نمی‌شوند. این قید هم‌وقوعی فقط در سطح بند است.

`pass_in_behind` علاوه بر پایان پشت خط دفاعی تخمینی، باید در یک‌سوم هجومی (x>70m) یا محوطهٔ جریمهٔ حریف پایان یابد. مختصات محوطه از ابعاد استاندارد زمین SPADL (عمق ۱۶٫۵m و عرض ۴۰٫۳۲m) گرفته می‌شود. چون محوطه در یک‌سوم هجومی قرار دارد، شرط محوطه صریح و مستند است اما از نظر مجموعهٔ نقاط زیرمجموعهٔ شرط یک‌سوم هجومی است. این مرز انتهایی `pass_in_behind` را از `line_breaking_pass` متمایز می‌کند؛ پاس خط‌شکن می‌تواند پیش از رسیدن به یک‌سوم هجومی هم رخ دهد.

`central_switch`, `recycle_to_same_flank`, و `center_cycling` قواعد Prolog روی دو پاس موفقِ پیاپی در ترتیب اکشن‌های پاس هستند؛ اکشن‌های غیرپاس میان آن دو مجازند، اما پاس دیگری نباید میانجی شود. مرکز همان `central_channel` در قرارداد پنج lane پروژه است. این‌ها پروکسی فضایی‌اند: ورودی جاری شناسه‌ی عمومی گیرنده‌ی پاس ندارد، بنابراین نمی‌توان اثبات کرد که همان هافبک پاس اول را دریافت و پاس دوم را ارسال کرده است. تعویض جناح جهت‌دار دو مفهوم مستقل `switch_to_right` و `switch_to_left` دارد: هر دو به موفقیت، تغییر lane و جابه‌جایی عرضی حداقل ۲۵ متر نیاز دارند؛ مقصد اولی باید در `right_half_space` یا `right_wing` و دومی در `left_half_space` یا `left_wing` باشد. هر دو با `central_switch` متفاوت‌اند: آن مفهوم مسیر دوپاسه از مرکز را می‌سنجد.

برچسب منبع `High Pass` در facts فاز ۴ برای ممیزی نگه داشته می‌شود، اما `aerial_pass` از bias Popper و واژگان miner حذف شده است تا برچسب ارتفاعِ کرنر و سایر ارسال‌های restart به قانون تاکتیکی تبدیل نشود. `long_pass` نیز در BK فقط برای pass-likeهای غیر restart برقرار است. سکانس‌های آغازشده با اوت، کرنر، ضربهٔ آزاد و ضربهٔ دروازهٔ بلند پیش از برچسب‌گذاری و ارزیابی حذف می‌شوند؛ ضربهٔ دروازهٔ کوتاه/متوسط به‌عنوان پاس بازی‌سازی دروازه‌بان باقی می‌ماند و می‌تواند ویژگی tempo بگیرد.

`costly_hesitation/1` در BK برای تحلیل تشخیصی حفظ شده، اما outcome-proximal است: مکث را به شکست اکشن بعدی (`failed(B)`) متصل می‌کند. در `OUTCOME_PROXIMAL_ACTION_PREDS` علامت خورده و عمداً به واژگان Popper وارد نمی‌شود. Predicateهای وابسته به `sequence_label/2` نیز با guard نشت اطلاعات رد می‌شوند. `action_result/2` و aliasهای `successful/1`, `failed/1` برای تعریف مفاهیم تاکتیکی در BK به کار می‌روند، اما result predicate خام در bias عرضه نمی‌شود.

Wrapperهای action-level و player-sequence به مفهوم sequence-level تبدیل می‌شوند؛ predicateهای sequence-level مستقیم عرضه می‌شوند. `seq_open_support/1` و `seq_no_open_support/1` ممکن است در BK wrapper تعریف شوند، اما اگر در فهرست body نباشند Popper آن‌ها را جست‌وجو نمی‌کند.

### Bias و محدودیت جست‌وجو

Bias با `head_pred`, `body_pred`, `type`, `direction` نوع و جایگاه آرگومان‌ها را مشخص می‌کند. پیش‌فرض‌های واژگان مشترک:

- `max_vars(2)`؛
- `max_body(3)`؛
- `max_clauses(4)`.

قیود منع هم‌وقوعی در یک بند برای زوج‌های هم‌پوشان از فهرست مرکزی تولید می‌شوند. کوچک نگه‌داشتن فضای فرضیه برای کنترل ترکیب‌های منطقی و grounding است؛ به Popper مختصات خام نمی‌دهیم، بلکه مفاهیم Boolean یا کوچک‌دامنه عرضه می‌کنیم.

### اجرای Popper

`scripts/run_phase5.py` ابتدا وجود facts و manifest جاری فاز ۴ را بررسی می‌کند، ورودی‌ها را می‌سازد و با SWI-Prolog بارگذاری/پاسخ‌گویی پایه را اعتبارسنجی می‌کند. اجرای Popper از محیط `uv` جداگانه و Python حداقل 3.14 با `janus_swi` انجام می‌شود؛ `POPPER_DIR` لازم است و SWI-Prolog باید حداقل 9.1.12 باشد.

Popper در حالت noisy اجرا می‌شود و می‌تواند خطای طبقه‌بندی را تحمل کند. worker پروژه `max_clauses` را از bias می‌خواند و سقف را به noisy combiner و greedy upper-bound نیز اعمال می‌کند. جست‌وجوی اصلی ۹۰۰ ثانیه و hard stop بیرونی ۹۶۰ ثانیه است. MDL در noisy setting هزینه‌ی اندازه‌ی برنامه و خطاهای مثبت/منفی را موازنه می‌کند؛ به‌طور شهودی، قاعده‌ی کوتاه با خطای کمتر ترجیح دارد. این بهینه‌سازی تضمین نمی‌کند که قاعده از نظر تاکتیکی علّی باشد.

خروجی در `data/prolog/popper/` و log در `data/reports/phase5_popper_output.txt` نوشته می‌شود. incumbent چاپ‌شده پیش از خاتمه provisional است؛ timeout، `NO SOLUTION` یا خطای اجرا نتیجه‌ی یادگیری موفق محسوب نمی‌شود. فرمان امتیازدهی روی holdout پس از run نمایش داده می‌شود.

## بازوی ۳: `hybrid_sequential`، patternهای ترتیبی در bias ترکیبی

### ساخت itemsetها

`sequential_patterns.py` برای هر sequence فهرستی مرتب از eventها می‌سازد. Runner آموزشی/CV از `vocabulary="tactical"` استفاده می‌کند: مفاهیم action-level مجاز را با اجرای SWI-Prolog روی همان facts و BK استخراج می‌کند تا تعریف تاکتیکی جداگانه‌ای در Python کپی نشود. هر رویداد تاکتیکی با نوع SPADL، کانال شروعِ درشت‌دانه و یک‌سوم طولی به یک اتم تبدیل می‌شود؛ برای نمونه `tactical:line_pass@pass@left_channel@middle_third`. پنج lane پایه فقط در واژگان miner ادغام می‌شوند: بال چپ و نیم‌فضای چپ کانال چپ، کانال مرکزی همان مرکز، و نیم‌فضای راست و بال راست کانال راست هستند. این ادغام فقط برای کاهش پراکندگی itemهای الگوی بازوی ترتیبی است و facts هندسی همچنان پنج lane دارند. ساخت اتم مرکب، همراهی مفهوم/مکان/نوع را حتی با سقف `max_items_per_event=1` حفظ می‌کند. مختصات مفقود به مقدار `unknown` تبدیل می‌شوند.

مفاهیم هندسی زیر از مختصات SPADL ساخته می‌شوند: `central_long_carry` (طول مسیر ≥۳٫۵ متر در هر یک از سه lane غیرکناری)، `central_take_on` (اکشن take-on در همین laneها)، `central_backward_pass` (طول ≥۷ و Δx≤−۵ متر)، `flank_backward_carry` (همان بال و Δx≤−۴ متر)، و `flank_to_center_progression` (پاس از wing به half-space/مرکز، طول ≥۱۰ و Δx≥۵ متر). این مفهوم در bias اکشنی و واژگان miner مستقل نیست؛ Popper فقط alias متقارن `seq_flank_to_center` را می‌بیند. گزارهٔ `seq_pass_exchange` فقط وجود تبادل موفقِ دوطرفه میان دو بازیکن متفاوت را ثبت می‌کند؛ شناسهٔ گیرنده فقط در محاسبهٔ گذرای فکت به کار می‌رود و در فایل Prolog ذخیره نمی‌شود.

سازنده علاوه بر آن modeهای `core` و `combined` دارد. core vocabulary شامل ۲۰ نماد بولی است: هفت action family (pass/cross/carry/defensive/shot/restart/other)، دو result، سه pressure bucket، سه جهت، progressive، diagonal، switch و timingهای hesitation/one-touch. این دو mode برای تحلیل/ابزارهای تشخیصی‌اند؛ arm تولیدی از tactical mode استفاده می‌کند.

### مرز زمانی و نشت

پیش‌فرض `excluded_suffix_actions=1` اکشن پایانی را پیش از ساخت pattern sequence کنار می‌گذارد. itemسازی روی آخرین action نگه‌داشته‌شده فهرست صریح `line_pass`, `first_time_play`, `dwelt_long`, `quick_line_break` را suppress می‌کند؛ `quick_line_break` در واژگان صادرشدهٔ miner تولیدی نیست و `line_pass` نیز در این مرز به‌صورت صریح mask می‌شود. برچسب outcome به نماد تبدیل نمی‌شود و mining به اکشن‌های حریف پس از مرز دسترسی نمی‌گیرد.

### الگوریتم mining و رتبه‌بندی

`GapConstrainedClosedMiner` یک prefix-projection محلی و bounded است؛ گزارش‌ها آن را canonical Gap-BIDE معرفی نمی‌کنند. دو بسط دارد: I-extension برای افزودن item به همان event و S-extension برای event بعدی با gap محدود. Closure با برابری دقیق مجموعه‌ی sequenceهای پشتیبان بررسی می‌شود و patternهای زیرساختار از pattern بزرگ‌تر با همان support حذف می‌شوند.

الگوهای بسته فقط از sequenceهای منفی train استخراج می‌شوند، سپس روی کنترل‌های train نیز پشتیبانی‌شان محاسبه می‌شود. انتخاب از حداقل حمایت هدف ۵ و growth هموارشده با alpha=0.5 و آستانهٔ 1.5 استفاده می‌کند. این معیار heuristic است؛ به‌تنهایی معناداری آماری یا عملکرد پیش‌بینی را نشان نمی‌دهد. انتخاب در هر fold فقط با train انجام می‌شود و معیار مقایسه، عملکرد روی مسابقات holdout در CV گروه‌بندی‌شده است.

| تنظیم پیش‌فرض | مقدار | نقش |
|---|---:|---|
| `max_gap` | 2 | حداکثر دو event میانی؛ فاصله‌ی index حداکثر 3 |
| `min_support` | 5 | حداقل حمایت الگو هنگام mining روی sequenceهای منفی train |
| `min_target_support` | 5 | حداقل تعداد sequence هدف که pattern را پشتیبانی می‌کند |
| `min_growth_rate` | 1.5 | حداقل نسبت نرخ هموارشدهٔ هدف به کنترل |
| `smoothing` | 0.5 | هموارسازی Laplace نرخ‌ها |
| `min_pattern_events` / `max_pattern_events` | 2 / 3 | طول pattern بر حسب event |
| `max_items_per_event` | 1 | سقف item در هر event |
| `max_selected_patterns` | 8 | سقف patternهای نهایی |
| `excluded_suffix_actions` | 1 | تعداد رخداد انتهایی که پیش از mining کنار گذاشته می‌شود |
| `max_candidates` | 50,000 | توقف سخت برای کنترل رشد جست‌وجو |

Patternها بر پایهٔ growth هموارشده، حمایت target، طول و ترتیب قطعی مرتب می‌شوند. پیش‌فرض سقف انتخاب ۸ است؛ بنابراین حداکثر هشت candidate برتر وارد گزینش/فشرده‌سازی و فایل facts می‌شوند.

### Bias ترکیبی و محدودیت مهم نام‌گذاری

`select_compressing_patterns()` بخش predicate را از اتم مکانی مرکب جدا می‌کند و فقط candidateی را می‌پذیرد که دست‌کم یک action predicate تاکتیکیِ هنوز موجود را مصرف کند. تمام predicateهای پایه در bias باقی می‌مانند. قید ضد tautology فقط ترکیب pattern با مؤلفه‌های تاکتیکی خودش را در یک clause منع می‌کند. `write_pattern_facts()` patternها را به `seq_pattern_01(S)` و مانند آن به شکل ground fact projection می‌کند. متادیتای گزارش برای هر pattern، زنجیره‌ی کامل گام‌ها (predicate، نوع SPADL، کانال درشت‌دانه، third)، قید فاصله‌ی مجاز، و شمارش فاصله‌های مشاهده‌شده‌ی embeddingها را نگه می‌دارد. فاصله‌ی مجاز از `max_gap` می‌آید؛ توزیع فاصله‌ی مشاهده‌شده فقط از sequenceهایی محاسبه می‌شود که در `candidate.support_ids` متعلق به داده‌ی train هستند، تا آمار holdout وارد متادیتای توصیفی pattern نشود. شمار پوشش فکت‌ها جداگانه روی frame ورودی projection گزارش می‌شود.

`patterns_only_inputs.py` دو builder دارد: bias واقعاً patterns-only، با `max_body=1`, `max_vars=1` و فقط pattern atoms؛ و bias ترکیبی. `run_phase5_sequential.py` از `build_hybrid_bias()` استفاده می‌کند، نه builder خالص. در bias ترکیبی همهٔ predicateهای پایه باقی می‌مانند و patternهای منتخب به‌صورت اتمی به آن‌ها افزوده می‌شوند؛ با حذف `aerial_pass`، ۳۴ predicate پایه و حداکثر ۸ pattern (در مجموع حداکثر ۴۲) وجود دارد. سقف جاری `max_vars=2`, `max_body=3`, `max_clauses=4` است.

در runner آموزشی، مثال‌های context-matched با seed `202021` در سطح match به ۷۵٪ آموزش و ۲۵٪ holdout تقسیم می‌شوند. pattern mining فقط روی labelled sequenceهای matchهای آموزش fit می‌شود؛ pattern features انتخاب‌شده سپس روی frame سکانس‌ها projection می‌شوند و Popper با `exs.pl` آموزش می‌بیند. `exs_holdout.pl` از matchهای جداگانه ساخته می‌شود. در CV، mining در هر fold فقط روی matchهای training همان fold fit و بازسازی می‌شود؛ pattern features برای task همان fold projection می‌شوند. بنابراین انتخاب pattern در هیچ‌یک از این دو ارزیابی به labelهای holdout وابسته نیست.

## صورت‌بندی Popper و خروجی قواعد

فایل‌های استاندارد task:

| فایل | محتوا |
|---|---|
| `exs.pl` | مثال‌های `pos/1` و `neg/1` sequence-level |
| `bias.pl` | head/body، نوع و جهت آرگومان، سقف متغیر/literal/clause |
| `bk.pl` | قواعد و wrapperها به‌همراه consult فایل facts |
| `facts_patterns.pl` | فکت‌های pattern برای arm ترتیبی |
| log | trace، incumbentها و نتیجه‌ی جست‌وجو |

قاعده‌ی فرضی:

```prolog
negative_outcome(S) :- seq_dead_end_support(S), opponent_visible_majority(S).
```

بدنه conjunction است؛ چند clause برای head، OR منطقی می‌سازند. Popper در bias محدودشده candidate programs را می‌سازد و در حالت noisy اجازه‌ی خطای داده را می‌دهد. محدودیت `max_clauses` در worker علاوه بر bias به combiner اعمال شده تا برنامه‌ی ترکیبی هم از سقف عبور نکند. نتیجه‌ی `New best hypothesis` یک incumbent است؛ فقط وجود پایان موفق `SOLUTION` آن را فرضیه‌ی نهایی آن run می‌کند.

## ارزیابی سه بازوی اصلی و تفسیر معیارها

`phase5_cv_common.py` matchها را بین foldها توزیع می‌کند تا هیچ match در train و test مشترک نباشد. seed پیش‌فرض `202021` است. Popper با نمونه‌های context-matched از matchهای train آموزش می‌بیند؛ ارزیابی هر fold روی **تمام labelled sequences** از matchهای test و در prevalence طبیعی انجام می‌شود. روش ترتیبی، pattern mining را در هر fold دوباره فقط با training matches انجام می‌دهد.

- `support = TP + FP`: تعداد مثال‌های پیش‌بینی‌شده به‌عنوان negative؛
- `precision = TP/(TP+FP)`: خلوص پیش‌بینی‌های negative؛
- `recall = TP/تعداد positiveهای واقعی`: پوشش outcomeهای منفی؛
- `lift = precision/base_rate`: precision نسبت به شیوع پایه؛
- `rule_stability`: تکرار بدنه‌های بدون pattern میان foldها را می‌شمارد. `seq_pattern_XX` نام محلی هر fold است؛ قاعده‌های دارای pattern فقط در fold کشف‌شان گزارش می‌شوند تا دو زنجیره‌ی متفاوت با شماره‌ی یکسان، تکرارشونده شمرده نشوند.

`compare_phase5_methods.py` جست‌وجوی Popper یا استخراج ویژگی انجام نمی‌دهد؛ گزارش ماینر تنها و هیبرید را از CV ترتیبی، گزارش CART و گزارش مرجع بازوی ۰ را می‌خواند. جدول اصلی فقط سه بازوی اصلی را نشان می‌دهد و بازوی ۰ را جداگانه به‌عنوان مرجع ثانویه درج می‌کند. اسکریپت نسخهٔ پروتکل و تقسیم مسابقه‌ها را تطبیق می‌دهد؛ تا وقتی هر سه روش با foldهای همسان و نسخهٔ جاری ارزیابی نشده باشند، گزارش را آمادهٔ مقایسه اعلام نمی‌کند.

معیارهای اصلی، ماتریس اغتشاش pooled OOF، support، precision، recall، specificity، F1، lift، balanced accuracy و MCC هستند. چون prevalence طبیعی آزمون حفظ می‌شود، accuracy و precision با دادهٔ نمونه‌گیری‌شدهٔ train یکی نیستند. گزارش یک اجرای سه‌fold، خروجی توصیفی است و آزمون فرض/فاصلهٔ اطمینان ارائه نمی‌کند؛ برای ادعای برتری، اجرای تکرارشده یا آزمون بیرونی روی فصل/تیم مستقل لازم است.

CV runner علاوه بر برنامهٔ نهایی هر fold، incumbentهای میانی را نیز به‌صورت فهرست اکتشافی گزارش می‌کند. فاز ۶ نامزدهای میانی ثبت‌شده در گزارش CV را بدون سقف تعداد و بدون آستانه‌ی سخت lift/support، مشروط بر قابلیت نگاشت فنی، برای ارزیابی و بازبینی مربی نگه می‌دارد. این معیارهای fold-local اکتشافی و وابسته به انتخاب‌اند؛ شواهد train/holdout فاز ۶ همراهشان نمایش داده می‌شود و هیچ‌یک به‌تنهایی تعمیم مستقل یا علیت را ثابت نمی‌کند.

## اجرا و فایل‌های خروجی

از ریشه‌ی پروژه، با داده و محیط Popper آماده:

```powershell
uv run python scripts/run_phase5.py
uv run python scripts/run_phase5_eval.py --rule "seq_dead_end_support,opponent_visible_majority"
uv run python scripts/run_phase5_cv.py
uv run python scripts/run_phase5_sequential.py
uv run python scripts/run_phase5_sequential_cv.py
uv run python scripts/compare_phase5_methods.py
```

CV پیش‌فرض سه fold دارد؛ sequential runner پارامترهای gap، support هدف، حداقل growth هموارشده، طول الگو، تعداد patternها، سقف candidate و تعداد suffix حذف‌شده را می‌پذیرد. `POPPER_DIR` باید مسیر checkout Popper باشد. مسیرهای متداول:

- `data/prolog/popper/` و `data/prolog/popper_sequential/`؛
- `data/prolog/popper_*_cv_fold_NN/` برای taskهای fold؛
- `data/reports/phase5_*training*.json` و logها؛
- `data/reports/phase5_*cv.json` و Markdownها؛
- `data/reports/phase5_method_comparison.json/.md`.

ورودی baseline manifest `phase4_feature_version` را با کد جاری تطبیق می‌دهند. `PATTERN_INPUT_VERSION=16` و `COMPARISON_PROTOCOL_REVISION=phase5-three-primary-arms-v16-shared-test-population-hash` نسخهٔ ورودی و قرارداد مقایسه را مشخص می‌کنند. هر گزارش CV علاوه بر split مسابقه، hash مسابقه/توالی/برچسب هر test fold را ثبت می‌کند؛ comparer باید splitها، نسخهٔ پروتکل و hashهای جمعیت آزمون را همسان ببیند تا مقایسه را آماده اعلام کند. تغییر نسخه‌ی pattern input باعث بازسازی خودکار cache نمادهای ترتیبی قدیمی با نگاشت سه‌کاناله می‌شود. نسخه‌های ورودی داده نیز `PHASE1_VERSION=phase1-v7-orientation-fix-ms-time-360-linkage`, `PHASE3_FEATURE_VERSION=active-popper-vocabulary-v7-prefix-only-final-action-facts` و `PHASE4_FEATURE_VERSION=phase4-v11-prefix-only-tempo-pass-exchange` هستند. نتایج ثبت‌شده با شناسهٔ پروتکل قدیمی یا bias نسخه‌ی قدیمی با خروجی این نسخه قابل‌مقایسهٔ مستقیم نیستند. باید فازهای مشتق‌شده را به‌ترتیب دوباره اجرا کرد؛ شناسه‌ها جای hash کامل داده/facts نیستند.

## محدودیت‌ها و تفسیر مسئولانه

1. Label فاز ۲ تعریف عملیاتی از outcome build-up است، نه سنجه‌ی کامل ارزش مالکیت یا کیفیت همه‌ی تصمیم‌ها.
2. Training controls context-matched هستند؛ توزیعشان مثل جمعیت کامل نیست. به همین علت ارزیابی روی همه‌ی sequenceهای held-out و نرخ طبیعی گزارش می‌شود.
3. Predicateهای existential وقوع را می‌سنجند، نه تعداد یا ترتیب؛ تنها patternهای ترتیبی ترتیب را نمایش می‌دهند.
4. `action_result/2` و label factها body predicate نیستند، اما بعضی مفهوم‌های تاکتیکی از موفقیت اکشن مشتق می‌شوند. چون label عملیاتی به رخدادهای پایانی هم وابسته است، این ویژگی‌ها می‌توانند outcome-proximal باشند؛ CV گروه‌بندی‌شده بر اساس match، این وابستگی درون sequence را حذف نمی‌کند. تفسیر را به «بدون دسترسی مستقیم به label/result خام» محدود کنید، نه ادعای حذف تمام target association.
5. Pattern arm به‌طور مستقل یک suffix action را پیش از mining حذف می‌کند و سیگنال‌های look-ahead مرزی را suppress می‌کند؛ این حفاظت فقط به ورودی mining همان arm محدود است.
6. هر association می‌تواند از وضعیت بازی، حریف یا انتخاب نمونه اثر گرفته باشد؛ pattern mining رابطه‌ی علّی اثبات نمی‌کند.
7. Timeout، `NO SOLUTION` و provisional incumbent را از نتیجه‌ی نهایی جدا نگه دارید.
8. precision، recall، lift، support، ثبات بین foldها، false positives، زمان جست‌وجو و پرسش پژوهشی را با هم بخوانید.

## فایل‌های منبع

| بخش | کد |
|---|---|
| vocabulary و محدودیت‌ها و wrapper BK | [`phase5_vocabulary.py`](../src/tactical_ilp/prolog/phase5_vocabulary.py) |
| ساخت مثال‌ها و bias روش `full_tactical` | [`run_phase5.py`](../scripts/run_phase5.py) |
| itemset، mining، scoring و pattern facts | [`sequential_patterns.py`](../src/tactical_ilp/sequences/sequential_patterns.py) |
| pure/hybrid bias و pattern compression | [`patterns_only_inputs.py`](../src/tactical_ilp/prolog/patterns_only_inputs.py) |
| اجرای Popper، validation و noisy clause cap | [`run_phase5.py`](../scripts/run_phase5.py) |
| آموزش بازوی ترتیبی | [`run_phase5_sequential.py`](../scripts/run_phase5_sequential.py) |
| fold، scoring و ابزارهای مشترک | [`phase5_cv_common.py`](../scripts/phase5_cv_common.py) |
| baseline درخت تصمیم و CV گروهی | [`run_phase5_decision_tree.py`](../scripts/run_phase5_decision_tree.py) |
| CV بازوی ۰ مرجع، سه بازوی اصلی و مقایسه | [`run_phase5_cv.py`](../scripts/run_phase5_cv.py), [`run_phase5_sequential_cv.py`](../scripts/run_phase5_sequential_cv.py), [`run_phase5_decision_tree.py`](../scripts/run_phase5_decision_tree.py), [`compare_phase5_methods.py`](../scripts/compare_phase5_methods.py) |
| تعریف outcome و نمونه‌گیری منفی | [`outcome_labeler.py`](../src/tactical_ilp/sequences/outcome_labeler.py), [`negative_sampling.py`](../src/tactical_ilp/sequences/negative_sampling.py) |
| facts و قواعد تاکتیکی مصرفی | [`prolog_facts.py`](../src/tactical_ilp/prolog/prolog_facts.py), [`tactical_facts.py`](../src/tactical_ilp/prolog/tactical_facts.py), [`bk.pl`](../src/tactical_ilp/prolog/bk.pl) |

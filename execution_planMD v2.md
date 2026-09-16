# فازبندی اجرایی و تسک‌بندی پروژه — نسخه ۲.۰ (اصلاح‌شده)
## سیستم آنالیز تاکتیکی نورو-سمبلیک — از سند فنی تا اجرا

**نسخه:** ۲.۰ — بازبینی‌شده پس از مرور نسخه ۱.۰
**پیش‌نیاز:** سند فنی نسخه ۲.۰ (تأییدشده)
**نحوه استفاده:** هر فاز شامل چند تسک است. هر تسک یک معیار پذیرش مشخص و یک دستور تست pytest دارد. تسک بعدی را فقط بعد از سبز شدن تست تسک قبلی شروع کنید.

> **تغییرات کلیدی نسبت به نسخه ۱.۰:**
>
> - تسک جدید ۲.۴ برای طراحی صریح استراتژی Negative Sampling (پیش از فاز ۴ و ۵، نه ضمنی در فاز ۵)
> - تسک جدید ۴.۰ برای ساخت مجموعه نمونه‌های Expert-labeled (پیش‌نیاز تسک‌های ۴.۱ تا ۴.۳)
> - روشن‌سازی صریح در فاز ۳/۴ درباره اینکه Predicateهای هندسی به چه شکل در bk.pl می‌نشینند
> - تسک جدید ۲.۲ب برای عملیاتی‌کردن تعریف "پیامد منفی" (بخش ۲.۷ سند فنی) از همان فاز برچسب‌گذاری
> - معیار توقف/بازگشت صریح در تسک ۵.۲ و ۵.۳ (مرتبط با ریسک ۱ سند فنی)
> - تسک جدید ۵.۰ برای Smoke Test کوچک Popper پیش از اجرای روی یک بازی کامل
> - تسک جدید ۶.۰ برای Hold-out درون‌فصلی (ریسک ۶ سند فنی — Overfitting)
> - تسک ۶.۲ (Lift) اصلاح‌شده تا به یک الگوی synthetic نیز متکی باشد، نه فقط ظهور تصادفی در داده
> - تسک جدید ۵.۴ — چک‌پوینت رسمی بازخورد فاز ۴ ↔ فاز ۵ (به‌جای صرفاً یک یادداشت متنی، اکنون معیار پذیرش و تست مستقل دارد)

---

## نمای کلی فازها

| فاز | عنوان | خروجی نهایی |
|---|---|---|
| ۰ | راه‌اندازی محیط و زیرساخت | محیط کاری آماده، تست اتصال به داده |
| ۱ | بارگذاری و آماده‌سازی داده خام | جدول SPADL + فریم‌های 360 برای تمام بازی‌های بارسا در فصل ۲۰۲۰/۲۱ |
| ۲ | تفکیک سکانس‌های Build-up + طراحی Negative Sampling | مجموعه سکانس‌های برچسب‌خورده + استراتژی نمونه‌های منفی مستند |
| ۳ | ساخت لایه Rule-based Feature Extractor | فایل Prolog Facts تولیدشده از داده |
| ۴ | فرمول‌بندی دانش تاکتیکی (bk.pl مشترک) | bk.pl کامل با مشارکت مستقیم شما |
| ۵ | تعریف بایاس و اجرای Popper (با چک‌پوینت بازخورد رسمی به فاز ۴) | اولین اجرای موفق ILP با قوانین خام خروجی |
| ۶ | فیلتر، ارزیابی، Hold-out و تفسیر قوانین | قوانین نهایی با Support/Confidence/Lift/Hold-out |
| ۷ | نمایش بصری و گزارش نهایی | گزارش قابل‌فهم با نقشه زمین برای هر الگو |
| ۸ | مستندسازی و بسته‌بندی MVP | جمع‌بندی، مستندات، آماده برای فاز توسعه |

---

## فاز ۰ — راه‌اندازی محیط و زیرساخت

### تسک ۰.۱ — راه‌اندازی پروژه Python و مدیریت پکیج
**کار:** نصب `uv`، ساخت پروژه، تعریف `pyproject.toml` با وابستگی‌های اصلی (statsbombpy, socceraction, kloppy, pandas, numpy, mplsoccer, pytest).
**معیار پذیرش:** دستور `uv sync` بدون خطا اجرا شود و تمام پکیج‌ها import شوند.
**تست:**
```python
def test_core_imports():
    import statsbombpy, socceraction, pandas, numpy, mplsoccer
    assert True
```

### تسک ۰.۲ — نصب و تست زنجیره Prolog/ILP
**کار:** نصب SWI-Prolog، Clingo، و Popper (طبق مستندات رسمی Popper). تست اجرای یک مثال نمونه‌ی پیش‌فرض Popper (مثال‌های آموزشی repo خودش).
**معیار پذیرش:** مثال نمونه Popper بدون خطا اجرا شود و خروجی قانون تولید کند.
**تست:**
```python
def test_popper_installed():
    import subprocess
    result = subprocess.run(["uv", "run", "popper.py", "examples/iggp-rps-next-score"], capture_output=True)
    assert result.returncode == 0
```
**یادداشت اصلاحی:** دستور تست نسخه ۱.۰ (`python -m popper --help`) با نحوه واقعی اجرای Popper طبق README رسمی (`uv run popper.py <input dir>`) هم‌خوانی نداشت؛ در این نسخه اصلاح شد.

### تسک ۰.۳ — تست دسترسی به دیتاست هدف
**کار:** با statsbombpy، اتصال به StatsBomb Open Data و دریافت لیست بازی‌های بارسلونا در `competition_id=11, season_id=90`.
**معیار پذیرش:** لیست بازی‌ها دریافت شود و شامل حداقل ۲۰ بازی بارسلونا باشد.
**تست:**
```python
def test_barca_matches_available():
    from statsbombpy import sb
    matches = sb.matches(competition_id=11, season_id=90)
    barca = matches[(matches['home_team'] == 'Barcelona') | (matches['away_team'] == 'Barcelona')]
    assert len(barca) >= 20
```

### تسک ۰.۴ — تست دسترسی به داده 360
**کار:** برای حداقل یک بازی بارسا، بررسی وجود و بارگذاری موفق داده 360 (`sb.frames`).
**معیار پذیرش:** حداقل یک بازی با داده 360 کامل پیدا و بارگذاری شود.
**تست:**
```python
def test_360_data_available_for_at_least_one_match(barca_match_ids):
    from statsbombpy import sb
    found = False
    for match_id in barca_match_ids:
        try:
            frames = sb.frames(match_id=match_id)
            if len(frames) > 0:
                found = True
                break
        except Exception:
            continue
    assert found
```

---

## فاز ۱ — بارگذاری و آماده‌سازی داده خام

### تسک ۱.۱ — دانلود و کش کردن داده خام تمام بازی‌های بارسا
**کار:** اسکریپت `load_raw_data.py` که Event + Lineup + 360 تمام بازی‌های بارسا در آن فصل را دانلود و به‌صورت محلی (Parquet/JSON) کش می‌کند تا هر بار از اینترنت نخواند.
**معیار پذیرش:** فایل‌های کش‌شده برای همه بازی‌ها ایجاد شوند؛ اجرای دوم اسکریپت از کش بخواند نه از اینترنت.
**تست:**
```python
def test_all_matches_cached(barca_match_ids, cache_dir):
    for match_id in barca_match_ids:
        assert (cache_dir / f"events_{match_id}.parquet").exists()
```

### تسک ۱.۲ — تبدیل Event Data به فرمت SPADL
**کار:** با socceraction، تبدیل داده خام هر بازی به جدول SPADL استاندارد.
**معیار پذیرش:** جدول SPADL برای هر بازی دارای ستون‌های الزامی (player_id, team_id, action_type, start_x, start_y, end_x, end_y, result) باشد و تعداد ردیف‌ها با تعداد Eventهای خام هم‌خوانی معقول داشته باشد.
**تست:**
```python
def test_spadl_schema(sample_spadl_df):
    required_cols = {"player_id","team_id","type_name","start_x","start_y","end_x","end_y","result_name"}
    assert required_cols.issubset(sample_spadl_df.columns)
    assert len(sample_spadl_df) > 0
```

### تسک ۱.۳ — پیوند فریم‌های 360 به رخدادهای SPADL
**کار:** merge کردن هر ردیف SPADL با فریم 360 متناظرش (بر اساس event_id مشترک)، طوری‌که موقعیت بازیکنان بدون توپ در لحظه هر اکشن قابل دسترسی باشد.
**معیار پذیرش:** حداقل ۷۰٪ از اکشن‌های Build-up (پاس/دریبل) دارای فریم 360 متناظر باشند (باقی به‌دلیل محدودیت پوشش دوربین طبیعی است).
**تست:**
```python
def test_360_merge_coverage(merged_df):
    build_up_actions = merged_df[merged_df["type_name"].isin(["pass", "dribble"])]
    coverage = build_up_actions["freeze_frame"].notna().mean()
    assert coverage >= 0.70
```

### تسک ۱.۴ — تعریف هویت و نقش بازیکنان بارسا
**کار:** ساخت جدول مرجع `players_reference` شامل شناسه هویتی (`player_identity`) و نقش تاکتیکی (`has_role`) هر بازیکن اصلی بارسا در آن فصل (طبق بخش ۲.۶ سند فنی).
**معیار پذیرش:** جدول شامل تمام بازیکنانی باشد که در حداقل ۵ بازی آن فصل حضور داشته‌اند؛ هر بازیکن دقیقاً یک نقش اصلی داشته باشد.
**تست:**
```python
def test_players_reference_complete(players_reference_df, spadl_all_matches):
    frequent_players = spadl_all_matches["player_id"].value_counts()
    frequent_players = frequent_players[frequent_players >= 5].index
    assert set(frequent_players).issubset(set(players_reference_df["player_id"]))
```

---

## فاز ۲ — تفکیک سکانس‌های Build-up + طراحی Negative Sampling

### تسک ۲.۱ — تعریف قاعده شروع/پایان یک سکانس Build-up
**کار:** پیاده‌سازی `sequence_builder.py` — منطقی که مشخص می‌کند یک سکانس Build-up از کجا شروع و کجا تمام می‌شود (مثلاً از دریافت توپ توسط دروازه‌بان/مدافع تا خروج از یک‌سوم دفاعی یا از دست دادن توپ).
**معیار پذیرش:** روی حداقل ۳ بازی نمونه اجرا شود و سکانس‌هایی با طول منطقی (نه ۱ اکشنی، نه کل نیمه) تولید کند.
**تست:**
```python
def test_sequence_length_reasonable(sample_sequences):
    lengths = [len(seq) for seq in sample_sequences]
    assert all(2 <= l <= 25 for l in lengths)
```

### تسک ۲.۲ — تعریف عملیاتی "پیامد منفی" و برچسب‌گذاری سکانس‌ها
**کار:** پیش از برچسب‌گذاری، طبق بخش ۲.۷ سند فنی («Target روی پیامد، نه صرف تکرار»)، یک تعریف دقیق و مستند از "پیامد منفی" بنویسید (مثلاً: از دست دادن توپ در ثانیه‌های X بعد از سکانس، در ناحیه Y از زمین، منجر به شوت/موقعیت حریف). سپس هر سکانس را طبق این تعریف مشخص‌شده برچسب بزنید.
**معیار پذیرش:** تعریف "پیامد منفی" در یک سند کوتاه مستقل مکتوب شده و شامل شرایط دقیق (نه توصیف کیفی مبهم) باشد؛ هر سکانس دقیقاً یک برچسب داشته باشد؛ نسبت مثبت به منفی به‌گونه‌ای نباشد که یک طرف کمتر از ۱۰٪ کل باشد.
**تست:**
```python
def test_label_balance(labeled_sequences_df):
    ratio = labeled_sequences_df["label"].value_counts(normalize=True)
    assert ratio.min() >= 0.10

def test_negative_outcome_definition_documented(outcome_definition_doc):
    # سند تعریف باید شرایط عددی/مکانی مشخص داشته باشد، نه فقط توضیح کیفی
    assert outcome_definition_doc["has_explicit_criteria"] is True
```
**یادداشت اصلاحی:** این تسک از نسخه ۱.۰ گسترش یافته تا مفهوم بخش ۲.۷ سند فنی (تفکیک الگوی سالم از مضر بر مبنای تعریف Target) را از همین مرحله عملیاتی کند، نه اینکه تا فاز ۶ (Lift) به تعویق بیفتد.

### تسک ۲.۳ — استخراج سکانس‌های چندبازیکنی نامزد (۲ تا ۳ نفره)
**کار:** برای هر سکانس، شناسایی زیرمجموعه‌های ۲ تا ۳ بازیکنی که در آن سکانس تعامل مستقیم داشته‌اند (پاس، پوشش، حمایت) — طبق بخش ۲.۵ سند فنی — تا بعداً به Popper داده شوند.
**معیار پذیرش:** برای حداقل ۵۰٪ سکانس‌ها، حداقل یک زیرمجموعه دو‌نفره معتبر استخراج شود.
**تست:**
```python
def test_multi_player_subsets_extracted(sequences_with_subsets):
    has_subset = sequences_with_subsets["player_pairs"].apply(lambda x: len(x) > 0)
    assert has_subset.mean() >= 0.50
```

### تسک ۲.۴ — طراحی و مستندسازی استراتژی Negative Sampling برای exs.pl (جدید)
**کار:** طبق ریسک ۲ سند فنی («نمونه‌های منفی نامعتبر → قوانین کاذب»)، پیش از ورود به فاز ۴ و ۵، استراتژی دقیق انتخاب نمونه‌های منفی برای `exs.pl` را تعریف کنید. این شامل: (الف) نمونه‌های منفی باید از نظر شرایط زمینه‌ای (فشار، ناحیه زمین، نوع اکشن) قابل‌مقایسه با نمونه‌های مثبت باشند، نه انتخاب تصادفی کور از کل دیتاست؛ (ب) مستندسازی این‌که چند نمونه منفی به ازای هر نمونه مثبت انتخاب می‌شود و چرا.
**معیار پذیرش:** سند استراتژی Negative Sampling نوشته و تأیید شود؛ روی یک نمونه کوچک (یک بازی)، بررسی شود که نمونه‌های منفی انتخاب‌شده از نظر توزیع ویژگی‌های زمینه‌ای (مثلاً ناحیه زمین) با نمونه‌های مثبت هم‌پوشانی معنادار دارند (نه کاملاً واگرا و نه کاملاً یکسان).
**تست:**
```python
def test_negative_samples_context_comparable(pos_examples_df, neg_examples_df):
    # میانگین موقعیت مکانی نمونه‌های منفی نباید به‌طرز افراطی با مثبت‌ها متفاوت باشد
    pos_zone_dist = pos_examples_df["zone"].value_counts(normalize=True)
    neg_zone_dist = neg_examples_df["zone"].value_counts(normalize=True)
    overlap = min(pos_zone_dist.get(z, 0), neg_zone_dist.get(z, 0)) 
    total_overlap = sum(min(pos_zone_dist.get(z, 0), neg_zone_dist.get(z, 0)) for z in pos_zone_dist.index)
    assert total_overlap >= 0.4  # حداقل همپوشانی توزیع، آستانه قابل‌تنظیم بعد از دیدن داده واقعی
```
**چرا این تسک اضافه شد:** در نسخه ۱.۰، فایل `exs_all.pl` در فاز ۵ (تسک ۵.۱) به‌طور ضمنی فرض شده بود که از قبل آماده است، بدون این‌که هیچ تسکی مشخص کند با چه منطقی نمونه‌های منفی انتخاب می‌شوند. با توجه به این‌که سند فنی صریحاً این را «حساس‌ترین بخش ILP» می‌داند، این باید یک تسک مستقل با معیار پذیرش روشن باشد، نه یک فرض ضمنی.

---

## فاز ۳ — ساخت لایه Rule-based Feature Extractor

> **یادداشت روشن‌سازی (جدید):** Predicateهای هندسی این فاز (`half_space`, `zone14`, `under_pressure`) به‌صورت **توابع Python** پیاده‌سازی می‌شوند و خروجی‌شان به شکل **Fact ثابت** (مثلاً `at_zone(p1, seq5, right_half_space).`) در فایل `.pl` دامپ می‌شود — نه به شکل یک Prolog rule زنده و قابل‌فراخوانی مجدد در bk.pl. این یک تصمیم معماری آگاهانه‌ است (چون به بخش ۲.۲ سند فنی، لایه عصبی/محاسباتی MVP در Python پیاده می‌شود)، اما باید در bk.pl هم به‌وضوح مستند شود که این Predicateها «از پیش محاسبه‌شده» هستند، تا در فاز ۴ با Predicateهای تاکتیکی (که rule واقعی Prolog‌اند) اشتباه گرفته نشوند.

### تسک ۳.۱ — پیاده‌سازی Predicateهای هندسی پایه
**کار:** توابع Python برای محاسبه `half_space`, `zone14`, `under_pressure` و مفاهیم مشابه از روی مختصات SPADL و فریم 360 (طبق بخش ۲.۲ و ۴.۱ سند فنی).
**معیار پذیرش:** هر تابع روی حداقل ۳ ورودی دستی (مختصات شناخته‌شده) خروجی درست و منطقی بدهد (تست واحد ساده).
**تست:**
```python
def test_half_space_geometry():
    assert occupies_half_space(x=70, y=15) == "right_half_space"
    assert occupies_half_space(x=70, y=34) == "central_channel"
```

### تسک ۳.۲ — تولید Fact های Prolog از یک بازی نمونه
**کار:** تبدیل خروجی توابع هندسی به فایل `.pl` واقعی با Predicateهای Prolog معتبر، برای یک بازی نمونه.
**معیار پذیرش:** فایل تولیدشده در SWI-Prolog بدون خطای Syntax لود شود.
**تست:**
```python
def test_generated_facts_load_in_prolog(facts_file_path):
    import subprocess
    result = subprocess.run(["swipl", "-g", f"consult('{facts_file_path}'), halt."], capture_output=True)
    assert result.returncode == 0
```

### تسک ۳.۳ — تعمیم تولید Fact به کل دیتاست
**کار:** اجرای پایپ‌لاین فوق روی تمام بازی‌های بارسا، ادغام در یک یا چند فایل `.pl` نهایی.
**معیار پذیرش:** فایل نهایی شامل Factهایی برای تمام سکانس‌های برچسب‌خورده در فاز ۲ باشد؛ تعداد Factهای هر نوع با تعداد سکانس‌های ورودی هم‌خوانی داشته باشد.
**تست:**
```python
def test_fact_count_matches_sequences(final_facts_file, labeled_sequences_df):
    fact_count = count_predicate_occurrences(final_facts_file, "involved_in")
    assert fact_count == len(labeled_sequences_df)
```

---

## فاز ۴ — فرمول‌بندی دانش تاکتیکی مشترک (bk.pl)

این فاز، برخلاف فازهای قبل، **مستلزم مشارکت مستقیم و مستمر شما** است. روال هر تسک: شما مفهوم را به زبان ساده توصیف می‌کنید → با هم به Predicate منطقی تبدیل می‌کنیم → تست کوچک روی چند نمونه واقعی می‌زنیم → تأیید یا اصلاح.

### تسک ۴.۰ — ساخت مجموعه نمونه‌های Expert-labeled (جدید، پیش‌نیاز تسک‌های ۴.۱ تا ۴.۳)
**کار:** پیش از فرمول‌بندی هر مفهوم تاکتیکی، یک مجموعه‌ی مشخص از نمونه‌های واقعی (حداقل ۱۰ پاس برای Line-breaking، حداقل ۵ نمونه برای تردید، حداقل ۱۰ جفت برای supports/covers) را از داده واقعی استخراج و دستی برچسب بزنید. این مجموعه‌ها باید ذخیره و نسخه‌بندی شوند تا در تسک‌های بعدی به‌عنوان Ground Truth استفاده شوند.
**معیار پذیرش:** سه فایل/جدول مجزا (`expert_labeled_passes`, `hesitation_labeled_samples`, `expert_labeled_pairs`) ساخته و ذخیره شوند؛ هر کدام حداقل به تعداد ذکرشده در تسک متناظرشان نمونه داشته باشند.
**تست:**
```python
def test_expert_labeled_datasets_exist(expert_labeled_passes, hesitation_labeled_samples, expert_labeled_pairs):
    assert len(expert_labeled_passes) >= 10
    assert len(hesitation_labeled_samples) >= 5
    assert len(expert_labeled_pairs) >= 10
```
**چرا این تسک اضافه شد:** در نسخه ۱.۰، تسک‌های ۴.۱ تا ۴.۳ به مجموعه‌های `expert_labeled_passes`، `hesitation_labeled_samples` و `expert_labeled_pairs` ارجاع می‌دادند بدون اینکه مشخص باشد این مجموعه‌ها چه زمانی و چگونه ساخته می‌شوند. این یک وابستگی پنهان بود که اکنون به‌صورت یک تسک مستقل با معیار پذیرش روشن درآمده است.

### تسک ۴.۱ — فرمول‌بندی مفهوم "Line-breaking Pass"
**کار:** تعریف مشترک اینکه یک پاس چه شرایطی داشته باشد تا "خط‌شکن" محسوب شود؛ تبدیل به قانون Prolog.
**معیار پذیرش:** روی حداقل ۱۰ پاس واقعی از داده (از تسک ۴.۰)، تشخیص سیستم با تشخیص شما (به‌عنوان آنالیزور) حداقل در ۸ مورد یکسان باشد.
**تست:**
```python
def test_line_breaking_pass_matches_expert_labels(expert_labeled_passes):
    predicted = [is_line_breaking(p) for p in expert_labeled_passes["pass_data"]]
    actual = expert_labeled_passes["expert_label"]
    accuracy = sum(p == a for p, a in zip(predicted, actual)) / len(actual)
    assert accuracy >= 0.8
```

### تسک ۴.۲ — فرمول‌بندی مفهوم "تردید/مکث تحت فشار"
**کار:** تعریف آستانه مکث که نشان‌دهنده تردید هافبک/مدافع زیر فشار است. **به‌جای یک عدد ثابت زمانی**، از یک معیار نسبی استفاده شود (مثلاً درصدک زمان تصمیم‌گیری نسبت به میانگین در موقعیت‌های مشابه) تا حساسیت به سرعت کلی بازی یا شرایط خاص مسابقه کاهش یابد.
**معیار پذیرش:** آستانه اولیه تعیین و روی حداقل ۵ نمونه واقعی (از تسک ۴.۰) صحت‌سنجی شود؛ علاوه‌بر این، پایداری آستانه روی حداقل دو زیرمجموعه متفاوت داده (مثلاً نیمه اول در مقابل نیمه دوم بازی‌ها) بررسی و مستند شود.
**تست:**
```python
def test_hesitation_threshold_reasonable(hesitation_labeled_samples, threshold_percentile):
    predicted = hesitation_labeled_samples["pause_duration"] > hesitation_labeled_samples["contextual_baseline_percentile"]
    assert (predicted == hesitation_labeled_samples["expert_label"]).mean() >= 0.7

def test_threshold_stable_across_subsets(hesitation_labeled_samples_first_half, hesitation_labeled_samples_second_half, threshold_percentile):
    acc_first = evaluate_threshold(hesitation_labeled_samples_first_half, threshold_percentile)
    acc_second = evaluate_threshold(hesitation_labeled_samples_second_half, threshold_percentile)
    assert abs(acc_first - acc_second) <= 0.15  # پایداری معقول بین دو زیرمجموعه
```
**یادداشت اصلاحی:** نسخه ۱.۰ از یک آستانه مطلق زمانی استفاده می‌کرد که به سرعت بازی/فشار مسابقه حساس است؛ در این نسخه به یک معیار نسبی تبدیل شده و یک تست پایداری اضافه شده است.

### تسک ۴.۳ — فرمول‌بندی مفاهیم حمایتی/رابطه‌ای (`supports`, `covers`)
**کار:** تعریف قوانین geometric/context-based برای اینکه چه زمانی یک بازیکن دوم "حمایت‌گر" بازیکن اول محسوب می‌شود (طبق نمونه‌های بخش ۲.۵ سند فنی).
**معیار پذیرش:** روی حداقل ۱۰ نمونه دو‌نفره واقعی (از تسک ۴.۰)، تشخیص سیستم با قضاوت شما همخوانی معقول (۷۰٪+) داشته باشد.
**تست:**
```python
def test_support_relation_matches_expert(expert_labeled_pairs):
    predicted = [supports(p1, p2, seq) for p1, p2, seq in expert_labeled_pairs["pairs"]]
    accuracy = sum(p == e for p, e in zip(predicted, expert_labeled_pairs["expert_label"])) / len(expert_labeled_pairs)
    assert accuracy >= 0.7
```

### تسک ۴.۴ — تجمیع نهایی bk.pl
**کار:** ادغام تمام Predicateهای هندسی (فاز ۳، به‌صورت Fact از پیش محاسبه‌شده) و تاکتیکی (تسک‌های ۴.۱ تا ۴.۳، به‌صورت rule واقعی Prolog) در یک فایل `bk.pl` واحد و مستند. مستندسازی باید صریحاً مشخص کند کدام Predicateها Fact ثابت و کدام‌ها Rule پویا هستند.
**معیار پذیرش:** فایل کامل بدون خطا در SWI-Prolog لود شود و تمام Predicateهای موردنیاز `bias.pl` (فاز بعد) را پوشش دهد.
**تست:**
```python
def test_bkpl_loads_and_covers_required_predicates(bkpl_path, required_predicates):
    result = load_prolog_file(bkpl_path)
    assert result.success
    assert required_predicates.issubset(result.defined_predicates)
```

---

## فاز ۵ — تعریف بایاس و اجرای Popper

> **یادداشت معماری مهم (جدید):** این فاز را **خطی محض** در نظر نگیرید. طبق نکته پایانی سند فنی («فرآیند تکرارشونده و مرحله‌ای به‌جای فرمول‌بندی یک‌باره»، ریسک ۷)، وقتی Popper روی bk.pl واقعی اجرا شود، معمولاً معلوم می‌شود بعضی Predicateها بیش‌ازحد کلی یا محدودند. اگر تسک ۵.۱ یا ۵.۲ نتایج بی‌کیفیت یا Timeout مکرر داد، به فاز ۴ (رفاینمنت Predicateها) برگردید — این یک شکست نیست، بخشی طبیعی از فرآیند است.

### تسک ۵.۰ — Smoke Test کوچک Popper روی داده دستی (جدید)
**کار:** پیش از اجرای Popper روی یک بازی کامل، زنجیره Popper + Clingo + SWI-Prolog را روی یک مجموعه بسیار کوچک (۵ تا ۱۰ Fact دستی‌ساخته، نه واقعی) با یک `bias.pl` حداقلی تست کنید. هدف صرفاً اطمینان از سلامت فنی زنجیره است، نه کیفیت قانون.
**معیار پذیرش:** Popper روی این مجموعه کوچک در کمتر از ۳۰ ثانیه بدون خطا اجرا شود و حداقل یک قانون (حتی بی‌معنی از نظر تاکتیکی) تولید کند.
**تست:**
```python
def test_popper_smoke_test_minimal_data():
    result = run_popper(bk="bk_toy.pl", bias="bias_toy.pl", exs="exs_toy.pl", timeout=30)
    assert result.completed
    assert len(result.rules) >= 1
```
**چرا این تسک اضافه شد:** اجرای مستقیم روی یک بازی کامل (تسک ۵.۱ در نسخه ۱.۰) هزینه دیباگ بالایی دارد اگر مشکل از تنظیمات پایه‌ی زنجیره باشد، نه از خود داده. این تسک هزینه دیباگ را با یک آزمایش تقریباً رایگان جدا می‌کند.

### تسک ۵.۱ — نوشتن `bias.pl` نسخه اول (محدود)
**کار:** تعریف `head_pred`, `body_pred`, `max_vars`, `max_body`، و سقف تعداد بازیکنان (حداکثر ۲ برای شروع، طبق راهکار ریسک ۱ سند فنی).
**معیار پذیرش:** Popper با این bias، بدون Timeout (زیر ۵ دقیقه) روی یک زیرمجموعه کوچک (یک بازی) اجرا شود.
**تست:**
```python
def test_popper_runs_within_timeout_small_sample():
    result = run_popper(bk="bk.pl", bias="bias.pl", exs="exs_single_match.pl", timeout=300)
    assert result.completed
```

### تسک ۵.۲ — اجرای Popper روی کل دیتاست (سطح تک‌بازیکنی)
**کار:** اجرای Popper با `head_pred` تک‌متغیره (`lose_possession/1`) روی کل مجموعه Fact.
**معیار پذیرش:** حداقل یک قانون خروجی تولید شود؛ زمان اجرا مستند و قابل قبول باشد (زیر یک ساعت برای این حجم داده). **معیار توقف/بازگشت:** اگر اجرا Timeout بخورد یا صفر قانون تولید کند، پیش از تلاش مجدد باید `bias.pl` (کاهش max_body یا max_vars) یا `bk.pl` (ساده‌سازی Predicateهای پرهزینه) بازبینی شود — تکرار کورکورانه بدون تغییر تنظیمات مجاز نیست.
**تست:**
```python
def test_popper_single_player_rules_generated():
    result = run_popper(bk="bk.pl", bias="bias_single.pl", exs="exs_all.pl", timeout=3600)
    assert len(result.rules) >= 1
```

### تسک ۵.۳ — افزایش سقف به دو‌بازیکنی و اجرای مجدد
**کار:** تغییر `head_pred` به دو‌متغیره (`lose_possession/2`)، افزایش سقف بازیکنان طبق بخش ۲.۵ سند فنی، اجرای مجدد.
**معیار پذیرش:** اجرا در زمان معقول (مطابق تنظیم Timeout مصوب در ریسک ۱) کامل شود؛ حداقل یک قانون دو‌نفره تولید شود. **معیار توقف/بازگشت:** همانند تسک ۵.۲ — اگر انفجار ترکیبیاتی رخ داد (Timeout مکرر در حداقل ۲ تلاش پیاپی)، به فاز ۴ برای کاهش تعداد یا کلی‌بودن Predicateهای body_pred بازگردید، پیش از افزایش بیشتر پیچیدگی.
**تست:**
```python
def test_popper_two_player_rules_generated():
    result = run_popper(bk="bk.pl", bias="bias_pairwise.pl", exs="exs_all.pl", timeout=3600)
    two_player_rules = [r for r in result.rules if r.arity == 2]
    assert len(two_player_rules) >= 1
```

### تسک ۵.۴ — چک‌پوینت بازخورد فاز ۴ ↔ فاز ۵ (جدید)
**کار:** بعد از هر دو اجرای اصلی Popper (تسک ۵.۲ و ۵.۳)، صرف‌نظر از موفقیت یا شکست، یک مرور صریح و مستند از کیفیت خروجی انجام دهید: آیا قوانین تولیدشده از نظر تاکتیکی معنادارند یا صرفاً از نظر آماری صحیح‌اند؟ آیا Predicateهای bk.pl بیش‌ازحد کلی (قانون‌های بی‌معنی زیاد تولید می‌کنند) یا بیش‌ازحد محدود (هیچ قانونی تولید نمی‌شود) هستند؟ نتیجه این مرور را در یک یادداشت کوتاه ثبت کنید و تصمیم بگیرید: ادامه به فاز ۶، یا بازگشت به فاز ۴ برای رفاینمنت یک یا چند Predicate مشخص.
**معیار پذیرش:** یک سند/یادداشت چک‌پوینت برای هر اجرای اصلی (۵.۲ و ۵.۳) ثبت شود که شامل: تعداد قوانین تولیدشده، ارزیابی کیفی اولیه (معنادار/نامعتبر/نامشخص) حداقل روی چند نمونه، و تصمیم صریح "ادامه" یا "بازگشت به فاز ۴ — Predicate(های) X". اگر تصمیم بازگشت باشد، تسک نباید به فاز ۶ برود تا این چک‌پوینت به‌روزرسانی و به "ادامه" تغییر یابد.
**تست:**
```python
def test_feedback_checkpoint_documented(popper_run_checkpoint):
    assert popper_run_checkpoint["rule_count"] is not None
    assert popper_run_checkpoint["quality_assessment"] in ["meaningful", "invalid", "unclear"]
    assert popper_run_checkpoint["decision"] in ["proceed", "return_to_phase_4"]
    if popper_run_checkpoint["decision"] == "return_to_phase_4":
        assert len(popper_run_checkpoint["predicates_to_refine"]) > 0
```
**چرا این تسک اضافه شد:** بدون یک چک‌پوینت رسمی، فرض خطی‌بودن فاز ۴→۵ به‌راحتی دوباره جا می‌افتد، چون هیچ تسکی صراحتاً از شما نمی‌خواهد کیفیت را بین دو اجرا ارزیابی و تصمیم بازگشت/ادامه را ثبت کنید. این تسک، توصیه ریسک ۷ سند فنی («فرآیند تکرارشونده») را از یک یادداشت کلی به یک گیت واقعی با معیار پذیرش تبدیل می‌کند.

---

## فاز ۶ — فیلتر، ارزیابی، Hold-out و تفسیر قوانین

### تسک ۶.۰ — تفکیک Hold-out درون‌فصلی (جدید)
**کار:** طبق ریسک ۶ سند فنی («Overfitting به سبک بازی یک تیم خاص»)، پیش از محاسبه Support/Confidence نهایی، دیتاست را به دو بخش تقسیم کنید: مثلاً ۸۰٪ بازی‌ها برای کشف قانون (فاز ۵) و ۲۰٪ بازی‌های باقی‌مانده (که Popper هرگز آن‌ها را ندیده) برای اعتبارسنجی مجدد Support/Confidence هر قانون کشف‌شده.
**معیار پذیرش:** هر قانون خروجی فاز ۵ باید مقادیر Support/Confidence را هم روی داده آموزشی و هم روی Hold-out داشته باشد؛ افت شدید (مثلاً بیش از ۵۰٪) در Confidence روی Hold-out باید علامت‌گذاری شود.
**تست:**
```python
def test_holdout_confidence_reported(rules_with_metrics):
    for r in rules_with_metrics:
        assert "confidence_train" in r and "confidence_holdout" in r
        if r["confidence_train"] > 0:
            drop_ratio = (r["confidence_train"] - r["confidence_holdout"]) / r["confidence_train"]
            r["flagged_for_review"] = drop_ratio > 0.5
```
**چرا این تسک اضافه شد:** ریسک ۶ سند فنی به این موضوع اشاره دارد ولی هیچ تسکی در نسخه ۱.۰ آن را عملیاتی نمی‌کرد. برای MVP یک تست کامل روی فصل/تیم دیگر لازم نیست، اما حداقل یک Hold-out درون‌فصلی هزینه کمی دارد و اعتبار قوانین را تقویت می‌کند.

### تسک ۶.۱ — محاسبه Support و Confidence برای هر قانون خروجی
**کار:** پیاده‌سازی محاسبه دو معیار طبق بخش ۶ سند فنی، روی تمام قوانین خروجی فاز ۵ (روی داده آموزشی؛ نسخه Hold-out طبق تسک ۶.۰ جداگانه محاسبه می‌شود).
**معیار پذیرش:** هر قانون یک مقدار Support و Confidence عددی داشته باشد؛ قوانین با Support کمتر از آستانه (۳ تا ۵) به‌صورت خودکار حذف شوند.
**تست:**
```python
def test_support_confidence_computed_and_filtered(raw_rules_with_metrics):
    assert all("support" in r and "confidence" in r for r in raw_rules_with_metrics)
    assert all(r["support"] >= 3 for r in raw_rules_with_metrics if r["kept"])
```

### تسک ۶.۲ — محاسبه Lift نسبت به Baseline تیم
**کار:** پیاده‌سازی معیار Lift طبق بخش ۲.۷ سند فنی، برای تفکیک الگوهای واقعاً پرخطر از الگوهای سالم پرتکرار.
**معیار پذیرش:** (الف) روی یک الگوی **synthetic (مصنوعی، ساخته‌شده دستی)** با Lift نزدیک به ۱ از پیش مشخص، پیاده‌سازی محاسبه Lift مقدار درست را برگرداند؛ (ب) در صورت وجود یک الگوی شناخته‌شده سالم در داده واقعی (مثلاً کمک هافبک دفاعی)، Lift آن نیز نزدیک به ۱ باشد.
**تست:**
```python
def test_lift_calculation_correct_on_synthetic_case(synthetic_pattern_stats):
    # مورد ساختگی با نرخ پیامد منفی برابر نرخ پایه تیم -> Lift باید ~1 باشد
    lift = compute_lift(synthetic_pattern_stats)
    assert abs(lift - 1.0) < 0.1

def test_healthy_pattern_has_low_lift_if_present(known_healthy_pattern_stats):
    if known_healthy_pattern_stats is not None:
        assert known_healthy_pattern_stats["lift"] < 1.3
```
**یادداشت اصلاحی:** تست نسخه ۱.۰ کاملاً به ظهور تصادفی یک الگوی سالم در داده واقعی وابسته بود و ممکن بود هیچ‌وقت اجرا نشود. حالا یک تست synthetic اجباری اضافه شده که مستقل از داده واقعی صحت محاسبه Lift را تضمین می‌کند؛ تست روی داده واقعی به‌صورت شرطی باقی می‌ماند.

### تسک ۶.۳ — جلسه Coach Validation (تأیید انسانی)
**کار:** مرور دستی تمام قوانین باقی‌مانده توسط شما؛ ثبت تأیید/رد هر کدام با یادداشت دلیل.
**معیار پذیرش:** هر قانون خروجی این فاز دارای وضعیت مشخص (`approved` / `rejected`) و یادداشت باشد.
**تست:**
```python
def test_all_rules_have_validation_status(final_rules_df):
    assert final_rules_df["validation_status"].isin(["approved", "rejected"]).all()
```

---

## فاز ۷ — نمایش بصری و گزارش نهایی

### تسک ۷.۱ — رسم نمونه‌ای از هر قانون تأییدشده روی نقشه زمین
**کار:** استفاده از mplsoccer برای رسم حداقل یک نمونه واقعی از هر قانون approved، با مسیر توپ و موقعیت بازیکنان کلیدی.
**معیار پذیرش:** هر قانون approved حداقل یک تصویر رسم‌شده داشته باشد.
**تست:**
```python
def test_visualization_exists_for_each_approved_rule(approved_rules, output_dir):
    for rule_id in approved_rules["rule_id"]:
        assert (output_dir / f"rule_{rule_id}.png").exists()
```

### تسک ۷.۲ — ترجمه قانون منطقی به توضیح قابل‌فهم
**کار:** پیاده‌سازی `logic_translator.py` که هر قانون Prolog را به یک جمله فارسی/انگلیسی قابل‌فهم برای مربی تبدیل می‌کند.
**معیار پذیرش:** هر قانون approved یک متن توضیحی داشته باشد که حداقل شامل بازیکن(ها)، شرایط، و پیامد باشد.
**تست:**
```python
def test_translation_contains_key_elements(translated_rules):
    for t in translated_rules:
        assert t["player_mentioned"] and t["condition_mentioned"] and t["outcome_mentioned"]
```

### تسک ۷.۳ — تجمیع گزارش نهایی
**کار:** ساخت گزارش نهایی (سند یا صفحه) شامل تمام قوانین approved، با تصویر، توضیح، و آمار (Support/Confidence/Lift/Hold-out).
**معیار پذیرش:** گزارش نهایی شامل حداقل ۳ الگو باشد (طبق معیار موفقیت MVP در بخش ۱.۵ سند فنی).
**تست:**
```python
def test_final_report_meets_mvp_criteria(final_report):
    assert len(final_report["patterns"]) >= 3
```

---

## فاز ۸ — مستندسازی و بسته‌بندی MVP

### تسک ۸.۱ — مستندسازی کد و پایپ‌لاین
**کار:** README کامل برای اجرای مجدد کل پایپ‌لاین از صفر، شامل نحوه اجرای هر فاز.
**معیار پذیرش:** یک نفر ثالث (یا خود شما بعد از مدتی) بتواند صرفاً با خواندن README، پایپ‌لاین را از ابتدا اجرا کند.

### تسک ۸.۲ — جمع‌بندی یافته‌ها و ثبت محدودیت‌های شناخته‌شده
**کار:** سند کوتاه جمع‌بندی: چه الگوهایی کشف شد، کدام رد شدند و چرا، چه محدودیت‌هایی (طبق بخش ۲.۴ سند فنی درباره داده 360) روی نتایج اثر گذاشت.
**معیار پذیرش:** سند شامل حداقل یک بخش صریح "محدودیت‌های این نسخه" باشد.

### تسک ۸.۳ — برنامه‌ریزی اولیه فاز توسعه
**کار:** بر اساس آنچه در MVP یاد گرفته شد (به‌خصوص سقف واقعی تعداد بازیکنان و Timeout Popper)، پیش‌نویس اهداف فاز توسعه (Set Pieces، احتمالاً Tracking Data واقعی در صورت تخصیص بودجه).
**معیار پذیرش:** سند پیش‌نویس فاز توسعه آماده و برای بررسی شما ارائه شود.

---

## نکته پایانی درباره نحوه استفاده از این فازبندی

- هر تسک را به‌ترتیب انجام دهید؛ تست pytest متناظر را بنویسید و اجرا کنید؛ فقط بعد از سبز شدن تست، تسک را تیک بزنید و به بعدی بروید.
- فاز ۴ کندتر از بقیه پیش می‌رود چون نیازمند مشارکت مستقیم شماست — این طبیعی و پیش‌بینی‌شده است (طبق ریسک ۷ سند فنی).
- **فاز ۴ و فاز ۵ را خطی تصور نکنید.** تسک ۵.۴ دقیقاً برای همین ساخته شده: بعد از هر اجرای اصلی Popper، رسماً کیفیت را ارزیابی و تصمیم "ادامه/بازگشت" را ثبت کنید. بازگشت به فاز ۴ برای رفاینمنت Predicateها بخشی طبیعی از فرآیند ILP است، نه شکست در برنامه‌ریزی.
- اگر در هر فاز با یک مانع غیرمنتظره مواجه شدید (مثلاً Timeout مکرر Popper، یا پوشش پایین‌تر از حد انتظار داده 360)، به‌جای پیش‌روی اجباری، به‌روزرسانی سند فنی (بخش ریسک‌ها) و بازنگری در بایاس یا دامنه توصیه می‌شود.



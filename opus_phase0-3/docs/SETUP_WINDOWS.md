# راهنمای راه‌اندازی و تست روی ویندوز

این سند فرض می‌کند ویندوز ۱۰ یا ۱۱ (۶۴ بیتی) دارید و از **PowerShell** استفاده
می‌کنید (نه CMD). هر بخش یک مرحله است؛ بعد از هر مرحله یک «تست تأیید» هست که
باید سبز شود، بعد بروید مرحله بعد.

---

## گام ۰ — ابزارهای پایه

### ۰.۱ نصب uv (مدیر پکیج)

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

بعد از نصب، PowerShell را **ببندید و دوباره باز کنید**، سپس:

```powershell
uv --version
```

### ۰.۲ نصب SWI-Prolog (نسخه ۹.۲ یا بالاتر — الزامی)

> ⚠️ نسخه‌های قدیمی‌تر از **9.1.12** کار نمی‌کنند. کتابخانه `janus_swi` که
> Popper به آن وابسته است، در زمان نصب نسخه SWI-Prolog را چک می‌کند و با
> نسخه پایین‌تر خطای `At least SWI-Prolog version 9.1.12 is required` می‌دهد.
> (این دقیقاً همان خطایی بود که در محیط تست من هم رخ داد و مجبور شدم
> SWI-Prolog را از سورس build کنم.)

ساده‌ترین راه:

```powershell
winget install --id SWI-Prolog.SWI-Prolog -e
```

یا نصب دستی از `https://www.swi-prolog.org/download/stable` (فایل
`swipl-x.y.z-1.x64.exe`). در حین نصب، گزینه **"Add swipl to the system PATH"**
را حتماً تیک بزنید.

**تست تأیید:**

```powershell
swipl --version
swipl -g "X is 2+3, format('~w~n',[X]), halt."
```

باید نسخه ≥ 9.1.12 و عدد `5` را ببینید. اگر `swipl` شناخته نشد، مسیر زیر را
دستی به PATH اضافه کنید:

```powershell
$env:Path += ";C:\Program Files\swipl\bin"
```

(برای دائمی شدن: System Properties → Environment Variables)

### ۰.۳ نصب Git

```powershell
winget install --id Git.Git -e
```

---

## گام ۱ — آماده‌سازی پروژه

فایل پروژه را در مسیر دلخواه باز کنید، مثلاً `C:\projects\football-tactical-ilp`.

```powershell
cd C:\projects\football-tactical-ilp
uv sync --extra dev
```

`uv` خودش Python مناسب (۳.۱۱ یا ۳.۱۲) را دانلود و یک محیط مجازی در `.venv`
می‌سازد. نیازی به `pip install` جداگانه نیست.

> **نکته مهم درباره یک باگ وابستگی:** `socceraction 1.5.3` به `pandera 0.17`
> وابسته است که با `multimethod >= 2.0` می‌شکند
> (`ImportError: cannot import name 'overload' from 'multimethod'`).
> در `pyproject.toml` این مورد با `multimethod<2.0` پین شده. اگر روزی دستی
> پکیج‌ها را آپدیت کردید و این خطا را دیدید، علتش همین است.

**تست تأیید:**

```powershell
uv run python -c "import statsbombpy, socceraction, pandas, numpy, mplsoccer; print('ok')"
```

---

## گام ۲ — نصب Popper

Popper باید **کنار** پروژه کلون شود، نه داخل آن (محیط پایتون جداگانه‌ای دارد:
Popper به Python 3.14 نیاز دارد، پروژه ما به 3.11/3.12).

```powershell
cd C:\projects\football-tactical-ilp
mkdir external -Force
cd external
git clone https://github.com/logic-and-learning-lab/Popper.git
cd Popper
uv run popper.py examples/iggp-rps-next-score
```

اجرای اول چند دقیقه طول می‌کشد چون `uv` باید Python 3.14 و وابسته‌های Popper
(`clingo`, `janus_swi`, `ortools`, ...) را دانلود کند.

**خبر خوب برای ویندوز:** `janus_swi` روی PyPI یک wheel آماده‌ی
`cp314-win_amd64` دارد، پس روی ویندوز **نیازی به Visual Studio Build Tools
ندارید** — برخلاف لینوکس که باید از سورس کامپایل شود.

**خروجی موفق** چیزی شبیه این است (روی stderr چاپ می‌شود):

```
2.1s ********** SOLUTION **********
2.1s Precision:1.00 Recall:1.00 TP:108 FN:0 TN:356 FP:0 Size:16
2.1s next_score(V0,V1,V2):- my_true_score(V0,V1,V2),different(V1,V3),...
```

سپس متغیر محیطی را ست کنید تا تست‌های پروژه Popper را پیدا کنند:

```powershell
cd C:\projects\football-tactical-ilp
$env:POPPER_DIR = "C:\projects\football-tactical-ilp\external\Popper"
```

(اگر Popper را در `external/Popper` گذاشته باشید، پروژه خودش پیدایش می‌کند و
این متغیر لازم نیست.)

---

## گام ۳ — بررسی کامل محیط (فاز ۰)

```powershell
uv run python scripts/run_phase0_check.py
```

خروجی مورد انتظار:

```
[ OK ] Python >= 3.11
[ OK ] core analysis packages
[ OK ] socceraction / pandera stack healthy
[ OK ] uv package manager
[ OK ] SWI-Prolog >= 9.1.12
[ OK ] Popper checkout
[ OK ] network access to StatsBomb
all checks passed — you can run scripts/run_phase1.py
```

---

## گام ۴ — اجرای فازها

### فاز ۱ — داده خام

```powershell
uv run python scripts/run_phase1.py
```

⏱ اجرای اول **۱۰ تا ۲۵ دقیقه** طول می‌کشد (بسته به سرعت اینترنت): ۳۵ بازی
Event + Lineup + داده ۳۶۰ دانلود و در `data\raw\` کش می‌شود (حدود ۵۰۰ مگابایت).
اجرای دوم چند ثانیه است.

**اعداد مورد انتظار:**

| مورد | مقدار |
|---|---|
| بازی‌های بارسا | ۳۵ |
| بازی‌های دارای داده ۳۶۰ | ۳۵/۳۵ |
| کل اکشن‌های SPADL | ۸۲٬۰۴۸ |
| پوشش ۳۶۰ روی اکشن‌های Build-up | ۹۲.۶۵٪ |
| بازیکنان مرجع | ۲۵ (۲۴ نفر core) |
| میانگین x شوت‌های بارسا | ~۹۲.۵ |
| میانگین x شوت‌های حریف | ~۱۳.۴ |

### فاز ۲ — سکانس‌ها و نمونه‌گیری

```powershell
uv run python scripts/run_phase2.py
```

**اعداد مورد انتظار:**

| مورد | مقدار |
|---|---|
| سکانس‌های Build-up | ۱۰۶۶ |
| طول min/median/max | ۲ / ۴ / ۲۱ |
| برچسب `negative` | ۱۹۱ (۱۷.۹۲٪) |
| برچسب `non_negative` | ۸۷۵ |
| لینک‌های پاس | ۲۱۸۱ |
| زنجیره‌های سه‌نفره | ۱۰۱۱ |
| سکانس‌های دارای جفت بازیکن | ۸۷.۲۴٪ |
| نمونه‌های pos / neg | ۱۹۱ / ۳۸۲ |
| همپوشانی ناحیه‌ای pos-neg | ۱.۰۰۰ |

### فاز ۳ — تولید Prolog Facts

```powershell
uv run python scripts/run_phase3.py
```

**اعداد مورد انتظار:**

| مورد | مقدار |
|---|---|
| فایل نمونه | `data\prolog\facts_match_3764440.pl` (۲۱۱۲ fact) |
| فایل کامل | `data\prolog\facts_all.pl` (۸۶٬۴۹۶ fact) |
| `swipl consult` | **clean** (exit 0، بدون Warning) |

---

## گام ۵ — اجرای تست‌ها (مهم‌ترین بخش برای شما)

کل مجموعه تست:

```powershell
uv run pytest -v -m "not network"
```

انتظار: **۵۸ تست سبز**.

تست به تفکیک فاز:

```powershell
uv run pytest -v tests/test_phase0_environment.py    # فاز ۰ (۹ تست، ۱ تای آن network)
uv run pytest -v tests/test_phase1_data.py           # فاز ۱ (۱۲ تست)
uv run pytest -v tests/test_phase2_sequences.py      # فاز ۲ (۲۰ تست)
uv run pytest -v tests/test_phase3_features.py       # فاز ۳ (۱۸ تست)
```

تست یک معیار پذیرش مشخص (مثلاً پوشش ۳۶۰):

```powershell
uv run pytest -v -k "360_merge_coverage"
uv run pytest -v -k "label_balance"
uv run pytest -v -k "negative_samples_context_comparable"
uv run pytest -v -k "generated_facts_load_in_prolog"
```

تست‌هایی که به اینترنت نیاز دارند (فقط برای بار اول، پیش از ساخت کش):

```powershell
uv run pytest -v -m network
```

---

## گام ۶ — بررسی دستی خروجی Prolog

اینجا می‌توانید مستقل از پایتون مطمئن شوید که خروجی واقعاً درست است:

```powershell
swipl -g "consult('data/prolog/facts_all.pl'), aggregate_all(count, sequence(_), N), format('sequences=~w~n',[N]), halt."
```

باید `sequences=1066` بدهد.

چند پرس‌وجوی تاکتیکی برای اطمینان از معنادار بودن داده:

```powershell
# چند سکانس منفی شامل یک اکشن زیر فشار بوده‌اند؟
swipl -g "consult('data/prolog/facts_all.pl'), aggregate_all(count, (sequence_label(S,negative), in_sequence(A,S), under_pressure(A)), N), format('~w~n',[N]), halt."

# بوسکتس در چند سکانس منفی حضور داشته است؟
swipl -g "consult('data/prolog/facts_all.pl'), aggregate_all(count, (involved_in(p_5203,S), sequence_label(S,negative)), N), format('~w~n',[N]), halt."

# پرتکرارترین نقش‌ها در سکانس‌های منفی
swipl -g "consult('data/prolog/facts_all.pl'), forall(has_role(P,R), (aggregate_all(count,(involved_in(P,S),sequence_label(S,negative)),N), format('~w ~w ~w~n',[P,R,N]))), halt."
```

و برای دیدن ساختار فایل:

```powershell
Get-Content data\prolog\facts_all.pl -TotalCount 60
Select-String -Path data\prolog\facts_all.pl -Pattern "^passes_to" | Select-Object -First 5
```

---

## مشکلات رایج

| علامت | علت | راه‌حل |
|---|---|---|
| `swipl` شناخته نمی‌شود | PATH ست نشده | `C:\Program Files\swipl\bin` را به PATH اضافه کنید |
| `At least SWI-Prolog version 9.1.12 is required` | SWI-Prolog قدیمی | نسخه ۹.۲+ نصب کنید |
| `cannot import name 'overload' from 'multimethod'` | `multimethod>=2` نصب شده | `uv sync --extra dev` را دوباره بزنید |
| دانلود فاز ۱ نصفه می‌ماند | قطعی اینترنت | دوباره اجرا کنید؛ از کش ادامه می‌دهد |
| تست‌ها `skip` می‌شوند با پیام `run scripts/run_phaseN.py first` | کش ساخته نشده | ابتدا اسکریپت فاز مربوطه را اجرا کنید |
| Popper خروجی روی صفحه ندارد | Popper روی **stderr** لاگ می‌دهد | `uv run popper.py ... 2>&1` |
| دسترسی به GitHub/StatsBomb بسته است | فیلترینگ | از یک اتصال باز استفاده کنید؛ داده از `raw.githubusercontent.com` می‌آید |

---

## نکته درباره حجم دیسک

| پوشه | حجم تقریبی |
|---|---|
| `data\raw\` | ~۵۰۰ MB (کش دانلود) |
| `data\processed\` | ~۵۰ MB |
| `data\prolog\` | ~۴ MB |
| `.venv\` | ~۵۰۰ MB |
| `external\Popper\` + محیطش | ~۸۰۰ MB |

پوشه‌های `data\` و `.venv\` در `.gitignore` هستند و قابل بازتولیدند.

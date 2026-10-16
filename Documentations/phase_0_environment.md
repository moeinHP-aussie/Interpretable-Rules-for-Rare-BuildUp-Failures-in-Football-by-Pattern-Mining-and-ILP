# فاز ۰ — محیط اجرا و پیش‌نیازهای زیرساختی

## هدف و جایگاه فاز

فاز ۰ پیش از پردازش داده اجرا می‌شود تا ناسازگاری‌های محیط توسعه، ابزارهای Prolog/ILP و دسترسی به منبع داده را در یک نوبت آشکار کند. این فاز داده را تبدیل نمی‌کند و مدل یا قانون تاکتیکی نمی‌سازد؛ کار آن تشخیص آمادگی محیط برای اجرای زنجیره است. پیاده‌سازی اصلی در [`scripts/run_phase0_check.py`](../scripts/run_phase0_check.py) است و تست‌های آن در [`tests/test_phase0_environment.py`](../tests/test_phase0_environment.py) قرار دارند.

اجرای معمول از ریشه‌ی پکیج:

```powershell
uv run python scripts/run_phase0_check.py
```

کد برای هر بررسی یک نتیجه‌ی مستقل چاپ می‌کند (`[ OK ]`، `[FAIL]` یا در تست‌های pytest گاهی skip). به‌جای توقف در اولین مشکل، همه‌ی بررسی‌ها را اجرا می‌کند و در پایان با کد خروجی `0` در صورت موفقیت کامل و `1` در صورت وجود حداقل یک شکست پایان می‌یابد. در موفقیت، پیام بعدی اجرای `scripts/run_phase1.py` را پیشنهاد می‌کند.

## معماری و روش اجرا

### مسیر واردسازی پکیج

در ابتدای اسکریپت، مسیر `src` که نسبت به محل فایل محاسبه می‌شود به ابتدای `sys.path` افزوده می‌شود. بنابراین اسکریپت در حالت اجرای مستقیم از ریشه‌ی مخزن می‌تواند `tactical_ilp` را وارد کند، حتی اگر پکیج در محیط به‌شکل editable نصب نشده باشد. ریشه‌ی واقعی پروژه و مسیر داده‌ها در [`src/tactical_ilp/config.py`](../src/tactical_ilp/config.py) تعریف می‌شوند.

### گزارش خطا به‌ازای هر چک

تابع `check(label, fn)` تابع بررسی را جداگانه اجرا می‌کند. استثناها گرفته می‌شوند و به متن نتیجه تبدیل می‌شوند؛ در نتیجه خطای یک وابستگی مانع گزارش سایر پیش‌نیازها نمی‌شود. جزئیات استثنا با نام نوع و پیام چاپ می‌شوند. نتیجه‌ی بولی هر چک برای محاسبه‌ی تعداد شکست‌ها نگهداری می‌شود.

### بررسی Python

تابع `_python()` نسخه‌ی جاری را از `sys.version_info` می‌خواند و `Python >= 3.11` را می‌سنجد. پیام جزئیات شامل نسخه‌ی Python و سیستم‌عامل است. قید بسته‌بندی در [`pyproject.toml`](../pyproject.toml) دقیق‌تر است: `requires-python = ">=3.11,<3.13"`؛ بنابراین نصب وابستگی‌ها برای Python 3.11 و 3.12 تعریف شده، گرچه چک اجرایی فقط حداقل 3.11 را اعلام می‌کند. توسعه‌دهنده باید این تفاوت را بداند و از بازه‌ی مجاز پیکربندی بسته پیروی کند.

### بررسی کتابخانه‌های تحلیل

تابع `_packages()` کتابخانه‌های اصلی را import می‌کند: `matplotlib`، `mplsoccer`، `numpy`، `pandas`، `socceraction` و `statsbombpy`. موفقیت import معیار اصلی سلامت این گروه است؛ نسخه‌ی pandas، numpy و socceraction نیز در خروجی گزارش می‌شود.

این تابع ابتدا متغیر محیطی `MPLBACKEND` را حذف می‌کند و سپس backend را به `Agg` تغییر می‌دهد. در یک فرایند headless ممکن است متغیر به backend داخلی Colab (`module://matplotlib_inline.backend_inline`) اشاره کند که در محیط محلی نصب نیست؛ Matplotlib این مقدار را هنگام import اعتبارسنجی می‌کند، پیش از آن‌که `matplotlib.use()` فرصت جایگزینی داشته باشد. حذف متغیر و انتخاب اجباری `Agg` این وابستگی ناخواسته به نوت‌بوک را برطرف می‌کند و برای اجرای بدون پنجره‌ی گرافیکی مناسب است. رفتار در [`tests/test_phase0_environment.py`](../tests/test_phase0_environment.py) با تنظیم مصنوعی `MPLBACKEND` نیز پوشش داده شده است.

### سازگاری زنجیره‌ی socceraction / pandera

تابع `_pandera_stack()` بررسی می‌کند که `socceraction.spadl.statsbomb.convert_to_actions` وجود داشته باشد و نسخه‌ی نصب‌شده‌ی `multimethod` از metadata بسته خوانده شود. نسخه‌ی اصلی باید کمتر از ۲ باشد. علت فنی این محدودیت، ناسازگاری پشته‌ی pandera مورداستفاده‌ی socceraction با `multimethod >= 2.0` است؛ محدودیت در dependencies فایل [`pyproject.toml`](../pyproject.toml) نیز به‌شکل `multimethod<2.0` تثبیت شده است. تست `test_socceraction_pandera_stack_is_healthy` هم import loader و converter را کنترل می‌کند.

### بررسی uv

تابع `_uv()` محل اجرایی `uv` را با `shutil.which` پیدا می‌کند و در صورت وجود، `uv --version` را اجرا می‌کند. وجود ابزار شرط لازم است چون راهنمای پروژه نصب و اجرا را با `uv sync` و `uv run` انجام می‌دهد. نبودن `uv` در PATH به‌عنوان شکست گزارش می‌شود.

### بررسی SWI-Prolog

تابع `_swipl()` محل اجرایی `swipl` را از PATH پیدا کرده و `swipl --version` را اجرا می‌کند. نسخه به اجزای عددی major/minor/patch شکسته می‌شود و حداقل `(9, 1, 12)` لازم است؛ این حد برای پل Prolog موردنیاز Popper/`janus_swi` است. نبودن executable یا نسخه‌ی پایین‌تر شکست است. در تست‌ها، نبود SWI-Prolog موجب skip می‌شود و در صورت وجود هم اجرای یک هدف ساده (`X is 2+3`) بررسی می‌شود؛ بنابراین تست، صرفاً وجود فایل اجرایی را نمی‌سنجد.

### پیدا کردن Popper

تابع `_popper()` مسیر checkout مربوط به Popper را به این ترتیب جست‌وجو می‌کند:

1. مقدار متغیر محیطی `POPPER_DIR`، اگر تنظیم شده باشد؛
2. `external/Popper` داخل ریشه‌ی پکیج؛
3. پوشه‌ی `Popper` در پوشه‌ی والد پکیج.

پوشه فقط وقتی معتبر است که فایل `popper.py` در ریشه‌ی آن وجود داشته باشد. اگر پیدا نشود، راهنمای خروجی تنظیم `POPPER_DIR` یا clone در `external/Popper` را می‌دهد. تست نصب Popper با marker `slow` مشخص شده و در صورت نبود checkout یا uv skip می‌شود؛ در صورت امکان مثال `examples/iggp-rps-next-score` را با سقف زمانی ۱۸۰۰ ثانیه اجرا کرده و وجود `SOLUTION` را می‌سنجد. در نتیجه چک سبک Phase 0 وجود checkout را می‌سنجد، ولی تست کندتر قابلیت اجرای مثال Popper را.

### دسترسی شبکه به داده‌ی عمومی

تابع `_network()` با `urllib.request.urlopen` نشانی فایل فهرست رقابت‌های StatsBomb Open Data را می‌خواند و حداکثر ۳۰ ثانیه منتظر می‌ماند. پاسخ HTTP 200 موفقیت محسوب می‌شود. این آزمون دسترسی عمومی به endpoint را کنترل می‌کند؛ اعتبارسنجی کامل دانلود همه‌ی بازی‌ها در اینجا انجام نمی‌شود. تست متناظر pytest با marker `network` علامت‌گذاری شده است، زیرا فقط برای محیط تازه/دسترسی زنده معنا دارد.

## خروجی‌ها و معنای شکست

ترتیب فهرست بررسی‌ها در `main()` چنین است: نسخه Python، import بسته‌های تحلیل، سلامت socceraction/multimethod، uv، SWI-Prolog، checkout Popper و دسترسی شبکه. اجرای هر مورد مستقل است و شمارنده‌ی شکست در پایان تصمیم کد خروجی را می‌گیرد.

چک‌ها راه‌حل خودکار اعمال نمی‌کنند و چیزی نصب یا دانلود نمی‌کنند. اگر شکست رخ داد، جزئیات همان ردیف و راهنمای محیطی را بررسی کنید؛ راهنمای Colab در [`history_docs/COLAB_GUIDE.md`](../history_docs/COLAB_GUIDE.md) و راهنمای Windows در [`history_docs/SETUP_WINDOWS.md`](../history_docs/SETUP_WINDOWS.md) موجود است.

## آزمون‌ها و معیار پذیرش

فایل [`tests/test_phase0_environment.py`](../tests/test_phase0_environment.py) موارد زیر را پوشش می‌دهد:

- import وابستگی‌های اصلی؛
- حذف backend ناموجود inline و انتخاب `Agg`؛
- import پکیج و مقادیر هویت رقابت/فصل در config؛
- موجودبودن converter مورد انتظار و سازگاری stack؛
- نصب و نسخه‌ی SWI-Prolog و اجرای هدف Prolog ساده (با skip اگر در محیط حاضر نیست)؛
- اجرای tutorial Popper (تست کند، مشروط به موجودبودن checkout و uv)؛
- وجود حداقل ۲۰ بازی در فهرست مورد انتظار و دسترسی به حداقل یک فریم 360 در مجموعه‌ی کش‌شده؛
- دسترسی زنده به StatsBomb (تست شبکه‌ای).

اجرای مستقیم phase check و pytest مکمل هم‌اند: اولی گزارش تشخیصی قابل‌فهم برای اپراتور می‌دهد؛ دومی رفتارهای مشخص و قراردادهای فنی را regression-test می‌کند. بعضی تست‌های داده‌محور به cache و بعضی تست‌ها به شبکه یا ابزار خارجی نیاز دارند.

## فایل‌های مرجع این فاز

| موضوع | مسیر |
|---|---|
| چک‌های محیط و کد خروجی | [`scripts/run_phase0_check.py`](../scripts/run_phase0_check.py) |
| تست‌های محیط | [`tests/test_phase0_environment.py`](../tests/test_phase0_environment.py) |
| نسخه مجاز Python و وابستگی‌ها | [`pyproject.toml`](../pyproject.toml) |
| مسیرهای ریشه و داده | [`src/tactical_ilp/config.py`](../src/tactical_ilp/config.py) |
| راهنمای Colab | [`history_docs/COLAB_GUIDE.md`](../history_docs/COLAB_GUIDE.md) |
| راهنمای Windows | [`history_docs/SETUP_WINDOWS.md`](../history_docs/SETUP_WINDOWS.md) |


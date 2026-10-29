"""
Teacher LLM client: JSON-mode chat completions with retry/backoff on transient errors.

Retry policy (carried over from the original single-course generator, where it survived
multi-hour concurrent runs): exponential backoff with full jitter on 429 / timeouts /
connection errors / 5xx, honouring Retry-After; but codes that will never fix themselves
(exhausted quota, bad key) fail fast with FatalAPIError instead of burning retries.
"""

import json
import random
import re
import threading
import time

from ..config import get_settings


class FatalAPIError(RuntimeError):
    """An API error retrying will never fix (bad key, no quota). Stops the stage cleanly."""


NON_RETRYABLE_ERROR_CODES = {"insufficient_quota", "invalid_api_key", "account_deactivated"}


def _error_code(e):
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        return body.get("code") or (body.get("error") or {}).get("code")
    return getattr(e, "code", None)


class Usage:
    """Thread-safe token counter, so the UI can show what the run actually cost."""

    def __init__(self):
        self._lock = threading.Lock()
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, prompt: int, completion: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.calls += 1

    def cost_usd(self) -> float:
        s = get_settings()
        return round(self.prompt_tokens / 1e6 * s.price_in_per_m
                     + self.completion_tokens / 1e6 * s.price_out_per_m, 4)

    def as_dict(self) -> dict:
        return {"calls": self.calls, "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens, "cost_usd": self.cost_usd()}


class OpenAITeacher:
    def __init__(self, api_key: str, model: str, max_api_retries: int = 8, log=print):
        import openai
        from openai import OpenAI

        if model not in get_settings().allowed_teacher_models:
            raise FatalAPIError(f"Teacher model {model!r} is not allowed.")
        self._openai = openai
        self.client = OpenAI(api_key=api_key, timeout=120)
        self.model = model
        self.max_api_retries = max_api_retries
        self.usage = Usage()
        self.log = log

    def verify(self) -> None:
        """Cheap credential check (no tokens spent)."""
        try:
            self.client.models.retrieve(self.model)
        except self._openai.AuthenticationError as e:
            raise FatalAPIError("The API key was rejected by OpenAI.") from e
        except self._openai.NotFoundError as e:
            raise FatalAPIError(f"This API key has no access to {self.model}.") from e

    def complete_json(self, system: str, user: str, temperature: float = 0.7) -> str:
        def _call():
            return self.client.chat.completions.create(
                model=self.model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )

        resp = self._with_backoff(_call)
        if resp.usage:
            self.usage.add(resp.usage.prompt_tokens, resp.usage.completion_tokens)
        return resp.choices[0].message.content or ""

    def _with_backoff(self, fn, base_delay=2.0, max_delay=90.0):
        o = self._openai
        retryable = (o.RateLimitError, o.APITimeoutError, o.APIConnectionError, o.InternalServerError)
        attempt = 0
        while True:
            try:
                return fn()
            except o.AuthenticationError as e:
                raise FatalAPIError("The API key was rejected by OpenAI.") from e
            except retryable as e:
                code = _error_code(e)
                if code in NON_RETRYABLE_ERROR_CODES:
                    raise FatalAPIError(
                        f"Non-retryable API error ({code}). Check the key's billing/quota, then "
                        "retry the job: it resumes from the last checkpoint."
                    ) from e
                attempt += 1
                if attempt > self.max_api_retries:
                    raise
                retry_after = None
                resp = getattr(e, "response", None)
                header = resp.headers.get("retry-after") if resp is not None and hasattr(resp, "headers") else None
                if header:
                    try:
                        retry_after = float(header)
                    except ValueError:
                        pass
                delay = retry_after if retry_after is not None else random.uniform(
                    0, min(max_delay, base_delay * 2 ** (attempt - 1)))
                self.log(f"API retry {attempt}/{self.max_api_retries} after {type(e).__name__}; sleeping {delay:.1f}s")
                time.sleep(delay)


class FakeTeacher:
    """Deterministic offline teacher for demo jobs (JobConfig.demo) and tests.
    Answers the two prompt shapes this platform sends: syllabus structuring and Q&A synthesis."""

    WORDS = ("graph tree stack queue proof lemma induction recursion pointer cache kernel thread "
             "process mutex socket packet router hash table heap sort search vector matrix limit "
             "integral series entropy model gradient layer neuron loss bias variance sample "
             "estimator protocol schema index query join normal form automaton grammar language "
             "token parser compiler register memory page frame signal filter circuit voltage").split()

    # Non-Latin-script languages must answer in their own script or validation rejects every row.
    NATIVE = {
        "Hebrew": {
            "words": ("גרף עץ מחסנית תור הוכחה למה אינדוקציה רקורסיה מצביע זיכרון תהליך חוט שקע חבילה "
                      "נתב גיבוב טבלה ערימה מיון חיפוש וקטור מטריצה גבול אינטגרל טור אנטרופיה מודל "
                      "שכבה נוירון הטיה שונות דגימה אומד פרוטוקול סכמה אינדקס שאילתה אוטומט דקדוק "
                      "שפה אסימון מהדר אוגר דף מסגרת אות מסנן מעגל מתח").split(),
            "instruction": "הסבר כיצד {w} קשורים זה לזה בקורס?",
            "rationale": ("1. הרעיון הראשון מקשר בין {a} לבין {b}.\n"
                          "2. למשל, 3 + 4 = 7 מראה את השלב עבור {c}.\n"
                          "3. לבסוף {d} משלים את הטיעון."),
            "output": "לסיכום, {w} מתחברים לרעיון אחד ועקבי בנושא {n}.",
        },
        "Arabic": {
            "words": ("رسم شجرة مكدس طابور برهان تمهيدية استقراء تكرار مؤشر ذاكرة نواة خيط عملية قفل "
                      "مقبس حزمة موجه تجزئة جدول كومة فرز بحث متجه مصفوفة نهاية تكامل متسلسلة إنتروبيا "
                      "نموذج طبقة عصبون انحياز تباين عينة مقدر بروتوكول مخطط فهرس استعلام آلة قواعد "
                      "لغة رمز محلل مترجم سجل صفحة إطار إشارة مرشح دارة جهد").split(),
            "instruction": "اشرح كيف ترتبط {w} في هذا المقرر؟",
            "rationale": ("1. الفكرة الأولى تربط بين {a} و{b}.\n"
                          "2. على سبيل المثال، 3 + 4 = 7 يوضح الخطوة الخاصة بـ{c}.\n"
                          "3. أخيراً يكمل {d} الحجة."),
            "output": "إجمالاً، تتكامل {w} في فكرة واحدة متماسكة حول الموضوع {n}.",
        },
        "Russian": {
            "words": ("граф дерево стек очередь доказательство лемма индукция рекурсия указатель кэш "
                      "ядро поток процесс мьютекс сокет пакет маршрутизатор хеш таблица куча сортировка "
                      "поиск вектор матрица предел интеграл ряд энтропия модель градиент слой нейрон "
                      "смещение дисперсия выборка оценка протокол схема индекс запрос автомат "
                      "грамматика язык токен парсер компилятор регистр страница кадр сигнал фильтр").split(),
            "instruction": "Объясните, как {w} связаны в этом курсе?",
            "rationale": ("1. Первая идея связывает {a} и {b}.\n"
                          "2. Например, 3 + 4 = 7 показывает шаг для {c}.\n"
                          "3. Наконец, {d} завершает рассуждение."),
            "output": "В итоге {w} объединяются в одну связную идею по теме {n}.",
        },
    }

    def __init__(self, *_, **__):
        self.usage = Usage()
        self._n = 0
        self._lock = threading.Lock()

    def verify(self) -> None:
        pass

    def complete_json(self, system: str, user: str, temperature: float = 0.7) -> str:
        with self._lock:
            self._n += 1
            n = self._n
        self.usage.add(1000, 500)
        time.sleep(0.01)
        if "COURSE_EXTRACTION" in system:
            # Short outlines ("Week 1: ...") yield their real topics, like the real teacher would.
            from ..outline import parse_outline   # lazy: keeps llm/ free of platform imports

            body = "\n".join(ln for ln in user.splitlines() if not ln.startswith("# Source:"))
            if outline := parse_outline(body):
                return json.dumps({"courses": [{
                    **c, "domain": c["course_name"], "language": "English",
                    "summary": f"Fake summary of {c['course_name'] or 'the course'}.",
                    "out_of_scope": [], "style_note": None,
                } for c in outline]})
            first_line = next((ln.strip() for ln in user.splitlines() if ln.strip() and not ln.startswith("#")), "Course")
            name = first_line[:60]
            return json.dumps({"courses": [{
                "course_name": name,
                "domain": name,
                "language": "English",
                "summary": f"Fake summary of {name}.",
                "topics": [f"{name} topic {i}" for i in range(1, 9)],
                "out_of_scope": [],
                "style_note": None,
            }]})
        rng = random.Random(n)
        m = re.search(r"fluent academic (.+?) - not a", system)
        native = self.NATIVE.get(m.group(1) if m else "")
        if native:
            pick = lambda: rng.choice(native["words"])  # noqa: E731
            words = " ".join(rng.sample(native["words"], 12))
            return json.dumps({
                "instruction": native["instruction"].format(w=words),
                "rationale": native["rationale"].format(a=pick(), b=pick(), c=pick(), d=pick()),
                "output": native["output"].format(w=words, n=n),
            }, ensure_ascii=False)
        words = " ".join(rng.sample(self.WORDS, 12))
        return json.dumps({
            "instruction": f"Explain how {words} relate in this course?",
            "rationale": (f"1. The first idea connects {rng.choice(self.WORDS)} with {rng.choice(self.WORDS)}.\n"
                          f"2. For example, 3 + 4 = 7 shows the step for {rng.choice(self.WORDS)}.\n"
                          f"3. Finally the {rng.choice(self.WORDS)} completes the argument."),
            "output": f"Overall, {words} combine into one coherent idea about the topic {n}.",
        })


def make_teacher(api_key: str | None, model: str, log=print, demo: bool = False):
    if demo or get_settings().teacher_backend == "fake":
        return FakeTeacher()
    if not api_key:
        raise FatalAPIError("No teacher API key is stored for this job. Retry the job and provide the key.")
    return OpenAITeacher(api_key, model, log=log)

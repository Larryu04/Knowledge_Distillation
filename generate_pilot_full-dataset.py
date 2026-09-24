"""
generate_pilot_dataset.py — industrial-scale Hebrew CS training-data generator, constrained to
the actual syllabi of the Academic College of Ramat Gan (CS program).

v3 changes (pilot -> industrial scale, ~1200 examples/course, concurrent runs)
-------------------------------------------------------------------------------
1. Parameterized by --course: each run targets exactly ONE course/domain and only ever draws
   on that course's syllabus-derived topics + exclusions. Run it once per terminal per course
   (that's also what makes concurrent runs on the same API key safe: each process writes its
   own --output file, so there's no shared-file contention across courses).
2. Checkpointing: the output JSON array is rewritten to disk (atomically, via a temp file +
   os.replace) after every single accepted example, not at the end. Re-running the same
   --output path resumes from wherever it left off (counts existing valid rows for this course
   and only generates the remainder) instead of starting over or duplicating work.
3. Resilience: every API call goes through a retry-with-exponential-backoff-and-jitter wrapper
   that specifically catches rate limits (429), timeouts, connection errors and 5xxs. Backoff
   respects a Retry-After header when the API sends one. A single stubborn item (never comes
   back clean after retries) is logged and skipped rather than crashing the whole multi-hour run.
4. Diversity: at 4 questions per course, one clean subtopic per question was enough. At 1200,
   the same 4-8 sub-topics would repeat constantly. This version:
     - expands each course's topic list to the full granularity of its syllabus (still nothing
       outside the syllabus — just more distinct bullet points to draw from),
     - pairs every generation with a randomly chosen "angle" (theoretical / debug-a-snippet /
       compare-two-approaches / edge case / common-student-mistake / real-world analogy /
       step-by-step trace) and a difficulty tier, all explicitly injected into the prompt,
     - shows the model a small rotating sample of already-accepted instructions for this course
       and explicitly forbids repeating or structurally mimicking them,
     - and independently double-checks novelty after generation with a similarity check
       (difflib) against ALL previously accepted instructions for this run; a near-duplicate is
       rejected and regenerated rather than silently written to the dataset.

Output format is unchanged: 5 keys per object — "course_name", "domain", "instruction",
"rationale", "output" — course_name/domain are for evaluation tracking only.

Usage:
    export OPENAI_API_KEY=sk-...
    python generate_pilot_dataset.py --course "C++" --num 1200 --output cpp_data.json
    python generate_pilot_dataset.py --course "Java" --num 1200 --output java_data.json &
    python generate_pilot_dataset.py --course "Python" --num 1200 --output python_data.json &
    # ... one process per terminal per course, all safe to run concurrently on the same key.

    # Crashed at item 850? Just re-run the exact same command — it resumes from the existing
    # --output file instead of starting over:
    python generate_pilot_dataset.py --course "C++" --num 1200 --output cpp_data.json

A note on scale vs. syllabus depth (read this before kicking off a big run)
-----------------------------------------------------------------------------
These are intro-level syllabi. There is a real ceiling on how many TRULY distinct, deep
questions can be asked about e.g. "Round-Robin scheduling" before either (a) quality/novelty
degrades, or (b) the model starts drifting outside the syllabus to find something new to say.
This script fights that with topic rotation + angle/difficulty variation + similarity
rejection, but it can't manufacture syllabus content that isn't there. Watch the per-course
summary this script prints (accepted / duplicate-rejections / content-validation-rejections /
API-retry counts): a duplicate-rejection rate that climbs steadily as a run progresses is the
signal that a course's syllabus is running out of genuinely new ground at your target --num,
and you should sanity-check a sample of the back half of that file before training on it.
"""

import argparse
import difflib
import json
import os
import random
import re
import sys
import tempfile
import time

from openai import OpenAI
import openai

# ------------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------------
GOLD_PATH = "hebrew_cs_eval_dataset_24.json"
MODEL = "gpt-4o-mini"

# ------------------------------------------------------------------------------------------
# Syllabus-constrained topics.
#
# Each domain maps to:
#   - "course_name": the exact course name as it appears on the Ramat Gan College syllabus,
#     written into every generated example for evaluation tracking.
#   - "topics": granular subtopics pulled ONLY from that course's official syllabus. At
#     industrial scale these are cycled through repeatedly (round-robin, reshuffled each pass)
#     and combined with a random angle/difficulty — they are the raw material, not the final
#     question wording.
#   - "forbidden" (optional): terms that belong to a LATER/OTHER course and must never appear,
#     because this course's syllabus explicitly does not cover them yet.
# ------------------------------------------------------------------------------------------
DOMAIN_TOPICS = {
    "Computer Networks": {
        "course_name": "מבוא לתקשורת נתונים",
        "topics": [
            "מושגי יסוד בתקשורת נתונים והגדרות LAN, MAN, WAN",
            "השוו בין מודל השכבות OSI למודל TCP/IP וכיצד ממופות השכבות זו לזו",
            "המטרה של מודלים שכבתיים בתקשורת (כגון OSI) ביצירת אבסטרקציה ופיתוח מקבילי",
            "תפקידי הפרוטוקולים השונים בהתאם לשכבות מודל TCP/IP",
            "השוו בין תווכי תקשורת בשכבה הפיזית - נחושת, אלחוטי ואופטי - יתרונות וחסרונות של כל אחד",
            "פרוטוקולי LLC מול פרוטוקולי MAC בשכבת קישור הנתונים",
            "יתרונות וחסרונות של רשתות אלחוטיות (Wireless LANs) לעומת רשתות קוויות",
            "תפקידיהם של נתבים (Routers) מול מתגים (Switches) ברשת",
            "תפקיד פרוטוקול IP ופרוטוקול ICMP בשכבת הרשת",
            "מהו ההבדל בין ניתוב מבוסס וקטור-מרחק (Distance-Vector) לניתוב מבוסס מצב-קישור (Link-State) בשכבת הרשת",
            "מהי בקרת עומס (Congestion Control) ברשת, וכיצד היא קשורה לאיכות שירות (QoS)",
            "מהם ההבדלים העיקריים בין פרוטוקול TCP לפרוטוקול UDP בשכבת התעבורה, ומתי כדאי להעדיף כל אחד",
            "בקרת זרימה (Flow Control) בשכבת התעבורה ותפקידה",
            "תפקידי הפרוטוקולים המרכזיים בשכבת האפליקציה",
            "כיצד פועלת תוכנית פייתון המשתמשת ב-Sockets לתקשורת בסיסית בין לקוח לשרת",
            "כיצד מחלצים מידע משדות כותרת (Header) של פרוטוקול IP בעזרת תוכנית פייתון",
            "כיצד מחלצים מידע משדות כותרת של פרוטוקול UDP בעזרת תוכנית פייתון",
            "כיצד מחלצים מידע משדות כותרת של פרוטוקול TCP בעזרת תוכנית פייתון",
            "כיצד פועל שרת HTTP בסיסי הבנוי מעל Sockets, כולל מבנה בקשת ותגובת HTTP",
            "כיצד ניתן להשתמש בכלי Wireshark לניתוח אירועי רשת ותעבורת פרוטוקולים",
        ],
    },
    "Operating Systems": {
        "course_name": "עקרונות מערכות הפעלה",
        "topics": [
            "מהי מערכת הפעלה ומהן המטרות השונות (ולעיתים הסותרות) שלה",
            "מהו ה-PCB (Process Control Block) ומהם מצבי התהליך האפשריים (New, Ready, Running, Waiting, Terminated) והמעברים ביניהם",
            "כיצד תהליך נוצר במערכת ההפעלה, ומהו מבנה ה-shell",
            "השוו בין אלגוריתמי התזמון FIFO ו-Round-Robin, כולל דוגמה מספרית לחישוב זמן המתנה ממוצע",
            "מהם מטרות התזמון (Scheduling) במערכת הפעלה, וכיצד הן מיושמות בפועל ב-Linux וב-Windows",
            "כיצד פועל מנגנון הדפדוף (Paging) בניהול זיכרון וירטואלי, בהשוואה לסגמנטציה (Segmentation)",
            "מהי טבלת דפים היררכית, מהי טבלת דפים הפוכה, ומהו תפקיד ה-TLB בהאצת תרגום כתובות",
            "מהו אלגוריתם השעון (Clock Algorithm) להחלפת דפים בזיכרון וירטואלי",
            "מהו ההבדל בין חוטים (Threads) לתהליכים (Processes), והציגו את מודל Pthreads",
            "כיצד מתבצעת תקשורת בין תהליכים (IPC) באמצעות Pipe ו-Signals",
            "מהי בעיית הקטע הקריטי (Critical Section Problem), והציגו פתרון תוכנה כגון פתרון פטרסון",
            "כיצד פותר מנגנון ה-Semaphore את בעיית הקטע הקריטי, בהשוואה לפתרון באמצעות Monitor",
            "כיצד פותרים את בעיית היצרן-צרכן (Producer-Consumer) באמצעות סמפורים",
            "כיצד פותרים את בעיית הקוראים-כותבים (Readers-Writers Problem)",
            "מהו מבוי סתום (Deadlock), מהם ארבעת תנאי Coffman הנדרשים להתרחשותו, והמחישו זאת באמצעות בעיית הפילוסופים הסועדים",
            "מהן אסטרטגיות למניעת מבוי סתום (Deadlock Prevention) לעומת התאוששות ממנו (Deadlock Recovery)",
            "מהו מבנה מערכת קבצים (File System), וכיצד RAID תורם לאמינות ולביצועים",
            "מהי טכניקת ה-Journaling במערכות קבצים, ומדוע היא חשובה לשחזור לאחר קריסה",
        ],
    },
    "Cybersecurity": {
        "course_name": "מבוא לסייבר",
        "topics": [
            "מהם מושגי היסוד המרכזיים בתחום הגנת הסייבר",
            "מהם הסוגים העיקריים של תוכנות זדוניות (Malware) - וירוס, תולעת, סוס טרויאני וכופרה - וכיצד מנתחים אותן",
            "כיצד פועל תהליך זיהוי אנומליות (Anomaly Detection) באיתור התנהגות חשודה במערכת",
            "מהי מטרתם של תקני אבטחת מידע (Security Standards), ומה חשיבותם לארגון",
            "מהו תהליך ניתוח סיכונים (Risk Analysis) במערכת תוכנה, ומהם השלבים המרכזיים בזיהוי נכסים, איומים ופגיעויות",
            "מהו ההבדל בין אימות זהות (Authentication) להרשאה (Authorization) בניהול משתמשים, וכיצד אימות רב-שלבי (MFA) משפר את האבטחה",
            "מהם האיומים הנפוצים על פרוטוקולי תקשורת, ואילו הגנות ניתן ליישם מולם",
            "מהם האתגרים הייחודיים באבטחת מוצרי IoT בהשוואה למחשבים רגילים",
            "מהם השלבים המרכזיים בתהליך פיתוח מאובטח (Secure Development Lifecycle)",
        ],
    },
    "C++": {
        "course_name": "מבוא למדעי המחשב",
        "topics": [
            "כיצד מיוצגים מספרים במחשב (בינארי, הקסדצימלי) וכיצד מהדר מתרגם קוד לשפת מכונה",
            "פעולות בינאריות וביטויים אריתמטיים בסיסיים ב-++C",
            "משתנים ואופרטורים בסיסיים ב-++C",
            "תנאים (if/else) ב-++C, כולל דוגמה עם תנאים מקוננים",
            "לולאות (for/while) ב-++C, כולל דוגמה לחישוב סכום או ממוצע",
            "כתיבת פונקציות בסיסיות ב-++C והעברת פרמטרים אליהן",
            "מערכים סטטיים ב-++C, כולל דוגמה לחיפוש או סכימה של איברים",
            "מחרוזות (C-style strings) ב-++C, כולל דוגמה לפעולה בסיסית עליהן",
            "כיצד פועלת פונקציה רקורסיבית ב-++C, ומהם תפקידי תנאי העצירה ומחסנית הקריאות, המחישו באמצעות חישוב עצרת (factorial)",
            "כיצד מנתחים סיבוכיות זמן ריצה (Runtime Analysis) של אלגוריתם עם לולאות מקוננות ב-++C, והציגו דוגמה",
            "מהו ההבדל בין מצביע (Pointer) רגיל לבין הקצאה דינאמית של זיכרון באמצעות new ו-delete ב-++C",
            "מהו ההבדל בין מערך סטטי למחרוזת (String) ב-++C מבחינת ייצוג בזיכרון וגודל קבוע",
            "כיצד עובדים מול קבצים (File I/O) בסיסי ב-++C",
            "יסודות אלגוריתמים - כגון מיון או חיפוש בסיסי - וניתוחם ב-++C",
        ],
        "forbidden": ["virtual function", "פונקציה וירטואלית", "vtable", "smart pointer",
                      "מצביע חכם", "unique_ptr", "shared_ptr", "template", "תבנית גנרית"],
    },
    "Java": {
        "course_name": "תכנות מונחה עצמים בשפת Java",
        "topics": [
            "מבוא לתכנות מונחה עצמים - מהם עצמים, מחלקות ומופעים ב-Java",
            "כיצד בונים מחלקה ב-Java - תכונות (Fields) ויכולות (Methods)",
            "כיצד מגדירים פונקציות מחלקה (מתודות) מחוץ להגדרת המחלקה עצמה",
            "כיצד מיושם כימוס (Encapsulation) ב-Java באמצעות הרשאות גישה public ,private ו-protected",
            "מהם משתני המחלקה (Member Data) והמאפיינים (Properties) של אובייקט ב-Java",
            "כיצד עובדים בנאים (Constructors) ב-Java, ומהו תפקיד ה-Garbage Collection בשחרור זיכרון",
            "מהי העמסת בנאים (Constructor Overloading) ב-Java, והמחישו בדוגמה",
            "מהו השימוש במילת המפתח this ב-Java",
            "כיצד יוצרים מערך של אובייקטים ב-Java",
            "מהו ההבדל בין הרכבה (Composition) להורשה (Inheritance) ב-Java, ומתי יש להעדיף כל אחת (has-a לעומת is-a)",
            "כיצד פועלת הורשה (Inheritance) ב-Java, וכיצד דריסת מתודה (Method Overriding) עם super מאפשרת התנהגות ייחודית למחלקה יורשת",
            "מהי הרשאת גישה protected ב-Java, ומדוע היא שונה מ-private וממ-public",
            "מהו ההבדל בין העמסת מתודות (Overloading) לבין דריסת מתודות (Overriding) ב-Java",
            "מהם משתנים ומתודות סטטיים ב-Java, ומהו בנאי סטטי (Static Constructor/Block)",
            "מהי רב-צורתיות (Polymorphism) ב-Java ברמה בסיסית, והמחישו כיצד היא באה לידי ביטוי דרך דריסת מתודות",
        ],
        "forbidden": ["exception", "חריגה", "try", "catch", "collections framework",
                      "arraylist", "hashmap", "thread", "תהליכון", "synchronized"],
    },
    "Python": {
        "course_name": "תכנות מתקדם בשפת פייתון",
        "topics": [
            "מהו ההבדל בין הקלדה דינאמית (Dynamic Typing) בפייתון לבין הקלדה סטטית, ובין מפרש (Interpreter) למהדר (Compiler)",
            "מהם List, Tuple ו-Dictionary בפייתון, וההבדלים המעשיים ביניהם",
            "פונקציות בפייתון עם ערכי ברירת מחדל לפרמטרים (Default Parameter Values)",
            "מהי סביבה וירטואלית (venv) בפייתון, ומדוע יש צורך בה, כולל שימוש ב-pip install",
            "תכנות מונחה עצמים בסיסי בפייתון - מחלקות, בנאי (__init__), self ו-super",
            "כיצד פועלות פעולות וקטוריות (Vectorized Operations) ב-Numpy בהשוואה ללולאת for רגילה על רשימה",
            "כיצד קוראים וכותבים קבצי טקסט וכן מערכי Numpy מקבצים ואליהם",
            "כיצד מסננים ומצרפים (Filtering and Aggregation) נתונים במסגרת DataFrame של Pandas, כולל שימוש ב-groupby",
            "כיצד מבצעים ניקוי נתונים (Data Cleaning) וטיפול בערכים חסרים ב-Pandas",
            "כיצד ממזגים (Merge) ומצרפים (Join) שני DataFrame-ים שונים ב-Pandas",
            "כיצד יוצרים ויזואליזציה של נתונים בפייתון באמצעות Matplotlib או Seaborn",
            "מהי דריסת מתודות (Method Overriding) בפייתון ומהי העמסת אופרטורים (Operator Overloading)",
            "מהו ה-dataclass בפייתון, וכיצד הוא פשוט יותר ממחלקה רגילה בהגדרת מחלקות המחזיקות בעיקר נתונים",
            "מהו ה-Enum class בפייתון, ומתי משתמשים בו",
            "מהו תבנית העיצוב Composite (Composite Design Pattern) בתכנות מונחה עצמים בפייתון",
            "מהי ארכיטקטורת צד-שרת/צד-לקוח (Client-Server) במסגרת Flask, וכיצד מנוע התבניות Jinja2 מרנדר HTML דינאמי",
            "מהם שיטות ה-HTTP וקודי הסטטוס הבסיסיים בהם משתמשים באפליקציית Flask",
        ],
    },
}

REQUIRED_KEYS = {"course_name", "domain", "instruction", "rationale", "output"}

# ------------------------------------------------------------------------------------------
# Diversity knobs: randomly combined with a syllabus topic on every single call so that 1200
# calls against ~15 subtopics don't collapse into the same handful of questions reworded.
# ------------------------------------------------------------------------------------------
ANGLES = [
    "שאלה תיאורטית ישירה על העיקרון עצמו",
    "שאלת ניתוח או דיבוג של קטע קוד קצר המדגים את הנושא",
    "שאלת השוואה בין שתי גישות, מבנים או אלגוריתמים הקשורים לנושא",
    "תרחיש קצה (edge case) יוצא דופן שממחיש את הנושא",
    "שאלה המבוססת על טעות נפוצה שסטודנטים עושים בנושא זה, ומדוע היא שגויה",
    "שאלה המשתמשת באנלוגיה מהעולם האמיתי (שאינה טכנולוגית) כדי להסביר את המושג",
    "שאלה המבקשת מעקב צעד-אחר-צעד (Trace) אחר ביצוע קוד או תהליך הקשור לנושא",
    "שאלה המציגה מקרה שימוש (Use Case) מעשי ומבקשת להסביר את הפתרון המושגי מאחוריו",
]

DIFFICULTIES = [
    "קל - ברמת יסודות הנושא בתוך היקף הסילבוס בלבד",
    "בינוני - דורש חיבור בין כמה מרכיבים של הנושא, עדיין בתוך היקף הסילבוס בלבד",
    "מאתגר - הנקודה העדינה או המורכבת ביותר בנושא, אך עדיין בתוך היקף הסילבוס בלבד ולא מעבר לו",
]

SIMILARITY_THRESHOLD_DEFAULT = 0.72
RECENT_SAMPLE_SHOWN_IN_PROMPT = 8

# ------------------------------------------------------------------------------------------
# Transient-error retry wrapper (429 / timeouts / connection errors / 5xx)
# ------------------------------------------------------------------------------------------
RETRYABLE_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class FatalAPIError(RuntimeError):
    """Raised for API errors that retrying will never fix (e.g. exhausted billing quota,
    invalid API key) so the run should stop cleanly instead of burning retries/time."""


# Error codes the SDK reports as 429 (so they'd otherwise be treated as "retryable") but that
# are actually permanent until a human intervenes — retrying just wastes minutes before crashing
# anyway. Anything with one of these codes fails fast instead.
NON_RETRYABLE_ERROR_CODES = {"insufficient_quota", "invalid_api_key", "account_deactivated"}


def _error_code(e):
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        return body.get("code") or (body.get("error") or {}).get("code")
    return getattr(e, "code", None)


def call_with_backoff(fn, max_retries=8, base_delay=2.0, max_delay=90.0, label=""):
    """Call fn() with exponential backoff + full jitter on transient API errors.
    Honors a Retry-After header when the SDK exposes one on the exception."""
    attempt = 0
    while True:
        try:
            return fn()
        except RETRYABLE_EXCEPTIONS as e:
            code = _error_code(e)
            if code in NON_RETRYABLE_ERROR_CODES:
                raise FatalAPIError(
                    f"Non-retryable API error (code={code!r}): {e}. "
                    "This will not fix itself on retry — check your OpenAI billing/API key, "
                    "then re-run the exact same command to resume from the last checkpoint."
                ) from e
            attempt += 1
            if attempt > max_retries:
                raise
            retry_after = None
            resp = getattr(e, "response", None)
            if resp is not None:
                header_val = resp.headers.get("retry-after") if hasattr(resp, "headers") else None
                if header_val:
                    try:
                        retry_after = float(header_val)
                    except ValueError:
                        retry_after = None
            if retry_after is not None:
                delay = retry_after
            else:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay = random.uniform(0, delay)  # full jitter
            print(f"    [api-retry {attempt}/{max_retries}]{(' ' + label) if label else ''} "
                  f"{type(e).__name__}: {e} -> sleeping {delay:.1f}s")
            time.sleep(delay)


# ------------------------------------------------------------------------------------------
# Few-shot style exemplars, pulled from the gold set itself (content only used to demonstrate
# STYLE to GPT-4o; the model is explicitly told not to reuse these topics/wording, and the
# gold set's own topics live outside our syllabus-constrained list anyway).
# ------------------------------------------------------------------------------------------
def load_style_exemplars(gold_path, n=3, seed=42):
    if not os.path.exists(gold_path):
        print(f"[warn] {gold_path} not found — proceeding without few-shot exemplars.")
        return []
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)
    rng = random.Random(seed)
    sample = rng.sample(gold, min(n, len(gold)))
    exemplars = []
    for row in sample:
        exemplars.append({
            "instruction": row["Instruction"].strip(),
            "rationale": row["Rationale"].strip(),
            "output": row["Output"].strip(),
        })
    return exemplars


def build_system_prompt(domain, course_name, forbidden, exemplars):
    exemplar_block = "\n\n".join(
        f"דוגמה לסגנון הנדרש {i+1} (רק לסגנון! הנושא עצמו לא רלוונטי לקורס הזה):\n"
        + json.dumps(ex, ensure_ascii=False, indent=2)
        for i, ex in enumerate(exemplars)
    ) if exemplars else "(אין דוגמאות זמינות — הקפד עדיין על כל כללי הסגנון להלן)"

    forbidden_intro = ""
    if forbidden:
        forbidden_intro = (
            "\nזהו קורס מבוא, ואסור בהחלט להזכיר או להתבסס על הנושאים המתקדמים הבאים "
            "(הם נלמדים בקורסי המשך אחרים): " + ", ".join(forbidden) + "."
        )

    return f"""אתה מומחה ליצירת חומרי לימוד אקדמיים בעברית בתחום מדעי המחשב, עבור מערך אימון (fine-tuning) \
של מודל שפה בשם dictalm2.0-instruct המשמש כמתרגל וירטואלי לקורס "{course_name}" ({domain}) במכללה \
האקדמית רמת גן. אתה מתבקש ליצור מספר רב של דוגמאות אימון עבור קורס זה, אחת בכל קריאה, ולכן חשוב \
במיוחד שכל דוגמה תהיה שונה באמת מהקודמות - לא רק ניסוח שונה לאותה שאלה.

חוקי תוכן מחייבים (החוק החשוב ביותר):
0. השאלה חייבת להישאר בגבולות התוכן שנלמד בפועל בקורס "{course_name}", כפי שמופיע בסילבוס הרשמי \
שלו. אסור בשום אופן לחרוג לנושאים מתקדמים יותר שאינם נלמדים בקורס הזה, גם אם הם קשורים לוגית לנושא \
שנשאלת עליו.{forbidden_intro}

חוקי גיוון מחייבים (קריטי במיוחד כשמייצרים מאות דוגמאות על אותו קורס):
1. גם כאשר הנושא הבסיסי שקיבלת דומה לנושא ששימש בקריאה קודמת, השאלה עצמה, הדוגמה הקונקרטית \
ברציונל, וניסוח ה-output חייבים להיות שונים באופן מהותי - לא תבנית זהה עם שמות משתנים אחרים.
2. אם קיבלת "זווית" (angle) ו"רמת קושי" (difficulty) בהודעת המשתמש - חובה לבנות את השאלה בהתאם \
להן ממש (למשל: אם הזווית היא דיבוג קוד, הרציונל חייב להתייחס לקטע קוד קונקרטי; אם הזווית היא \
אנלוגיה מהעולם האמיתי, חובה לכלול אנלוגיה שאינה טכנולוגית).
3. אם קיבלת רשימת שאלות שכבר נוצרו לקורס זה - אסור בהחלט לחזור עליהן, לנסח אותן מחדש, או ליצור \
שאלה בעלת מבנה זהה לאחת מהן (אותה נקודת דגש, אותה דוגמה מספרית, אותו קטע קוד).

חוקי סגנון מחייבים (אין לחרוג מהם):
4. שדה ה-"rationale" חייב להכיל בדיוק בין 3 ל-5 נקודות ממוספרות (1. 2. 3. וכו'), כל נקודה משפט \
מלא ומפורט ולא סתם רמז.
5. בתוך ה-"rationale" חייבת להופיע לפחות דוגמה מוחשית אחת - קטע קוד (בתוך ``` אם רלוונטי), חישוב \
מספרי מפורש, או שימוש מפורש במילה "לדוגמה" המוביל להסבר קונקרטי (לא הפשטה נוספת).
6. שדה ה-"output" חייב להיות תמצות מסונתז (Synthesized) של כל הנקודות ברציונל יחד - משפט או שניים \
המשלבים את התובנה המרכזית מכמה נקודות שונות. אסור ש-"output" יהיה חזרה או ציטוט של הנקודה הראשונה \
בלבד ברציונל.
7. עברית תקנית, זורמת וטבעית ברמה אקדמית - לא תרגום מכני מאנגלית. מונחים טכניים באנגלית יכולים \
להישאר באנגלית בתוך המשפט העברי, כפי שמקובל בהוראת מדעי המחשב בעברית.
8. שדה ה-"instruction" הוא שאלה אקדמית אחת, ברורה וממוקדת, המזמינה תשובה מרובת-שלבים (לא שאלה של \
כן/לא ולא שאלת הגדרה בת מילה אחת), ותוך שמירה קפדנית על חוק 0 לעיל.

חשוב: אל תעתיק או תנסח מחדש את הדוגמאות המוצגות להלן - הן מובאות אך ורק להמחשת הסגנון (מבנה, עומק, \
שימוש ב"לדוגמה", רמת הפירוט), הנושאים בהן אינם רלוונטיים לקורס הזה ואין לגעת בהם.

{exemplar_block}

פורמט הפלט: יש להחזיר אך ורק אובייקט JSON יחיד, ללא טקסט נוסף, ללא markdown fences, עם בדיוק חמישה \
מפתחות: "course_name", "domain", "instruction", "rationale", "output". השדות course_name ו-domain \
יימסרו לך בהודעת המשתמש ויש להעתיק אותם כלשונם."""


def build_user_prompt(domain, course_name, topic_hint, forbidden, angle, difficulty, recent_instructions):
    forbidden_line = ""
    if forbidden:
        forbidden_line = (
            "\nתזכורת: זהו קורס מבוא - אסור להזכיר את הנושאים המתקדמים הבאים: "
            + ", ".join(forbidden) + "."
        )

    recent_block = ""
    if recent_instructions:
        sample = random.sample(recent_instructions, min(RECENT_SAMPLE_SHOWN_IN_PROMPT, len(recent_instructions)))
        recent_block = (
            "\n\nשאלות שכבר נוצרו לקורס זה - אסור לחזור עליהן או ליצור שאלה במבנה דומה:\n"
            + "\n".join(f"- {q}" for q in sample)
        )

    return (
        f'צור דוגמת אימון אחת בפורמט JSON עם בדיוק חמישה מפתחות: "course_name", "domain", '
        f'"instruction", "rationale", "output".\n'
        f'course_name: "{course_name}"\n'
        f'domain: "{domain}"\n'
        f"הנושא הספציפי לשאלה (מתוך הסילבוס הרשמי של הקורס): {topic_hint}\n"
        f"זווית מחייבת לשאלה: {angle}\n"
        f"רמת קושי מחייבת: {difficulty}"
        f"{forbidden_line}"
        f"{recent_block}\n\n"
        "החזר אך ורק את אובייקט ה-JSON, בהתאם לכל כללי הסגנון, התוכן והגיוון שהוגדרו."
    )


# ------------------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------------------
def normalize_for_similarity(text):
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


def most_similar(instruction, existing_instructions, threshold):
    """Return (best_ratio, best_match) if any existing instruction is too similar, else None."""
    norm = normalize_for_similarity(instruction)
    best_ratio, best_match = 0.0, None
    for other in existing_instructions:
        ratio = difflib.SequenceMatcher(None, norm, normalize_for_similarity(other)).ratio()
        if ratio > best_ratio:
            best_ratio, best_match = ratio, other
    if best_ratio >= threshold:
        return best_ratio, best_match
    return None


def coerce_string_fields(obj):
    """response_format={"type": "json_object"} guarantees valid JSON but NOT a specific shape —
    GPT-4o occasionally returns e.g. "rationale" as a list of point-strings instead of one
    string. Normalize every expected field to a string here so that shows up as an ordinary
    validation error (numbered-point regex won't match -> retry) instead of an AttributeError
    that crashes the whole run deep inside validate_example."""
    if not isinstance(obj, dict):
        return obj
    for key in ("course_name", "domain", "instruction", "rationale", "output"):
        if key not in obj:
            continue
        val = obj[key]
        if isinstance(val, str):
            continue
        elif isinstance(val, list):
            obj[key] = "\n".join(str(item) for item in val)
        else:
            obj[key] = json.dumps(val, ensure_ascii=False) if isinstance(val, dict) else str(val)
    return obj


def validate_example(obj, expected_domain, expected_course, forbidden, existing_instructions, similarity_threshold):
    errors = []
    if set(obj.keys()) != REQUIRED_KEYS:
        errors.append(f"keys must be exactly {REQUIRED_KEYS}, got {set(obj.keys())}")
        return errors  # no point checking further if keys are wrong

    if obj.get("domain") != expected_domain:
        errors.append(f"domain must be exactly {expected_domain!r}, got {obj.get('domain')!r}")
    if obj.get("course_name") != expected_course:
        errors.append(f"course_name must be exactly {expected_course!r}, got {obj.get('course_name')!r}")

    instruction, rationale, output = obj["instruction"].strip(), obj["rationale"].strip(), obj["output"].strip()

    if len(instruction) < 15:
        errors.append("instruction too short")

    numbered_points = re.findall(r"(?:^|\s)([1-5])\.\s", rationale)
    if not (3 <= len(numbered_points) <= 5):
        errors.append(f"rationale must have 3-5 numbered points, found {len(numbered_points)}")

    has_example_marker = ("לדוגמה" in rationale) or ("```" in rationale) or bool(re.search(r"\d+\s*[-=]\s*\d+", rationale))
    if not has_example_marker:
        errors.append("rationale missing a concrete worked example / code / 'לדוגמה'")

    if len(output) < 30:
        errors.append("output too short / not synthesized")

    # crude check that output isn't just the first rationale point copy-pasted
    first_point = rationale.split(". ", 1)[-1][:40] if rationale else ""
    if first_point and first_point in output:
        errors.append("output looks like a copy of the first rationale point, not a synthesis")

    # syllabus-scope guard: reject any forbidden (out-of-scope / later-course) term
    combined_text = f"{instruction}\n{rationale}\n{output}".lower()
    for term in forbidden:
        if term.lower() in combined_text:
            errors.append(f"mentions out-of-scope term for this course: {term!r}")

    # diversity guard: reject near-duplicates of anything already accepted this run
    if instruction:
        dup = most_similar(instruction, existing_instructions, similarity_threshold)
        if dup is not None:
            ratio, match = dup
            errors.append(
                f"too similar (ratio={ratio:.2f}) to an already-accepted instruction: {match[:80]!r}"
            )

    return errors


# ------------------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------------------
def generate_one(client, system_prompt, domain, course_name, topic_hint, forbidden,
                  existing_instructions, model, temperature,
                  max_content_retries, max_api_retries, similarity_threshold):
    last_errors = []
    angle = random.choice(ANGLES)
    difficulty = random.choice(DIFFICULTIES)

    for attempt in range(1, max_content_retries + 1):
        user_prompt = build_user_prompt(
            domain, course_name, topic_hint, forbidden, angle, difficulty, existing_instructions
        )
        if last_errors:
            user_prompt += (
                "\n\nהניסיון הקודם נכשל בבדיקות האיכות הבאות, תקן אותן: "
                + "; ".join(last_errors)
            )

        def _call():
            return client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )

        resp = call_with_backoff(_call, max_retries=max_api_retries, label=f"topic={topic_hint[:40]!r}")
        raw = resp.choices[0].message.content
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            last_errors = [f"invalid JSON: {e}"]
            continue

        if not isinstance(obj, dict):
            last_errors = [f"expected a JSON object, got {type(obj).__name__}"]
            continue
        obj = coerce_string_fields(obj)

        try:
            errors = validate_example(obj, domain, course_name, forbidden, existing_instructions, similarity_threshold)
        except Exception as e:  # noqa: BLE001 - defensive backstop; a shape surprise we
            # didn't anticipate should cost one retry, not the whole multi-hour run.
            last_errors = [f"validation crashed unexpectedly ({type(e).__name__}: {e}) — retrying"]
            print(f"    [content-retry {attempt}/{max_content_retries}] {last_errors}")
            continue

        if not errors:
            return {
                "course_name": obj["course_name"].strip(),
                "domain": obj["domain"].strip(),
                "instruction": obj["instruction"].strip(),
                "rationale": obj["rationale"].strip(),
                "output": obj["output"].strip(),
            }
        last_errors = errors
        print(f"    [content-retry {attempt}/{max_content_retries}] {errors}")

    return None  # caller decides whether to skip-and-continue


# ------------------------------------------------------------------------------------------
# Checkpointing (atomic write-after-every-item, resume on restart)
# ------------------------------------------------------------------------------------------
def load_existing(output_path, expected_domain):
    if not os.path.exists(output_path):
        return []
    try:
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"[warn] could not read existing {output_path} ({e}) — starting fresh.")
        return []
    if not isinstance(data, list):
        print(f"[warn] {output_path} does not contain a JSON array — starting fresh.")
        return []
    kept = [row for row in data if isinstance(row, dict) and row.get("domain") == expected_domain
            and REQUIRED_KEYS.issubset(row.keys())]
    if len(kept) != len(data):
        print(f"[warn] {output_path} had {len(data) - len(kept)} row(s) not matching domain "
              f"{expected_domain!r} or missing keys — dropping them from the resumed set.")
    return kept


def atomic_write_json(output_path, rows):
    directory = os.path.dirname(os.path.abspath(output_path)) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, output_path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--course", required=True, choices=list(DOMAIN_TOPICS.keys()),
                     help="Which course/domain to generate data for. Run one process per course.")
    ap.add_argument("--num", type=int, required=True, help="Target number of examples for this course.")
    ap.add_argument("--output", required=True, help="Output JSON path (also used to resume).")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                     help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gold-path", default=GOLD_PATH)
    ap.add_argument("--temperature", type=float, default=0.9,
                     help="Higher than the pilot default on purpose — helps diversity; "
                          "content/syllabus/forbidden-term validation keeps it in bounds.")
    ap.add_argument("--max-content-retries", type=int, default=4,
                     help="Retries for style/scope/duplicate validation failures per item.")
    ap.add_argument("--max-api-retries", type=int, default=8,
                     help="Retries for transient API errors (429/timeout/5xx) per call.")
    ap.add_argument("--similarity-threshold", type=float, default=SIMILARITY_THRESHOLD_DEFAULT,
                     help="difflib ratio above which a new instruction is treated as a near-duplicate (0-1).")
    ap.add_argument("--sleep-between-calls", type=float, default=0.4,
                     help="Base pacing delay between calls (jittered) — keep low since --max-api-retries "
                          "already backs off on real rate-limit responses.")
    ap.add_argument("--no-resume", action="store_true",
                     help="Ignore any existing --output file and start this course from scratch "
                          "(overwrites the file once new data is produced).")
    ap.add_argument("--seed", type=int, default=None, help="Optional RNG seed for topic/angle shuffling.")
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("[error] No API key found. Pass --api-key or set OPENAI_API_KEY.")

    if args.seed is not None:
        random.seed(args.seed)

    cfg = DOMAIN_TOPICS[args.course]
    course_name = cfg["course_name"]
    topics = cfg["topics"]
    forbidden = cfg.get("forbidden", [])
    domain = args.course

    client = OpenAI(api_key=args.api_key)
    exemplars = load_style_exemplars(args.gold_path)
    system_prompt = build_system_prompt(domain, course_name, forbidden, exemplars)

    results = [] if args.no_resume else load_existing(args.output, domain)
    existing_instructions = [r["instruction"] for r in results]
    start_count = len(results)

    if start_count >= args.num:
        print(f"[done] {args.output} already has {start_count} >= {args.num} examples for "
              f"{domain} — nothing to do.")
        return

    print(f"[run] course={domain} ({course_name})  target={args.num}  "
          f"resuming_from={start_count}  output={args.output}")

    # Shuffle topic order per pass so round-robin cycling doesn't always hit topics in the same
    # sequence; reshuffle whenever we wrap around the topic list.
    topic_order = topics[:]
    random.shuffle(topic_order)
    topic_idx = 0

    accepted = 0
    skipped = 0
    remaining = args.num - start_count

    while accepted < remaining:
        if topic_idx >= len(topic_order):
            topic_idx = 0
            random.shuffle(topic_order)
        topic_hint = topic_order[topic_idx]
        topic_idx += 1

        current_n = start_count + accepted + skipped + 1
        print(f"[{current_n}/{args.num}] topic={topic_hint[:60]!r}")

        try:
            example = generate_one(
                client, system_prompt, domain, course_name, topic_hint, forbidden,
                existing_instructions, args.model, args.temperature,
                args.max_content_retries, args.max_api_retries, args.similarity_threshold,
            )
        except FatalAPIError as e:
            print(f"\n[fatal] {e}")
            print(f"[fatal] Stopping this run. {len(results)} examples for {domain} are already "
                  f"safely saved in {args.output}. Once you've fixed the API key/billing issue, "
                  f"re-run the exact same command — it will resume from {len(results)}, not restart.")
            sys.exit(1)

        if example is None:
            skipped += 1
            print(f"    [skip] gave up on this item after {args.max_content_retries} content retries "
                  f"— moving on. (skipped so far: {skipped})")
        else:
            results.append(example)
            existing_instructions.append(example["instruction"])
            accepted += 1
            atomic_write_json(args.output, results)
            if accepted % 25 == 0 or accepted == remaining:
                print(f"    [checkpoint] {len(results)} total examples saved to {args.output} "
                      f"(accepted this run: {accepted}, skipped: {skipped})")

        time.sleep(args.sleep_between_calls + random.uniform(0, args.sleep_between_calls))

    print(f"\n[done] {args.output} now has {len(results)} examples for {domain}. "
          f"(this run: accepted={accepted}, skipped={skipped})")
    if skipped > accepted * 0.15 and accepted > 0:
        print(f"[warn] skip rate is high ({skipped}/{accepted + skipped}). This usually means the "
              f"syllabus is running out of genuinely novel ground at this volume — spot-check the "
              f"back half of {args.output} before training on it.")


if __name__ == "__main__":
    main()

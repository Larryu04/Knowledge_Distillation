"""
generate_pilot_dataset.py — pilot for a new Hebrew CS training-data generation pipeline,
constrained to the actual syllabi of the Academic College of Ramat Gan (CS program).

v2 changes (Distribution Shift fix)
------------------------------------
The first pilot nailed the STYLE (deep, multi-point rationale matching the gold eval set),
but drifted in CONTENT: it generated general/advanced CS trivia (C++ smart pointers, Java
multithreading, etc.) that is never taught in our actual courses. Our fine-tuned model is a
tutor for specific Ramat Gan College courses, so training data must stay inside what those
courses actually cover — otherwise we're teaching the model to sound deep about things
students were never taught, which is its own kind of Style/Content mismatch.

Fix:
1. Every topic below is extracted directly from the six course syllabi (PDFs) and mapped to
   the specific course that teaches it. Nothing outside these syllabi is used.
2. Two domains carry an explicit exclusion list, because their syllabi are intro-level and
   later, more advanced material is taught in OTHER courses:
   - C++ (מבוא למדעי המחשב, CS101): NO virtual functions, NO smart pointers, NO templates —
     those aren't covered until later courses.
   - Java (תכנות מונחה עצמים, intro OOP): NO exceptions, NO Collections framework, NO
     threading — same reason.
   The system prompt states these exclusions explicitly and the validator rejects any
   generated example that mentions a forbidden term for that domain.
3. Output format now carries 5 keys instead of 3: "course_name", "domain", "instruction",
   "rationale", "output". course_name/domain are for evaluation tracking only — the training
   script's masking code is expected to ignore them and mask/build on instruction+rationale+
   output exactly as before.

Usage:
    export OPENAI_API_KEY=sk-...
    python generate_pilot_dataset.py
    # or: python generate_pilot_dataset.py --api-key sk-... --output my_pilot.json
"""

import argparse
import json
import os
import random
import re
import sys
import time

from openai import OpenAI

# ------------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------------
GOLD_PATH = "hebrew_cs_eval_dataset_24.json"
DEFAULT_OUTPUT = "hebrew_cs_pilot_dataset_24.json"
MODEL = "gpt-4o-mini"
N_PER_DOMAIN = 4
MAX_RETRIES = 3
SLEEP_BETWEEN_CALLS = 1.0  # seconds, be gentle on rate limits

# ------------------------------------------------------------------------------------------
# Syllabus-constrained topics.
#
# Each domain maps to:
#   - "course_name": the exact course name as it appears on the Ramat Gan College syllabus,
#     written into every generated example for evaluation tracking.
#   - "topics": questions built ONLY from subjects listed in that course's syllabus.
#   - "forbidden" (optional): terms that belong to a LATER/OTHER course and must never
#     appear, because the syllabus this course teaches from explicitly does not cover them.
# ------------------------------------------------------------------------------------------
DOMAIN_TOPICS = {
    "Computer Networks": {
        "course_name": "מבוא לתקשורת נתונים",
        "topics": [
            "השוו בין מודל השכבות OSI למודל TCP/IP וכיצד ממופות השכבות זו לזו",
            "השוו בין תווכי תקשורת בשכבה הפיזית - נחושת, אלחוטי ואופטי - יתרונות וחסרונות של כל אחד",
            "מהו ההבדל בין ניתוב מבוסס וקטור-מרחק (Distance-Vector) לניתוב מבוסס מצב-קישור (Link-State) בשכבת הרשת",
            "מהם ההבדלים העיקריים בין פרוטוקול TCP לפרוטוקול UDP בשכבת התעבורה, ומתי כדאי להעדיף כל אחד",
        ],
    },
    "Operating Systems": {
        "course_name": "עקרונות מערכות הפעלה",
        "topics": [
            "מהו ה-PCB (Process Control Block) ומהם מצבי התהליך האפשריים (New, Ready, Running, Waiting, Terminated) והמעברים ביניהם",
            "השוו בין אלגוריתמי התזמון FIFO ו-Round-Robin, כולל דוגמה מספרית לחישוב זמן המתנה ממוצע",
            "כיצד פועל מנגנון הדפדוף (Paging) בניהול זיכרון וירטואלי, ומהו תפקיד ה-TLB בהאצת תרגום כתובות",
            "מהו מבוי סתום (Deadlock), מהם ארבעת תנאי Coffman הנדרשים להתרחשותו, והמחישו זאת באמצעות בעיית הפילוסופים הסועדים",
        ],
    },
    "Cybersecurity": {
        "course_name": "מבוא לסייבר",
        "topics": [
            "מהם הסוגים העיקריים של תוכנות זדוניות (Malware) - וירוס, תולעת, סוס טרויאני וכופרה - וכיצד מנתחים אותן",
            "מהו תהליך ניתוח סיכונים (Risk Analysis) במערכת תוכנה, ומהם השלבים המרכזיים בזיהוי נכסים, איומים ופגיעויות",
            "מהו ההבדל בין אימות זהות (Authentication) להרשאה (Authorization) בניהול משתמשים, וכיצד אימות רב-שלבי (MFA) משפר את האבטחה",
            "מהם האתגרים הייחודיים באבטחת מוצרי IoT בהשוואה למחשבים רגילים",
        ],
    },
    "C++": {
        "course_name": "מבוא למדעי המחשב",
        "topics": [
            "כיצד מנתחים סיבוכיות זמן ריצה (Runtime Analysis) של אלגוריתם עם לולאות מקוננות ב-++C, והציגו דוגמה",
            "כיצד פועלת פונקציה רקורסיבית ב-++C, ומהם תפקידי תנאי העצירה ומחסנית הקריאות, המחישו באמצעות חישוב עצרת (factorial)",
            "מהו ההבדל בין מצביע (Pointer) רגיל לבין הקצאה דינאמית של זיכרון באמצעות new ו-delete ב-++C",
            "מהו ההבדל בין מערך סטטי למחרוזת (String) ב-++C מבחינת ייצוג בזיכרון וגודל קבוע",
        ],
        "forbidden": ["virtual function", "פונקציה וירטואלית", "vtable", "smart pointer",
                      "מצביע חכם", "unique_ptr", "shared_ptr", "template", "תבנית גנרית"],
    },
    "Java": {
        "course_name": "תכנות מונחה עצמים בשפת Java",
        "topics": [
            "כיצד מיושם כימוס (Encapsulation) ב-Java באמצעות הרשאות גישה public ,private ו-protected",
            "כיצד פועלת הורשה (Inheritance) ב-Java, וכיצד דריסת מתודה (Method Overriding) עם super מאפשרת התנהגות ייחודית למחלקה יורשת",
            "מהו ההבדל בין העמסת מתודות (Overloading) לבין דריסת מתודות (Overriding) ב-Java",
            "מהו ההבדל בין הרכבה (Composition) להורשה (Inheritance) ב-Java, ומתי יש להעדיף כל אחת (has-a לעומת is-a)",
        ],
        "forbidden": ["exception", "חריגה", "try", "catch", "collections framework",
                      "arraylist", "hashmap", "thread", "תהליכון", "synchronized"],
    },
    "Python": {
        "course_name": "תכנות מתקדם בשפת פייתון",
        "topics": [
            "כיצד פעולות וקטוריות (Vectorized Operations) ב-Numpy משפרות ביצועים לעומת לולאת for רגילה על רשימה",
            "כיצד מסננים ומצרפים (Filtering and Aggregation) נתונים במסגרת DataFrame של Pandas, כולל שימוש ב-groupby",
            "מהו ה-dataclass בפייתון, וכיצד הוא פשוט יותר ממחלקה רגילה בהגדרת מחלקות המחזיקות בעיקר נתונים",
            "מהי ארכיטקטורת צד-שרת/צד-לקוח (Client-Server) במסגרת Flask, וכיצד מנוע התבניות Jinja2 מרנדר HTML דינאמי",
        ],
    },
}

REQUIRED_KEYS = {"course_name", "domain", "instruction", "rationale", "output"}
VALID_DOMAINS = set(DOMAIN_TOPICS.keys())


# ------------------------------------------------------------------------------------------
# Few-shot style exemplars, pulled from the gold set itself (content only used to demonstrate
# STYLE to GPT-4o; the model is explicitly told not to reuse these topics/wording, and the
# gold set's own topics live outside our syllabus-constrained list anyway).
# ------------------------------------------------------------------------------------------
def load_style_exemplars(gold_path, n=3):
    if not os.path.exists(gold_path):
        print(f"[warn] {gold_path} not found — proceeding without few-shot exemplars.")
        return []
    with open(gold_path, encoding="utf-8") as f:
        gold = json.load(f)
    random.seed(42)
    sample = random.sample(gold, min(n, len(gold)))
    exemplars = []
    for row in sample:
        exemplars.append({
            "instruction": row["Instruction"].strip(),
            "rationale": row["Rationale"].strip(),
            "output": row["Output"].strip(),
        })
    return exemplars


def build_system_prompt(exemplars):
    exemplar_block = "\n\n".join(
        f"דוגמה לסגנון הנדרש {i+1} (רק לסגנון! הנושא עצמו לא רלוונטי לקורסים שלנו):\n"
        + json.dumps(ex, ensure_ascii=False, indent=2)
        for i, ex in enumerate(exemplars)
    ) if exemplars else "(אין דוגמאות זמינות — הקפד עדיין על כל כללי הסגנון להלן)"

    forbidden_block = "\n".join(
        f"- בקורס {d['course_name']} ({domain}): אסור בהחלט להזכיר או להתבסס על: "
        + ", ".join(d["forbidden"])
        for domain, d in DOMAIN_TOPICS.items() if "forbidden" in d
    )
    forbidden_intro = (
        "להלן רשימת נושאים אסורים במפורש לקורסים מסוימים, מכיוון שהם נלמדים בקורסי המשך אחרים:\n"
        + forbidden_block
    ) if forbidden_block else ""

    return f"""אתה מומחה ליצירת חומרי לימוד אקדמיים בעברית בתחום מדעי המחשב, עבור מערך אימון (fine-tuning) \
של מודל שפה בשם dictalm2.0-instruct המשמש כמתרגל וירטואלי לקורסים ספציפיים במכללה האקדמית רמת גן. \
תפקידך ליצור דוגמת אימון יחידה בכל פעם, ברמת עומק גבוהה.

חוקי תוכן מחייבים (החוק החשוב ביותר):
0. השאלה חייבת להישאר בגבולות התוכן שנלמד בפועל בקורס הספציפי שצוין, כפי שמופיע בסילבוס הרשמי שלו. \
אסור בשום אופן לחרוג לנושאים מתקדמים יותר שאינם נלמדים בקורס הזה, גם אם הם קשורים לוגית לנושא. \
{forbidden_intro}

חוקי סגנון מחייבים (אין לחרוג מהם):
1. שדה ה-"rationale" חייב להכיל בדיוק בין 3 ל-5 נקודות ממוספרות (1. 2. 3. וכו'), כל נקודה משפט \
מלא ומפורט ולא סתם רמז.
2. בתוך ה-"rationale" חייבת להופיע לפחות דוגמה מוחשית אחת — קטע קוד (בתוך ``` אם רלוונטי), חישוב \
מספרי מפורש, או שימוש מפורש במילה "לדוגמה" המוביל להסבר קונקרטי (לא הפשטה נוספת).
3. שדה ה-"output" חייב להיות תמצות מסונתז (Synthesized) של כל הנקודות ברציונל יחד — משפט או שניים \
המשלבים את התובנה המרכזית מכמה נקודות שונות. אסור ש-"output" יהיה חזרה או ציטוט של הנקודה הראשונה \
בלבד ברציונל.
4. עברית תקנית, זורמת וטבעית ברמה אקדמית — לא תרגום מכני מאנגלית. מונחים טכניים באנגלית יכולים \
להישאר באנגלית בתוך המשפט העברי, כפי שמקובל בהוראת מדעי המחשב בעברית.
5. שדה ה-"instruction" הוא שאלה אקדמית אחת, ברורה וממוקדת, המזמינה תשובה מרובת-שלבים (לא שאלה של \
כן/לא ולא שאלת הגדרה בת מילה אחת), ותוך שמירה קפדנית על חוק 0 לעיל.

חשוב: אל תעתיק או תנסח מחדש את הדוגמאות המוצגות להלן — הן מובאות אך ורק להמחשת הסגנון (מבנה, עומק, \
שימוש ב"לדוגמה", רמת הפירוט), הנושאים בהן אינם רלוונטיים לקורסים שלנו ואין לגעת בהם.

{exemplar_block}

פורמט הפלט: יש להחזיר אך ורק אובייקט JSON יחיד, ללא טקסט נוסף, ללא markdown fences, עם בדיוק חמישה \
מפתחות: "course_name", "domain", "instruction", "rationale", "output". השדות course_name ו-domain \
יימסרו לך בהודעת המשתמש ויש להעתיק אותם כלשונם.

חשוב מאוד: All values in the JSON object MUST be single strings, NOT arrays or lists. גם אם שדה \
ה-"rationale" מכיל מספר נקודות ממוספרות, יש לכתוב את כולן בתוך מחרוזת טקסט אחת (עם ירידות שורה \
בתוך אותה מחרוזת), ולא כמערך (array/list) של מחרוזות."""


def build_user_prompt(domain, course_name, topic_hint, forbidden):
    forbidden_line = ""
    if forbidden:
        forbidden_line = (
            "\nתזכורת: זהו קורס מבוא - אסור להזכיר את הנושאים המתקדמים הבאים: "
            + ", ".join(forbidden) + "."
        )
    return (
        f'צור דוגמת אימון אחת בפורמט JSON עם בדיוק חמישה מפתחות: "course_name", "domain", '
        f'"instruction", "rationale", "output".\n'
        f'course_name: "{course_name}"\n'
        f'domain: "{domain}"\n'
        f"הנושא הספציפי לשאלה (מתוך הסילבוס הרשמי של הקורס): {topic_hint}"
        f"{forbidden_line}\n"
        "החזר אך ורק את אובייקט ה-JSON, בהתאם לכל כללי הסגנון והתוכן שהוגדרו."
    )


# ------------------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------------------
def validate_example(obj, expected_domain, expected_course, forbidden):
    errors = []
    if set(obj.keys()) != REQUIRED_KEYS:
        errors.append(f"keys must be exactly {REQUIRED_KEYS}, got {set(obj.keys())}")
        return errors  # no point checking further if keys are wrong

    # GPT-4o sometimes returns a field as a JSON array instead of a single string (e.g. when
    # "rationale" is described as "3-5 points"). Safely join any list values into one string
    # before we ever call .strip() on them.
    for key in REQUIRED_KEYS:
        if isinstance(obj[key], list):
            obj[key] = " ".join(str(item).strip() for item in obj[key])

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

    return errors


# ------------------------------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------------------------------
def generate_one(client, system_prompt, domain, course_name, topic_hint, forbidden, model, temperature):
    last_errors = []
    for attempt in range(1, MAX_RETRIES + 1):
        user_prompt = build_user_prompt(domain, course_name, topic_hint, forbidden)
        if last_errors:
            user_prompt += (
                "\n\nהניסיון הקודם נכשל בבדיקות האיכות הבאות, תקן אותן: "
                + "; ".join(last_errors)
            )

        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            last_errors = [f"invalid JSON: {e}"]
            continue

        errors = validate_example(obj, domain, course_name, forbidden)
        if not errors:
            return {
                "course_name": obj["course_name"].strip(),
                "domain": obj["domain"].strip(),
                "instruction": obj["instruction"].strip(),
                "rationale": obj["rationale"].strip(),
                "output": obj["output"].strip(),
            }
        last_errors = errors
        print(f"    [retry {attempt}/{MAX_RETRIES}] {errors}")

    raise RuntimeError(f"Failed to generate a valid example for topic: {topic_hint!r} — last errors: {last_errors}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                     help="OpenAI API key (defaults to OPENAI_API_KEY env var)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--gold-path", default=GOLD_PATH)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    ap.add_argument("--n-per-domain", type=int, default=N_PER_DOMAIN)
    ap.add_argument("--temperature", type=float, default=0.7)
    args = ap.parse_args()

    if not args.api_key:
        sys.exit("[error] No API key found. Pass --api-key or set OPENAI_API_KEY.")

    client = OpenAI(api_key=args.api_key)

    exemplars = load_style_exemplars(args.gold_path)
    system_prompt = build_system_prompt(exemplars)

    results = []
    total = len(DOMAIN_TOPICS) * args.n_per_domain
    done = 0

    for domain, cfg in DOMAIN_TOPICS.items():
        course_name = cfg["course_name"]
        topics = cfg["topics"]
        forbidden = cfg.get("forbidden", [])
        print(f"\n[domain] {domain} ({course_name})")
        for topic in topics[: args.n_per_domain]:
            done += 1
            print(f"  [{done}/{total}] {topic[:70]}...")
            example = generate_one(
                client, system_prompt, domain, course_name, topic, forbidden,
                args.model, args.temperature,
            )
            results.append(example)
            time.sleep(SLEEP_BETWEEN_CALLS)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[done] wrote {len(results)} examples to {args.output}")


if __name__ == "__main__":
    main()

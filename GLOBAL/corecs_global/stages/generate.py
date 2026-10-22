"""
Stage 2 — Synthetic dataset generation.

For each course, the teacher LLM writes one training row per call:
    {"instruction": question, "rationale": 3-5 numbered reasoning steps, "output": synthesis}
Each call is anchored to one syllabus topic (round-robin, reshuffled every pass) and paired
with a random question "angle" and difficulty tier, and it is shown a sample of questions
already accepted for that course. Every row is then validated locally (structure, worked
example present, output is a synthesis, out-of-scope terms, language/script, near-duplicate
check); failures are fed back to the teacher and retried.

Courses run concurrently (generation_concurrency threads); rows within a course stay
sequential so the duplicate check sees everything accepted before it. Every accepted row is
checkpointed atomically, so a killed run resumes where it stopped.

Output rows keep the 5-key contract used by training:
    course_name, domain, instruction, rationale, output
"""

import difflib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed


from ..languages import preset, script_ratio
from ..llm.teacher import FatalAPIError, make_teacher
from .base import StageContext, StageError, atomic_write_json, read_json

SIMILARITY_THRESHOLD = 0.72
RECENT_SAMPLE_SHOWN = 8
MAX_CONTENT_RETRIES = 4
MAX_CONSECUTIVE_SKIPS = 25      # stall detector: the course's topic pool is exhausted
PACING_S = float(os.environ.get("GLOBAL_GEN_PACING_S", 0.3))   # jittered pause between calls per course
MIN_SCRIPT_RATIO = 0.4          # technical terms in English are fine; whole answers are not
TEMPERATURE = 0.9               # diversity; validation keeps it in bounds

ANGLES = [
    "a direct theoretical question about the principle itself",
    "analysing or debugging a short concrete artifact (code snippet, derivation, table, or case) that demonstrates the topic",
    "comparing two approaches, structures, methods or algorithms related to the topic",
    "an unusual edge case that illuminates the topic",
    "a common student mistake on this topic and why it is wrong",
    "a real-world, non-technical analogy that explains the concept",
    "a step-by-step trace of a process, computation or procedure related to the topic",
    "a practical use case, asking to explain the conceptual solution behind it",
]

DIFFICULTIES = [
    "easy - fundamentals of the topic, strictly within the syllabus",
    "medium - connects several elements of the topic, strictly within the syllabus",
    "challenging - the subtlest point of the topic, but still strictly within the syllabus",
]

STRUCTURE_EXAMPLE = json.dumps({
    "instruction": "<one focused academic question that invites a multi-step answer>",
    "rationale": "1. <first reasoning step, a full sentence>\n2. <second step, with a concrete "
                 "worked example: numbers, a code snippet in ``` fences, or 'for example ...'>\n"
                 "3. <third step>\n4. <optional fourth step>",
    "output": "<one or two sentences that synthesize ALL the steps into the key insight>",
}, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------------------
# Prompts
# --------------------------------------------------------------------------------------
def build_system_prompt(course: dict, program: str, language: str) -> str:
    oos = ""
    if course.get("out_of_scope"):
        oos = ("\nThese topics belong to other/later courses and must NEVER be mentioned or relied on: "
               + ", ".join(course["out_of_scope"]) + ".")
    style = f"\n\nSpecial instruction for this course (follow strictly): {course['style_note']}" if course.get("style_note") else ""
    markers = preset(language)["example_markers"]
    example_hint = f' (in {language}, e.g. "{markers[0]}")' if markers else ""
    return f"""You create academic study material for fine-tuning a small language model that will act as a \
virtual teaching assistant for the course "{course['course_name']}" ({course['domain']}) in the \
"{program}" program. You will be asked for many training examples for this course, one per call, so \
every example must be genuinely different from the previous ones, not a rewording.

Content rule (most important):
0. The question must stay within what is actually taught in "{course['course_name']}" according to \
its syllabus. Never drift into more advanced topics the course does not teach, even if they are \
logically related.{oos}

Diversity rules:
1. Even when the topic resembles one used before, the question, the concrete example in the \
rationale and the output wording must differ substantially - not the same template with renamed variables.
2. Build the question exactly according to the given angle and difficulty (e.g. a debugging angle \
must reference a concrete artifact; an analogy angle must include a non-technical analogy).
3. If you are shown questions already created for this course, never repeat or rephrase them and \
never reuse their structure (same focal point, same numeric example, same snippet).

Style rules:
4. "rationale" contains between 3 and 5 numbered points (1. 2. 3. ...), each a full, detailed sentence.
5. "rationale" includes at least one concrete worked example: a code snippet in ``` fences, an \
explicit numeric calculation, or an explicit "for example"{example_hint} leading to a concrete explanation.
6. "output" is a synthesis of ALL rationale points in one or two sentences - never a copy of the first point.
7. Write "instruction", "rationale" and "output" entirely in natural, fluent academic {language} - not a \
mechanical translation. Standard technical terms may stay in English inside {language} sentences, as is \
customary in university teaching.
8. "instruction" is a single clear, focused academic question that invites a multi-step answer \
(not yes/no, not a one-word definition), obeying rule 0.{style}

Required structure (placeholders only - do not copy):
{STRUCTURE_EXAMPLE}

Output format: return ONLY a single JSON object with exactly three keys: "instruction", "rationale", "output"."""


def build_user_prompt(topic: str, angle: str, difficulty: str, recent: list[str], errors: list[str]) -> str:
    parts = [
        f"Create one training example.\nSyllabus topic for this question: {topic}\n"
        f"Required angle: {angle}\nRequired difficulty: {difficulty}"
    ]
    if recent:
        sample = random.sample(recent, min(RECENT_SAMPLE_SHOWN, len(recent)))
        parts.append("Questions already created for this course - do not repeat them or mimic their structure:\n"
                     + "\n".join(f"- {q}" for q in sample))
    if errors:
        parts.append("The previous attempt failed these quality checks, fix them: " + "; ".join(errors))
    parts.append('Return only the JSON object with "instruction", "rationale" and "output".')
    return "\n\n".join(parts)


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def normalize_for_similarity(text: str) -> str:
    text = re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip().lower()


class DuplicateIndex:
    """Near-duplicate detector over accepted instructions (difflib ratio, with the cheap
    quick_ratio upper bound as a prefilter so a 1,500-row course stays fast)."""

    def __init__(self, threshold: float):
        self.threshold = threshold
        self._norm: list[tuple[str, str]] = []

    def add(self, instruction: str) -> None:
        self._norm.append((normalize_for_similarity(instruction), instruction))

    def most_similar(self, instruction: str):
        norm = normalize_for_similarity(instruction)
        # SequenceMatcher caches its analysis of seq2 (b): fix b = the candidate, vary a.
        sm = difflib.SequenceMatcher(None, b=norm)
        best = (0.0, None)
        for other_norm, other in self._norm:
            sm.set_seq1(other_norm)
            if sm.real_quick_ratio() < self.threshold or sm.quick_ratio() < self.threshold:
                continue
            r = sm.ratio()
            if r > best[0]:
                best = (r, other)
        return best if best[0] >= self.threshold else None


def coerce_string_fields(obj: dict) -> dict:
    """JSON mode guarantees JSON, not shape: the teacher sometimes returns rationale as a list."""
    for key in ("instruction", "rationale", "output"):
        val = obj.get(key)
        if val is None or isinstance(val, str):
            continue
        if isinstance(val, list):
            obj[key] = "\n".join(str(v) for v in val)
        else:
            obj[key] = json.dumps(val, ensure_ascii=False) if isinstance(val, dict) else str(val)
    return obj


def validate_example(obj: dict, course: dict, language: str, dupes: DuplicateIndex) -> list[str]:
    missing = [k for k in ("instruction", "rationale", "output") if not isinstance(obj.get(k), str) or not obj[k].strip()]
    if missing:
        return [f"missing or empty keys: {missing}"]
    instruction, rationale, output = (obj[k].strip() for k in ("instruction", "rationale", "output"))
    p = preset(language)
    errors = []

    if len(instruction) < 15:
        errors.append("instruction too short")

    points = re.findall(r"(?:^|\s)([1-9])[.)]\s", rationale)
    if not 3 <= len(points) <= 5:
        errors.append(f"rationale must have 3-5 numbered points, found {len(points)}")

    low = rationale.lower()
    has_example = ("```" in rationale
                   or bool(re.search(r"\d+\s*[-+*/×=]\s*\d+", rationale))
                   or any(m in low for m in p["example_markers"])
                   or any(m in low for m in ("for example", "e.g.")))
    if not has_example:
        errors.append("rationale is missing a concrete worked example (code, calculation or 'for example')")

    if len(output) < 30:
        errors.append("output too short / not a synthesis")
    first_point = rationale.split(". ", 1)[-1][:40]
    if first_point and len(first_point) >= 20 and first_point in output:
        errors.append("output copies the first rationale point instead of synthesizing all of them")

    combined = f"{instruction}\n{rationale}\n{output}".lower()
    for term in course.get("out_of_scope", []):
        if term and term.lower() in combined:
            errors.append(f"mentions out-of-scope topic {term!r}")

    if p["script"]:
        ratio = script_ratio(f"{instruction}\n{rationale}\n{output}", p["script"])
        if ratio < MIN_SCRIPT_RATIO:
            errors.append(f"text must be written in {language} (only {ratio:.0%} of letters are)")

    dup = dupes.most_similar(instruction)
    if dup is not None:
        errors.append(f"too similar ({dup[0]:.2f}) to an existing question: {dup[1][:80]!r}")
    return errors


# --------------------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------------------
class CourseRun:
    def __init__(self, ctx: StageContext, course: dict, teacher, target: int, counter, stop: threading.Event):
        self.ctx, self.course, self.teacher, self.target = ctx, course, teacher, target
        self.counter, self.stop = counter, stop
        self.path = ctx.ws.course_data_dir / f"{course['slug']}.json"
        self.language = ctx.config.language
        self.system_prompt = build_system_prompt(course, ctx.config.program_name, self.language)
        rows = read_json(self.path, default=[])
        self.rows = [r for r in rows if isinstance(r, dict) and r.get("domain") == course["domain"]]
        self.dupes = DuplicateIndex(SIMILARITY_THRESHOLD)
        for r in self.rows:
            self.dupes.add(r["instruction"])
        self.skipped = 0

    def generate_one(self, topic: str) -> dict | None:
        angle, difficulty = random.choice(ANGLES), random.choice(DIFFICULTIES)
        recent = [r["instruction"] for r in self.rows[-200:]]
        errors: list[str] = []
        for _ in range(MAX_CONTENT_RETRIES):
            raw = self.teacher.complete_json(self.system_prompt,
                                             build_user_prompt(topic, angle, difficulty, recent, errors),
                                             temperature=TEMPERATURE)
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as e:
                errors = [f"invalid JSON: {e}"]
                continue
            if not isinstance(obj, dict):
                errors = ["expected a JSON object"]
                continue
            obj = coerce_string_fields(obj)
            errors = validate_example(obj, self.course, self.language, self.dupes)
            if not errors:
                return {
                    "course_name": self.course["course_name"],
                    "domain": self.course["domain"],
                    "instruction": obj["instruction"].strip(),
                    "rationale": obj["rationale"].strip(),
                    "output": obj["output"].strip(),
                }
        return None

    def run(self) -> dict:
        name = self.course["course_name"]
        if self.rows:
            self.ctx.log(f"{name}: resuming with {len(self.rows)} existing rows")
        self.counter(len(self.rows))
        topics = self.course["topics"][:]
        random.shuffle(topics)
        idx, consecutive = 0, 0
        while len(self.rows) < self.target and not self.stop.is_set():
            if idx >= len(topics):
                idx = 0
                random.shuffle(topics)
            topic = topics[idx]
            idx += 1
            row = self.generate_one(topic)
            if row is None:
                self.skipped += 1
                consecutive += 1
                if consecutive >= MAX_CONSECUTIVE_SKIPS:
                    self.ctx.log(f"{name}: {consecutive} items in a row failed validation - the syllabus "
                                 f"seems exhausted at this volume. Stopping this course at {len(self.rows)} rows.")
                    break
            else:
                consecutive = 0
                self.rows.append(row)
                self.dupes.add(row["instruction"])
                atomic_write_json(self.path, self.rows)
                self.counter(1)
            time.sleep(PACING_S + random.uniform(0, PACING_S))
        if self.skipped > 0.15 * max(1, len(self.rows)):
            self.ctx.log(f"{name}: high skip rate ({self.skipped} skipped) - the syllabus may be too short "
                         f"for {self.target} distinct questions; spot-check this course's data.")
        return {"course_name": name, "domain": self.course["domain"], "rows": len(self.rows),
                "target": self.target, "skipped": self.skipped}


def merge(ctx: StageContext, courses: list[dict]) -> list[dict]:
    merged = []
    for c in courses:
        merged += read_json(ctx.ws.course_data_dir / f"{c['slug']}.json", default=[])
    atomic_write_json(ctx.ws.merged, merged)
    return merged


def run(ctx: StageContext) -> None:
    syllabi = read_json(ctx.ws.syllabi)
    if not syllabi or not syllabi.get("courses"):
        raise StageError("No structured syllabi found; the ingestion stage must run first.")
    courses = syllabi["courses"]
    target = ctx.config.examples_per_course
    total = target * len(courses)

    teacher = make_teacher(os.environ.get("GLOBAL_TEACHER_API_KEY"), ctx.config.teacher_model, log=ctx.log)
    ctx.reporter.start(f"Generating {total:,} examples across {len(courses)} course(s)")

    lock = threading.Lock()
    done = [0]

    def counter(n: int) -> None:
        with lock:
            done[0] += n
            ctx.reporter.update(done[0], total, f"{done[0]:,} / {total:,} examples")

    stop = threading.Event()
    runs = [CourseRun(ctx, c, teacher, target, counter, stop) for c in courses]
    stats = []
    workers = min(ctx.config.generation_concurrency, len(runs))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(r.run): r for r in runs}
        for fut in as_completed(futures):
            try:
                stats.append(fut.result())
            except FatalAPIError as e:
                stop.set()
                raise StageError(str(e)) from e
            except Exception:
                stop.set()
                raise

    merged = merge(ctx, courses)
    empty = [s["course_name"] for s in stats if s["rows"] == 0]
    if empty:
        raise StageError(f"No valid examples could be generated for: {', '.join(empty)}.")
    for s in stats:
        if s["rows"] < 0.5 * s["target"]:
            ctx.log(f"Warning: {s['course_name']} reached only {s['rows']}/{s['target']} examples.")

    atomic_write_json(ctx.ws.dataset_stats, {"courses": stats, "total": len(merged),
                                             "usage": teacher.usage.as_dict()})
    ctx.reporter.finish(f"{len(merged):,} validated examples", usage=teacher.usage.as_dict(), rows=len(merged))

"""
Rule-based reading of plain-text course outlines ("Week 1: Arrays", "1. Recursion", "- Heaps").

Ingestion asks the teacher LLM first; this parser is the fallback when the LLM returns no usable
course for a short or informal outline, and it is what the offline FakeTeacher answers with, so
demo mode extracts the topics the lecturer actually typed.
"""

import re

_UNIT = (r"week|wk|lecture|lesson|session|class|unit|module|chapter|part|topic|day"
         r"|שבוע|הרצאה|שיעור|מפגש|יחידה|פרק|נושא"
         r"|أسبوع|الأسبوع|محاضرة|درس|وحدة|فصل"
         r"|неделя|лекция|занятие|урок|тема|модуль")

# "Week 1: Arrays", "Lecture 3 - Heaps", "שבוע 2: רקורסיה", "Week #4) Graphs"
UNIT_LINE = re.compile(rf"^\s*(?:{_UNIT})\s*#?\s*\d+[a-z]?\s*[:.)\-–—]\s*(?P<topic>.+)$", re.I)
# "1. Arrays", "2) Heaps", "- Graphs", "• Proofs"
LIST_LINE = re.compile(r"^\s*(?:\d+[.)]|[-*•▪●])\s+(?P<topic>.+)$")
# "Course: Data Structures", "Course title - Algorithms", "שם הקורס: מבני נתונים"
TITLE_LINE = re.compile(r"^\s*(?:course(?:\s+(?:title|name))?|subject|title|שם\s+הקורס|קורס"
                        r"|المقرر|اسم المقرر|курс|название курса)\s*[:\-–—]\s*(?P<title>.+)$", re.I)
UNIT_PREFIX = re.compile(rf"^\s*(?:{_UNIT})\s*#?\s*\d+[a-z]?\s*[:.)\-–—]\s*", re.I)


def strip_unit_prefix(s: str) -> str:
    """'Week 3: Take chances' -> 'Take chances' (keeps the text if nothing would remain)."""
    rest = UNIT_PREFIX.sub("", s, count=1).strip()
    return rest or s.strip()


def parse_outline(text: str, min_topics: int = 5) -> list[dict]:
    """Courses found in an outline, as {"course_name", "topics"}. course_name is "" when the
    text has no title line; the caller supplies a fallback. A "Course: X" line starts a new
    course. Free-standing lines count as topics only when nothing else in the block looks like
    a list, so prose paragraphs are not mistaken for topics; a free line above the list then
    serves as the course name ("Intro to Entrepreneurship" over "Week 1: ...")."""
    blocks: list[dict] = []
    cur = {"course_name": "", "heading": "", "marked": [], "plain": []}
    for raw in text.splitlines():
        line = raw.strip().lstrip("#").strip()
        if not line:
            continue
        if m := TITLE_LINE.match(line):
            if cur["marked"] or cur["plain"] or cur["course_name"]:
                blocks.append(cur)
            cur = {"course_name": m["title"].strip(), "heading": "", "marked": [], "plain": []}
        elif m := UNIT_LINE.match(line) or LIST_LINE.match(line):
            cur["marked"].append(m["topic"].strip())
        elif len(line) <= 120 and len(line.split()) <= 14:
            if not cur["marked"] and not cur["plain"]:
                cur["heading"] = line
            cur["plain"].append(line)
    blocks.append(cur)

    courses = []
    for b in blocks:
        if len(b["marked"]) >= min_topics:
            topics = b["marked"]
            if not b["course_name"] and b["heading"]:
                b["course_name"] = b["heading"]
        else:
            topics = b["marked"] + b["plain"]
        seen, uniq = set(), []
        for t in (strip_unit_prefix(t).rstrip(" .;,") for t in topics):
            if t and t.casefold() not in seen:
                seen.add(t.casefold())
                uniq.append(t)
        if len(uniq) >= min_topics:
            courses.append({"course_name": b["course_name"], "topics": uniq})
    return courses

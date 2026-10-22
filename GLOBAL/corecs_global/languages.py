"""
Per-language conventions shared by data generation, training and the model profile.

final_answer: the marker line the tutor ends every answer with ("explain, then conclude").
example_markers: words that signal a concrete worked example in the rationale.
script: a regex character class; if set, generated text must be mostly in that script
        (catches the teacher drifting into English for non-Latin-script courses).
preamble: the instruction prepended to every user turn, identically at train and run time.
"""

import re

PRESETS = {
    "English": {
        "final_answer": "Final answer:",
        "example_markers": ["for example", "e.g.", "for instance", "consider"],
        "script": None,
        "preamble": ("You are an expert academic teaching assistant for the {program} program, covering: "
                     "{courses}. Answer in English only. First explain the reasoning step by step, then "
                     "write a line that starts with '{final}' containing a short, concise answer."),
    },
    "Hebrew": {
        "final_answer": "תשובה סופית:",
        "example_markers": ["לדוגמה", "למשל", "לדוגמא"],
        "script": r"֐-׿",
        "preamble": ("אתה מתרגל אקדמי מומחה בתוכנית {program}, בקורסים: {courses}. ענה בעברית בלבד. "
                     "תחילה הסבר את ההיגיון שלב אחר שלב, ולאחר מכן כתוב שורה שמתחילה ב'{final}' "
                     "ובה תשובה קצרה ותמציתית."),
    },
    "Arabic": {
        "final_answer": "الإجابة النهائية:",
        "example_markers": ["على سبيل المثال", "مثلاً", "مثلا"],
        "script": r"؀-ۿ",
        "preamble": ("أنت مساعد تدريس أكاديمي خبير في برنامج {program}، في المقررات: {courses}. أجب باللغة "
                     "العربية فقط. اشرح أولاً المنطق خطوة بخطوة، ثم اكتب سطراً يبدأ بـ '{final}' يتضمن "
                     "إجابة قصيرة وموجزة."),
    },
    "Russian": {
        "final_answer": "Окончательный ответ:",
        "example_markers": ["например", "к примеру", "рассмотрим"],
        "script": r"Ѐ-ӿ",
        "preamble": ("Ты опытный академический ассистент преподавателя программы {program} по курсам: "
                     "{courses}. Отвечай только на русском. Сначала объясни рассуждение шаг за шагом, затем "
                     "напиши строку, начинающуюся с '{final}', с кратким ответом."),
    },
    "Spanish": {
        "final_answer": "Respuesta final:",
        "example_markers": ["por ejemplo", "p. ej.", "consideremos"],
        "script": None,
        "preamble": ("Eres un asistente docente académico experto del programa {program}, en los cursos: "
                     "{courses}. Responde solo en español. Primero explica el razonamiento paso a paso y "
                     "luego escribe una línea que empiece con '{final}' con una respuesta breve y concisa."),
    },
    "French": {
        "final_answer": "Réponse finale :",
        "example_markers": ["par exemple", "p. ex.", "considérons"],
        "script": None,
        "preamble": ("Tu es un assistant pédagogique expert du programme {program}, pour les cours : "
                     "{courses}. Réponds uniquement en français. Explique d'abord le raisonnement étape par "
                     "étape, puis écris une ligne commençant par '{final}' avec une réponse courte et concise."),
    },
    "German": {
        "final_answer": "Endgültige Antwort:",
        "example_markers": ["zum beispiel", "z. b.", "z.b.", "beispielsweise", "betrachten wir"],
        "script": None,
        "preamble": ("Du bist ein erfahrener akademischer Tutor für den Studiengang {program} in den Kursen: "
                     "{courses}. Antworte nur auf Deutsch. Erkläre zuerst die Überlegung Schritt für Schritt "
                     "und schreibe dann eine Zeile, die mit '{final}' beginnt, mit einer kurzen Antwort."),
    },
}

GENERIC = {
    "final_answer": "Final answer:",
    "example_markers": [],
    "script": None,
    "preamble": ("You are an expert academic teaching assistant for the {program} program, covering: "
                 "{courses}. Answer in {language} only. First explain the reasoning step by step, then "
                 "write a line that starts with '{final}' containing a short, concise answer."),
}


def preset(language: str) -> dict:
    return PRESETS.get(language, GENERIC)


def build_preamble(language: str, program: str, course_names: list[str]) -> str:
    p = preset(language)
    sep = "، " if language == "Arabic" else ", "
    return p["preamble"].format(program=program, courses=sep.join(course_names),
                                final=p["final_answer"], language=language)


def script_ratio(text: str, script: str) -> float:
    """Share of letters in `text` that belong to `script` (code blocks excluded)."""
    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    letters = re.findall(r"[^\W\d_]", text)
    if not letters:
        return 1.0
    in_script = re.findall(f"[{script}]", text)
    return len(in_script) / len(letters)

"""Stage logic without the GPU: ingestion parsing/structuring, generation validation, deploy guide."""

import json
import zipfile

import docx

from corecs_global import db
from corecs_global.jobs import JobConfig
from corecs_global.stages import deploy, generate, ingest
from corecs_global.stages.base import StageContext, atomic_write_json
from corecs_global.stages.package import zip_dir

from conftest import SYLLABUS_A


def _ctx(stage="ingest", **cfg):
    job_id = db.create_job(JobConfig.from_dict({"program_name": "Test Program", **cfg}).to_dict(), {})
    return StageContext.for_job(job_id, stage)


def test_extract_text_txt_and_docx(tmp_path):
    t = tmp_path / "a.txt"
    t.write_text(SYLLABUS_A, encoding="utf-8")
    assert "Hash tables" in ingest.extract_text(t)

    d = docx.Document()
    d.add_paragraph("Course: Algorithms")
    table = d.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Week 1"
    table.rows[0].cells[1].text = "Divide and conquer"
    p = tmp_path / "b.docx"
    d.save(p)
    text = ingest.extract_text(p)
    assert "Course: Algorithms" in text and "Week 1 | Divide and conquer" in text

    h = tmp_path / "c.txt"
    h.write_bytes("קורס: מבני נתונים".encode("cp1255"))
    assert "מבני נתונים" in ingest.extract_text(h)


def test_normalize_and_dedupe_courses():
    raw = [
        {"course_name": "Algorithms", "domain": "Algorithms", "topics": ["1. Sorting", "- Graphs", "DP", "Greedy", "Flows"]},
        {"course_name": "algorithms", "domain": "Algo", "topics": ["Sorting", "Hashing", "A", "B", "C"]},
        {"course_name": "Too small", "topics": ["a", "b"]},
        {"course_name": "Algorithms II", "domain": "Algorithms", "topics": "x\ny\nz\nw\nv"},
        "garbage",
    ]
    courses = ingest.dedupe_courses(ingest.normalize_courses(raw, "src"))
    names = [c["course_name"] for c in courses]
    assert names == ["Algorithms", "Algorithms II"]
    assert courses[0]["topics"][:2] == ["Sorting", "Graphs"] and "Hashing" in courses[0]["topics"]
    assert courses[1]["domain"] == "Algorithms 2"           # clashing labels get suffixed
    assert len({c["slug"] for c in courses}) == 2


GOOD = {
    "instruction": "Why does inserting into a hash table with chaining stay O(1) on average?",
    "rationale": ("1. Each bucket holds a linked list of colliding keys.\n"
                  "2. For example, with n = 100 keys and m = 50 buckets the load factor is 100 / 50 = 2.\n"
                  "3. A good hash spreads keys uniformly, so the expected list length equals the load factor."),
    "output": "With uniform hashing and a bounded load factor, chains stay short, so average insertion is constant time.",
}
COURSE = {"course_name": "Data Structures", "domain": "Data Structures", "out_of_scope": ["dynamic programming"]}


def test_validate_example_accepts_good_and_rejects_bad():
    dupes = generate.DuplicateIndex(0.72)
    assert generate.validate_example(dict(GOOD), COURSE, "English", dupes) == []

    no_points = dict(GOOD, rationale="It is fast because buckets are short, e.g. 2 items.")
    assert any("numbered points" in e for e in generate.validate_example(no_points, COURSE, "English", dupes))

    oos = dict(GOOD, output=GOOD["output"] + " Unlike dynamic programming, no table is needed.")
    assert any("out-of-scope" in e for e in generate.validate_example(oos, COURSE, "English", dupes))

    assert any("written in Hebrew" in e for e in generate.validate_example(dict(GOOD), COURSE, "Hebrew", dupes))

    dupes.add(GOOD["instruction"])
    near = dict(GOOD, instruction=GOOD["instruction"].replace("Why does", "Why do"))
    assert any("too similar" in e for e in generate.validate_example(near, COURSE, "English", dupes))

    assert generate.validate_example({"instruction": "x"}, COURSE, "English", dupes)[0].startswith("missing")


def test_coerce_list_rationale():
    obj = generate.coerce_string_fields({"rationale": ["1. a", "2. b"], "output": {"k": 1}})
    assert obj["rationale"] == "1. a\n2. b" and obj["output"] == '{"k": 1}'


def test_prompts_carry_course_language_and_scope():
    sp = generate.build_system_prompt(dict(COURSE, style_note="Prefer verbal proofs."), "B.Sc. CS", "Hebrew")
    assert "Data Structures" in sp and "dynamic programming" in sp and "Hebrew" in sp and "לדוגמה" in sp
    assert "Prefer verbal proofs." in sp
    up = generate.build_user_prompt("Heaps", "an edge case", "easy", ["Q1"], ["too short"])
    assert "Heaps" in up and "- Q1" in up and "too short" in up


def test_deploy_guide_from_packaged_bundle():
    ctx = _ctx("deploy", program_name="B.Sc. Biology")
    model_dir = ctx.ws.package_dir / "bundle" / "model"
    profile = {"display_name": "Bio Tutor", "courses": ["Genetics"], "context_length": 4096,
               "source": {"base_model": "Qwen/Qwen2.5-7B-Instruct", "quant": "Q4_K_M"}}
    atomic_write_json(model_dir / "b-sc-biology-tutor-q4_k_m.corecs.json", profile)
    deploy.run(ctx)
    guide = json.loads(ctx.ws.deploy.read_text(encoding="utf-8"))
    assert guide["gguf"] == "b-sc-biology-tutor-q4_k_m.gguf"
    assert guide["model_repo"] == "{HF_USER}/b-sc-biology-tutor-gguf"
    text = deploy.render_markdown(guide)
    assert "hf upload {HF_USER}/b-sc-biology-tutor-gguf model/b-sc-biology-tutor-q4_k_m.gguf" in text
    assert "resolve/main/b-sc-biology-tutor-q4_k_m.gguf?download=true" in text
    space = deploy.space_files(profile, guide["gguf"], "b-sc-biology")
    assert "sdk: docker" in space["README.md"] and "--hf-file" in space["Dockerfile"]
    assert db.get_job(ctx.job_id)["progress"]["deploy"]["status"] == "done"


def test_zip_dir_stores_weights_uncompressed(tmp_path):
    src = tmp_path / "bundle"
    (src / "model").mkdir(parents=True)
    (src / "model" / "m.gguf").write_bytes(b"GGUF" + b"\0" * 4096)
    (src / "README.txt").write_text("hello " * 200)
    out = tmp_path / "b.zip"
    zip_dir(src, out, "tutor")
    with zipfile.ZipFile(out) as z:
        info = {i.filename: i for i in z.infolist()}
    assert info["tutor/model/m.gguf"].compress_type == zipfile.ZIP_STORED
    assert info["tutor/README.txt"].compress_type == zipfile.ZIP_DEFLATED
    assert not (tmp_path / "b.zip.part").exists()

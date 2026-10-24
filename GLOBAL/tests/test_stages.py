"""Stage logic without the GPU: ingestion parsing/structuring, generation validation, deploy guide."""

import io
import json
import zipfile

import docx

from corecs_global import db, outline
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


WEEKLY = ("week 1: being yourself\nweek 2: being bold\nweek 3: take chances\n"
          "week 4: do actions\nweek 5: networking\nweek 6: talk with investors")


def test_parse_outline_untitled_weekly_list():
    assert outline.parse_outline(WEEKLY) == [{"course_name": "", "topics": [
        "being yourself", "being bold", "take chances", "do actions", "networking", "talk with investors"]}]
    # A free line above the list is the course name; "Course:" lines split courses.
    assert outline.parse_outline("Intro to Startups\n" + WEEKLY)[0]["course_name"] == "Intro to Startups"
    two = outline.parse_outline("Course: A\n" + WEEKLY + "\nCourse: B\n1. x\n2. y\n3. z\n4. w\n5. v")
    assert [c["course_name"] for c in two] == ["A", "B"] and two[1]["topics"] == ["x", "y", "z", "w", "v"]
    assert outline.parse_outline("שבוע 1: מערכים\nשבוע 2: רשימות\nשבוע 3: עצים\nשבוע 4: גרפים\nשבוע 5: מיון")[0]["topics"][0] == "מערכים"
    assert outline.parse_outline("too short to be a syllabus") == []


def test_ingest_short_untitled_outline(tmp_path):
    """The 6-line weekly list (113 chars, no title) used to be skipped as 'too short'."""
    ctx = _ctx(program_name="Entrepreneurship")
    ctx.ws.uploads.mkdir(parents=True, exist_ok=True)
    (ctx.ws.uploads / "pasted-01.txt").write_text(WEEKLY, encoding="utf-8")
    atomic_write_json(ctx.ws.sources, [{"name": "Pasted text 1", "file": "pasted-01.txt"}])
    ingest.run(ctx)
    [course] = json.loads(ctx.ws.syllabi.read_text(encoding="utf-8"))["courses"]
    assert course["course_name"] == "Entrepreneurship"       # fallback title = program name
    assert course["topics"] == ["being yourself", "being bold", "take chances", "do actions",
                                "networking", "talk with investors"]


def test_normalize_untitled_courses_get_fallback_names():
    raw = [{"course_name": "", "topics": ["Week 1: a", "Week 2: b", "c", "d", "e"]},
           {"topics": ["v", "w", "x", "y", "z"]}]
    courses = ingest.normalize_courses(raw, "Pasted text 1", "Program")
    assert [c["course_name"] for c in courses] == ["Program", "Program 2"]
    assert courses[0]["topics"][:2] == ["a", "b"]


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


BIO_PROFILE = {"display_name": "Bio Tutor", "courses": ["Genetics"], "context_length": 4096,
               "source": {"base_model": "Qwen/Qwen2.5-7B-Instruct", "quant": "Q4_K_M"}}


def _packaged(stage, stem="b-sc-biology-tutor-q4_k_m", weights=b"GGUF" + b"\1" * 1000, **cfg):
    ctx = _ctx(stage, program_name="B.Sc. Biology", **cfg)
    bundle = ctx.ws.package_dir / "bundle"
    atomic_write_json(bundle / "model" / f"{stem}.corecs.json", BIO_PROFILE)
    (bundle / "model" / f"{stem}.gguf").write_bytes(weights)
    (bundle / "huggingface").mkdir(parents=True)
    (bundle / "huggingface" / "README.md").write_text("# Bio Tutor", encoding="utf-8")
    return ctx


def test_deploy_guide_published_model():
    ctx = _packaged("deploy", hf_repo="prof-cohen/b-sc-biology-tutor-gguf")
    deploy.run(ctx)
    guide = json.loads(ctx.ws.deploy.read_text(encoding="utf-8"))
    assert guide["gguf"] == "b-sc-biology-tutor-q4_k_m.gguf" and guide["published"] is True
    assert guide["model_repo"] == "prof-cohen/b-sc-biology-tutor-gguf"
    assert guide["space_repo"] == "prof-cohen/b-sc-biology-tutor-demo"
    text = deploy.render_markdown(guide)
    assert "published your model automatically" in text and "{HF_USER}" not in text
    assert "hf upload prof-cohen/b-sc-biology-tutor-gguf" not in text      # no manual model upload
    assert "prof-cohen/b-sc-biology-tutor-gguf/resolve/main/b-sc-biology-tutor-q4_k_m.gguf?download=true" in text
    space = deploy.space_files(BIO_PROFILE, guide["gguf"], guide["model_repo"])
    assert "sdk: docker" in space["README.md"] and "MODEL_REPO=prof-cohen/b-sc-biology-tutor-gguf" in space["Dockerfile"]
    assert db.get_job(ctx.job_id)["progress"]["deploy"]["status"] == "done"


def test_deploy_guide_unpublished_model():
    ctx = _packaged("deploy")
    deploy.run(ctx)
    guide = json.loads(ctx.ws.deploy.read_text(encoding="utf-8"))
    assert guide["published"] is False and guide["model_repo"] is None
    text = deploy.render_markdown(guide)
    assert "not published" in text and "hf upload" not in text and "{HF_USER}" not in text
    assert [s["id"] for s in guide["sections"]] == ["download", "test", "hub", "share", "checklist"]


class FakeHub:
    """Stands in for huggingface_hub.HfApi in the publish stage."""
    calls: list = []
    private = False

    def __init__(self, token=None):
        self.token = token

    def create_repo(self, repo_id, **kw):
        FakeHub.calls.append(("create_repo", repo_id, kw))

    def repo_info(self, repo_id, **kw):
        return type("Info", (), {"private": FakeHub.private})()

    def create_commit(self, repo_id, operations, **kw):
        FakeHub.calls.append(("commit", repo_id, sorted(op.path_in_repo for op in operations), self.token))
        return type("Commit", (), {"oid": "abc123"})()


def test_publish_stage_creates_repo_and_pushes(monkeypatch):
    from corecs_global.stages import publish

    FakeHub.calls = []
    monkeypatch.setattr(publish, "HfApi", FakeHub)
    monkeypatch.setenv("HF_TOKEN", "hf_write")
    ctx = _packaged("publish", hf_repo="prof-cohen/b-sc-biology-tutor-gguf")
    publish.run(ctx)
    create, commit = FakeHub.calls
    assert create[:2] == ("create_repo", "prof-cohen/b-sc-biology-tutor-gguf")
    assert create[2]["exist_ok"] is True and create[2]["private"] is False
    assert commit == ("commit", "prof-cohen/b-sc-biology-tutor-gguf",
                      ["README.md", "b-sc-biology-tutor-q4_k_m.corecs.json", "b-sc-biology-tutor-q4_k_m.gguf"], "hf_write")
    result = json.loads(ctx.ws.publish.read_text(encoding="utf-8"))
    assert result["published"] and result["url"] == "https://huggingface.co/prof-cohen/b-sc-biology-tutor-gguf"
    assert result["commit"] == "abc123" and result["private"] is False

    # The token is deleted when a job fails: a retry without it explains what to do.
    monkeypatch.delenv("HF_TOKEN")
    try:
        publish.run(ctx)
        raise AssertionError("expected StageError")
    except publish.StageError as e:
        assert "enter your Write token again" in str(e)


def test_publish_stage_skips_without_repo(monkeypatch):
    from corecs_global.stages import publish

    monkeypatch.setattr(publish, "HfApi", None)               # must not be touched
    ctx = _packaged("publish")
    publish.run(ctx)
    assert json.loads(ctx.ws.publish.read_text(encoding="utf-8"))["published"] is False


def test_token_write_access_check(monkeypatch):
    from corecs_global import hf_publish

    def whoami_with(role, fine=None):
        info = {"type": "user", "name": "prof-cohen",
                "auth": {"accessToken": {"role": role, **({"fineGrained": fine} if fine else {})}}}
        monkeypatch.setattr(hf_publish, "HfApi", lambda token: type("A", (), {"whoami": lambda self: info})())

    scoped = lambda *perms: {"scoped": [{"entity": {"type": "user", "name": "prof-cohen"}, "permissions": list(perms)}]}
    whoami_with("write")
    assert hf_publish.check_token("hf_x") == "prof-cohen"
    whoami_with("fineGrained", scoped("repo.content.read", "repo.write"))
    assert hf_publish.check_token("hf_x") == "prof-cohen"
    for role, fine in (("read", None), ("fineGrained", scoped("repo.content.read"))):
        whoami_with(role, fine)
        try:
            hf_publish.check_token("hf_x")
            raise AssertionError(role)
        except hf_publish.PublishError as e:
            assert "read-only" in str(e)


def test_job_submit_assigns_repo_from_token(monkeypatch):
    from fastapi.testclient import TestClient

    from corecs_global import config, hf_publish
    from corecs_global.api import app

    def check(token):
        if token != "hf_good":
            raise hf_publish.PublishError("This Hugging Face token is read-only.")
        return "prof-cohen"

    monkeypatch.setattr(hf_publish, "check_token", check)
    client = TestClient(app)
    form = {"config": json.dumps({"program_name": "B.Sc. Biology", "hf_repo": "someone-else/x"}),
            "pasted_text": SYLLABUS_A}
    job_id = client.post("/api/jobs", data={**form, "hf_token": "hf_good"}).json()["id"]
    assert db.get_job(job_id)["config"]["hf_repo"] == "prof-cohen/b-sc-biology-tutor-gguf"
    r = client.post("/api/jobs", data={**form, "hf_token": "hf_read"})
    assert r.status_code == 400 and "read-only" in r.json()["detail"]
    # Demo mode: no token means no publishing, and the client can't choose a repository.
    job_id = client.post("/api/jobs", data=form).json()["id"]
    assert db.get_job(job_id)["config"]["hf_repo"] == ""
    # Outside demo mode the token is required.
    monkeypatch.setenv("GLOBAL_TEACHER_BACKEND", "openai")
    config.get_settings.cache_clear()
    monkeypatch.setattr("corecs_global.api._verify_key", lambda *a: None)
    r = client.post("/api/jobs", data={**form, "api_key": "sk-x"})
    assert r.status_code == 400 and "Write access is required" in r.json()["detail"]


def test_student_app_manifest_and_zip(tmp_path, monkeypatch):
    """Deploy checksums the GGUF; the API turns it into the desktop app's corecs-model.json."""
    import hashlib

    from fastapi.testclient import TestClient

    from corecs_global import config
    from corecs_global.api import app

    installer = tmp_path / "CoreCS_0.1.0_x64-setup.exe"
    installer.write_bytes(b"MZ fake installer")
    monkeypatch.setenv("GLOBAL_DESKTOP_INSTALLER", str(installer))
    config.get_settings.cache_clear()

    weights = b"GGUF" + b"\1" * 1000
    ctx = _packaged("deploy", stem="bio-q4_k_m", weights=weights, hf_repo="prof-cohen/bio-tutor-gguf")
    deploy.run(ctx)

    client = TestClient(app)
    url = f"/api/jobs/{ctx.job_id}"
    m = client.get(f"{url}/model-manifest").json()                 # repo = the published one
    assert m == {"version": 1, "source": "corecs-global", "display_name": "Bio Tutor",
                 "repo": "prof-cohen/bio-tutor-gguf", "revision": "main", "gguf": "bio-q4_k_m.gguf",
                 "profile": "bio-q4_k_m.corecs.json", "sha256": hashlib.sha256(weights).hexdigest(),
                 "size_bytes": len(weights)}
    for bad in ("{HF_USER}/bio-tutor-gguf", "no-slash", "a/b/c", "a/b?x=1"):
        assert client.get(f"{url}/model-manifest", params={"repo": bad}).status_code == 400, bad

    r = client.get(f"{url}/student-app")
    assert r.status_code == 200 and 'filename="bio-tutor-app.zip"' in r.headers["content-disposition"]
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        assert sorted(z.namelist()) == ["bio-tutor-app/CoreCS_0.1.0_x64-setup.exe",
                                        "bio-tutor-app/README.txt", "bio-tutor-app/corecs-model.json"]
        assert json.loads(z.read("bio-tutor-app/corecs-model.json")) == m
        readme = z.read("bio-tutor-app/README.txt").decode()
        assert "EXTRACT ALL FILES BEFORE RUNNING THE SETUP" in readme and '"Extract All..."' in readme

    monkeypatch.setenv("GLOBAL_DESKTOP_INSTALLER", str(tmp_path / "missing.exe"))
    config.get_settings.cache_clear()
    assert client.get(f"{url}/student-app").status_code == 404

    unpublished = _packaged("deploy", stem="bio-q4_k_m")
    deploy.run(unpublished)
    r = client.get(f"/api/jobs/{unpublished.job_id}/model-manifest")
    assert r.status_code == 400 and "not published" in r.json()["detail"]


def test_desktop_installer_drop_folder(tmp_path, monkeypatch):
    """runtime/installer/ wins over a local build, and is re-scanned on every use."""
    import os

    from corecs_global import config

    monkeypatch.delenv("GLOBAL_DESKTOP_INSTALLER", raising=False)
    monkeypatch.setenv("GLOBAL_RUNTIME_DIR", str(tmp_path))
    config.get_settings.cache_clear()
    s = config.get_settings()
    assert s.desktop_installer is None

    nsis = tmp_path / "desktop" / "src-tauri" / "target" / "release" / "bundle" / "nsis"
    nsis.mkdir(parents=True)
    (nsis / "CoreCS_0.1.0_x64-setup.exe").write_bytes(b"MZ")
    assert s.desktop_installer == nsis / "CoreCS_0.1.0_x64-setup.exe"

    drop = tmp_path / "installer"
    drop.mkdir()
    (drop / "CoreCS_0.1.0_x64-setup.exe").write_bytes(b"MZ")
    (drop / "CoreCS_0.2.0_x64-setup.exe").write_bytes(b"MZ")
    os.utime(drop / "CoreCS_0.2.0_x64-setup.exe", (1, 1))      # newest by time, not by name
    assert s.desktop_installer == drop / "CoreCS_0.1.0_x64-setup.exe"
    config.get_settings.cache_clear()


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

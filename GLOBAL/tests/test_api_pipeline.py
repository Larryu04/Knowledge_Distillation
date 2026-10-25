"""API + worker end to end (io lane, fake teacher): submit -> ingest -> review -> generate."""

import json

from fastapi.testclient import TestClient

from corecs_global import db, vault, worker
from corecs_global.api import app
from corecs_global.stages.base import Workspace

from conftest import SYLLABUS_A, SYLLABUS_B


def _submit(client, review: bool, n: int = 20, **extra):
    cfg = {"program_name": "Test CS", "examples_per_course": n, "review_syllabi": review,
           "base_model": "Qwen/Qwen2.5-1.5B-Instruct", "generation_concurrency": 2}
    files = [("files", ("ds.txt", SYLLABUS_A.encode(), "text/plain"))]
    r = client.post("/api/jobs", data={"config": json.dumps(cfg), "api_key": "sk-test",
                                       "pasted_text": SYLLABUS_B, **extra}, files=files)
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_options_and_estimate():
    with TestClient(app) as client:
        opts = client.get("/api/options").json()
        assert opts["teacher_models"] == ["gpt-4o-mini"] and opts["demo_mode"] is True
        est = client.post("/api/estimate", json={"config": {"examples_per_course": 100}, "n_sources": 2}).json()
        assert est["rows"] == 200


def test_job_list_is_per_browser_unless_admin():
    with TestClient(app) as client:
        a, b = _submit(client, review=True), _submit(client, review=True)
        assert client.get("/api/jobs").json() == []                    # no ids: nothing leaks
        assert [j["id"] for j in client.get(f"/api/jobs?ids={a},nope,{a}").json()] == [a]
        assert {j["id"] for j in client.get("/api/jobs?admin=w22222").json()} == {a, b}
        assert client.get("/api/jobs?admin=wrong").json() == []


def test_installer_download(tmp_path, monkeypatch):
    with TestClient(app) as client:
        monkeypatch.setenv("GLOBAL_DESKTOP_INSTALLER", str(tmp_path / "missing.exe"))
        assert client.get("/api/installer").status_code == 404
        exe = tmp_path / "CoreCS_0.1.0_x64-setup.exe"
        exe.write_bytes(b"MZ installer")
        monkeypatch.setenv("GLOBAL_DESKTOP_INSTALLER", str(exe))
        r = client.get("/api/installer")
        assert r.status_code == 200 and r.content == b"MZ installer"
        assert "CoreCS_0.1.0_x64-setup.exe" in r.headers["content-disposition"]


def test_rejects_bad_uploads_and_config():
    with TestClient(app) as client:
        r = client.post("/api/jobs", data={"config": json.dumps({"program_name": "x"})},
                        files=[("files", ("evil.exe", b"MZ", "application/octet-stream"))])
        assert r.status_code == 400 and "unsupported" in r.json()["detail"]
        assert db.list_jobs() == []                                   # nothing half-created
        r = client.post("/api/jobs", data={"config": json.dumps({"program_name": "x", "teacher_model": "gpt-4o"}),
                                           "pasted_text": SYLLABUS_A})
        assert r.status_code == 400
        r = client.post("/api/jobs", data={"config": json.dumps({"program_name": "x"})})
        assert r.status_code == 400


def test_full_io_pipeline_with_review():
    with TestClient(app) as client:
        job_id = _submit(client, review=True)
        assert vault.load(job_id, vault.TEACHER_KEY) == "sk-test"

        worker.serve(("io",), once=True)                              # ingest, then pause
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "awaiting_review", job["error"]
        assert job["stages"][0]["status"] == "done"
        syl = client.get(f"/api/jobs/{job_id}/syllabi").json()
        assert len(syl["courses"]) == 2

        # Lecturer drops one course and edits the other's topics.
        kept = syl["courses"][0]
        kept["topics"] = kept["topics"][:5] + ["A topic the lecturer added"]
        r = client.put(f"/api/jobs/{job_id}/syllabi", json={"courses": [kept]})
        assert r.status_code == 200 and len(r.json()["courses"]) == 1
        assert client.post(f"/api/jobs/{job_id}/approve").json()["status"] == "queued"

        worker.serve(("io",), once=True)                              # generate; train is gpu-lane
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "queued" and job["next_stage"] == "train", job["error"]
        gen = next(s for s in job["stages"] if s["name"] == "generate")
        assert gen["status"] == "done" and gen["rows"] == 20
        assert vault.load(job_id, vault.TEACHER_KEY) is None          # key scrubbed after generation

        rows = json.loads(Workspace(db.job_dir(job_id)).merged.read_text(encoding="utf-8"))
        assert len(rows) == 20 and set(rows[0]) == {"course_name", "domain", "instruction", "rationale", "output"}
        assert client.get(f"/api/jobs/{job_id}/dataset").status_code == 200

        events = client.get(f"/api/jobs/{job_id}/events").json()
        assert any(e["kind"] == "stage" and e.get("stage") == "generate" for e in events)


def test_generation_resumes_from_checkpoint():
    with TestClient(app) as client:
        job_id = _submit(client, review=False, n=20)
        worker.serve(("io",), once=True)
        ws = Workspace(db.job_dir(job_id))
        merged = json.loads(ws.merged.read_text(encoding="utf-8"))
        assert len(merged) == 40

        # Simulate a crash mid-generation with a larger target: existing rows must be kept.
        db.update_job(job_id, config={**db.get_job(job_id)["config"], "examples_per_course": 25})
        db.update_stage_progress(job_id, "generate", {"status": "pending"})
        db.update_job(job_id, status="queued", next_stage="generate")
        worker.serve(("io",), once=True)
        merged2 = json.loads(ws.merged.read_text(encoding="utf-8"))
        assert len(merged2) == 50
        assert [r["instruction"] for r in merged2 if r["domain"] == merged[0]["domain"]][:20] == \
               [r["instruction"] for r in merged if r["domain"] == merged[0]["domain"]]


def test_cancel_and_retry_queued_job():
    with TestClient(app) as client:
        job_id = _submit(client, review=False)
        assert client.post(f"/api/jobs/{job_id}/cancel").json()["status"] == "cancelled"
        assert vault.load(job_id, vault.TEACHER_KEY) is None
        r = client.post(f"/api/jobs/{job_id}/retry", json={})
        assert r.status_code == 200 and r.json()["status"] == "queued" and r.json()["next_stage"] == "ingest"
        assert client.delete(f"/api/jobs/{job_id}").status_code == 409   # not terminal
        client.post(f"/api/jobs/{job_id}/cancel")
        assert client.delete(f"/api/jobs/{job_id}").status_code == 200
        assert client.get(f"/api/jobs/{job_id}").status_code == 404


def test_failed_stage_reports_error():
    with TestClient(app) as client:
        r = client.post("/api/jobs", data={"config": json.dumps({"program_name": "x", "review_syllabi": False}),
                                           "pasted_text": "too short to be a syllabus"})
        job_id = r.json()["id"]
        worker.serve(("io",), once=True)
        job = client.get(f"/api/jobs/{job_id}").json()
        assert job["status"] == "failed" and "No usable courses" in job["error"]
        assert job["stages"][0]["status"] == "failed"


def test_edit_and_rerun_failed_job():
    with TestClient(app) as client:
        cfg = {"program_name": "x", "review_syllabi": False, "learning_rate": 5e-5}
        r = client.post("/api/jobs", data={"config": json.dumps(cfg), "pasted_text": "too short\n---\nalso short"},
                        files=[("files", ("ds.txt", SYLLABUS_A.encode(), "text/plain")),
                               ("files", ("junk.txt", b"junk", "text/plain"))])
        old_id = r.json()["id"]
        db.update_job(old_id, status="failed", error="No usable courses")

        inputs = client.get(f"/api/jobs/{old_id}/inputs").json()
        assert inputs["pasted_text"] == "too short\n---\nalso short"
        assert inputs["config"]["learning_rate"] == 5e-5 and inputs["error"] == "No usable courses"
        assert [f["name"] for f in inputs["files"]] == ["ds.txt", "junk.txt"]

        # Lecturer fixes the pasted text, drops junk.txt, keeps ds.txt.
        keep = [f["file"] for f in inputs["files"] if f["name"] == "ds.txt"]
        r = client.post("/api/jobs", data={"config": json.dumps({**cfg, "program_name": "Fixed"}), "pasted_text": SYLLABUS_B,
                                           "retry_of": old_id, "keep_files": json.dumps(keep)})
        assert r.status_code == 200, r.text
        new_id = r.json()["id"]
        new = client.get(f"/api/jobs/{new_id}").json()
        assert new["status"] == "queued" and new["config"]["program_name"] == "Fixed"
        assert [f["name"] for f in client.get(f"/api/jobs/{new_id}/inputs").json()["files"]] == ["ds.txt"]
        assert (Workspace(db.job_dir(new_id)).uploads / keep[0]).read_text() == SYLLABUS_A

        # The old job is hidden from the list and can't be re-run or resumed twice.
        assert client.get(f"/api/jobs/{old_id}").json()["replaced_by"] == new_id
        assert [j["id"] for j in client.get(f"/api/jobs?ids={old_id},{new_id}").json()] == [new_id]
        r = client.post("/api/jobs", data={"config": json.dumps(cfg), "pasted_text": SYLLABUS_B, "retry_of": old_id})
        assert r.status_code == 409
        assert client.post(f"/api/jobs/{old_id}/retry", json={}).status_code == 409

        worker.serve(("io",), once=True)
        assert client.get(f"/api/jobs/{new_id}").json()["stages"][0]["status"] == "done"


def test_edit_and_rerun_requires_failed_job():
    with TestClient(app) as client:
        job_id = _submit(client, review=False)
        r = client.post("/api/jobs", data={"config": json.dumps({"program_name": "x"}), "pasted_text": SYLLABUS_A,
                                           "retry_of": job_id})
        assert r.status_code == 409


def test_access_token(monkeypatch):
    from corecs_global import config

    monkeypatch.setenv("GLOBAL_ACCESS_TOKEN", "s3cret")
    config.get_settings.cache_clear()
    with TestClient(app) as client:
        assert client.get("/api/options").status_code == 401
        assert client.get("/api/options", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert client.get("/api/options?token=s3cret").status_code == 200
        assert client.get("/").status_code == 200                      # the page itself is public


def test_cors_origins_are_normalized():
    from corecs_global.api import cors_origins
    site = "https://knowledgedistillation-web.vercel.app"
    assert cors_origins("*") == ["*"]
    assert cors_origins(f"{site}/") == [site]
    assert cors_origins(f"[{site}]({site})") == [site]          # pasted as a Markdown link
    assert cors_origins(f' "{site}" , http://localhost:5173/x') == [site, "http://localhost:5173"]
    assert cors_origins("not-a-url") == []

"""Unit tests: config validation, estimates, queue/lanes, vault, progress/ETA, languages."""

import time

import pytest

from corecs_global import db, vault
from corecs_global.jobs import ConfigError, JobConfig, estimate, model_size_b
from corecs_global.languages import build_preamble, preset, script_ratio
from corecs_global.progress import Reporter, overall_fraction, pipeline_eta


def test_config_validation_and_teacher_lock():
    cfg = JobConfig.from_dict({"program_name": "  B.Sc. CS  ", "unknown_key": 1})
    assert cfg.program_name == "B.Sc. CS"
    assert cfg.display_name == "B.Sc. CS Tutor"
    with pytest.raises(ConfigError):
        JobConfig.from_dict({"program_name": ""})
    with pytest.raises(ConfigError, match="Teacher model"):
        JobConfig.from_dict({"program_name": "x", "teacher_model": "gpt-4o"})
    with pytest.raises(ConfigError):
        JobConfig.from_dict({"program_name": "x", "base_model": "not a repo id"})
    with pytest.raises(ConfigError):
        JobConfig.from_dict({"program_name": "x", "epochs": 99})


def test_estimate_scales_with_courses_and_model_size():
    small = JobConfig.from_dict({"program_name": "x", "base_model": "Qwen/Qwen2.5-1.5B-Instruct"})
    big = JobConfig.from_dict({"program_name": "x", "base_model": "Qwen/Qwen2.5-7B-Instruct"})
    assert model_size_b(small.base_model) == 1.5
    assert model_size_b("dicta-il/dictalm2.0-instruct") == 7.0
    e1, e3 = estimate(big, 1, 1), estimate(big, 3, 3)
    assert e3["rows"] == 3 * e1["rows"]
    assert e3["api_cost_usd"] > e1["api_cost_usd"] > 0
    assert estimate(small, 3, 3)["stage_seconds"]["train"] < e3["stage_seconds"]["train"]


def test_demo_job_halves_training_and_costs_nothing():
    real = JobConfig.from_dict({"program_name": "x"})
    demo = JobConfig.from_dict({"program_name": "x", "demo": True})
    assert not real.demo and demo.demo
    er, ed = estimate(real, 2, 2), estimate(demo, 2, 2)
    assert ed["train_steps"] == -(-er["train_steps"] // 2)
    assert ed["api_cost_usd"] == 0 and er["api_cost_usd"] > 0
    assert ed["total_seconds"] < er["total_seconds"]


def test_queue_lanes_and_staging():
    cfg = JobConfig.from_dict({"program_name": "x"}).to_dict()
    job_id = db.create_job(cfg, {"stage_seconds": {}})
    assert db.claim_next("w", ("io", "gpu")) is None          # staging jobs are invisible
    db.enqueue(job_id)
    assert db.claim_next("w", ("gpu",)) is None               # ingest is an io-lane stage
    job = db.claim_next("w", ("io",))
    assert job["id"] == job_id and job["status"] == "running" and job["current_stage"] == "ingest"
    assert db.claim_next("w2", ("io",)) is None               # claimed exactly once

    db.update_job(job_id, status="queued", next_stage="train")
    assert db.claim_next("w", ("io",)) is None
    assert db.claim_next("w", ("gpu",))["current_stage"] == "train"


def test_requeue_stale_resumes_same_stage():
    job_id = db.create_job(JobConfig.from_dict({"program_name": "x"}).to_dict(), {})
    db.enqueue(job_id)
    db.claim_next("w", ("io",))
    db.update_job(job_id, heartbeat=time.time() - 3600)
    assert db.requeue_stale(max_age_s=60) == [job_id]
    job = db.get_job(job_id)
    assert job["status"] == "queued" and job["next_stage"] == "ingest"


def test_vault_roundtrip_is_encrypted():
    job_id = db.create_job(JobConfig.from_dict({"program_name": "x"}).to_dict(), {})
    vault.store(job_id, vault.TEACHER_KEY, "sk-secret-123")
    assert b"sk-secret-123" not in db.get_secret(job_id, vault.TEACHER_KEY)
    assert vault.load(job_id, vault.TEACHER_KEY) == "sk-secret-123"
    vault.forget(job_id, vault.TEACHER_KEY)
    assert vault.load(job_id, vault.TEACHER_KEY) is None


def test_reporter_eta_and_pipeline_eta():
    job_id = db.create_job(JobConfig.from_dict({"program_name": "x"}).to_dict(),
                           {"stage_seconds": {"ingest": 10, "generate": 100, "train": 1000, "package": 100, "publish": 10, "deploy": 1}})
    r = Reporter(job_id, "generate")
    r.start("go")
    for i in range(1, 6):
        r._last_t = time.time() - 2           # pretend each unit took 2 s
        r.update(i, 10, force=True)
    p = db.get_job(job_id)["progress"]["generate"]
    assert p["status"] == "running" and p["done"] == 5
    assert 8 <= p["eta_s"] <= 12              # 5 units left x ~2 s
    job = db.get_job(job_id)
    assert pipeline_eta(job) == pytest.approx(10 + p["eta_s"] + 1000 + 100 + 10 + 1, abs=2)
    assert 0 < overall_fraction(job) < 0.1


def test_languages():
    he = build_preamble("Hebrew", "מדעי המחשב", ["מבני נתונים", "אלגוריתמים"])
    assert "תשובה סופית:" in he and "מבני נתונים" in he
    en = build_preamble("Klingon", "Xenolinguistics", ["Grammar"])
    assert "Answer in Klingon only" in en and preset("Klingon")["final_answer"] == "Final answer:"
    assert script_ratio("מהו TCP ולמה הוא חשוב?", preset("Hebrew")["script"]) > 0.5
    assert script_ratio("What is TCP and why does it matter?", preset("Hebrew")["script"]) == 0

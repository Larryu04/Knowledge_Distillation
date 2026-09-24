# project_memory.md — context anchor (read this first in any new session)

_Last updated: 2026-09-20. Update the changelog at the bottom whenever an issue is solved._

## 1. Goal
- Fine-tune a **Hebrew CS academic tutor** ("Student" LLM) with **4-bit NF4 QLoRA**.
- Training data = GPT-generated Q&A grounded in **syllabi of Academic College Ramat Gan** (6 courses).
- Base: `dicta-il/dictalm2.0-instruct` (Mistral-7B, Hebrew tokenizer). Hardware: **RTX 4070 Super, 12 GB** (also drives the Windows desktop).
- Top priority: **Hebrew robustness** (no language drift to Chinese/Spanish, no EOS loops). Earlier 3B attempt collapsed.
- Output style the model must learn: numbered step-by-step rationale, then a line starting `תשובה סופית:`.

## 2. Strict data format (DO NOT deviate)
Dataset file: `merged_6_courses_new.json` = **flat JSON list, 7,200 rows (1,200 per course)**. Every row has **exactly these 5 keys**, all **strings**:

```json
{
  "course_name": "מבוא לתקשורת נתונים",
  "domain": "Computer Networks",
  "instruction": "<Hebrew question>",
  "rationale": "1. ... 2. ... 3. ... (3-5 numbered points, incl. a concrete 'לדוגמה' example or code)",
  "output": "<short synthesized final answer, NOT a copy of rationale point 1>"
}
```
- `set(keys) == {course_name, domain, instruction, rationale, output}` — no extra/missing keys (validators reject).
- `domain` ∈ {Computer Networks, Cybersecurity, Operating Systems, C++, Java, Python}; `course_name` must match the course exactly:
  Networks=מבוא לתקשורת נתונים · Cyber=מבוא לסייבר · OS=עקרונות מערכות הפעלה · C++=מבוא למדעי המחשב · Java=תכנות מונחה עצמים בשפת Java · Python=תכנות מתקדם בשפת פייתון.
- Validation (in `generate_pilot_full-dataset.py`): instruction ≥15 chars; rationale has 3-5 numbered points (`1.`–`5.`) + a worked example marker; output ≥30 chars; no forbidden (out-of-syllabus) terms; near-duplicate instructions rejected.
- `course_name` / `domain` are **metadata only** — `train.py` never feeds them into the prompt; `domain` drives the stratified val split.
- Legacy V1/V4 datasets used a different 3-key capitalised schema (`Instruction/Rationale/Output`); **don't mix the two**.
- Generation: `generate_pilot_full-dataset.py` (gpt-4o-mini, per-course config in-file), `generate_pilot_dataset_for_24.py`. Raw syllabus inputs in `raw_courses_data/*.json`.
- Eval sets: **`hebrew_cs_pilot_dataset_24.json` = the OFFICIAL gold benchmark** (generated directly from the Ramat Gan syllabi, lowercase 5-key schema, 4/domain; what `evaluate_models.py` scores; not a subset of the corpus, 0 exact instruction overlaps). `hebrew_cs_eval_dataset_24.json` = **DEPRECATED** (generic CS questions, not syllabus-derived; only the historical v4/v5 benchmark — don't use it for new evaluations) ⊂ `hebrew_cs_eval_dataset_100.json` (eval-only, **never train on it** — leaks into the benchmark).

## 3. Training / eval config (train.py, v6 — uncommitted working copy)
- Out dir `out/dictalm2-tutor-qlora-v6`. LoRA r=16 / α=32 / dropout 0.05 on attn+MLP; **embed_tokens & lm_head frozen**.
- LR 1e-4 cosine, 2 epochs, **effective batch 16** (micro-batch 2 × accum 8, see §4), max_grad_norm 0.3, NEFTune α=5, `MAX_SEQ_LEN=832` (data max tok = 822).
- Prompt tokens masked (`labels=-100`); PAD ≠ EOS; explicit EOS in every label; identical `PREAMBLE` in `train.py` and `inference.py`.
- Stratified per-domain val split (`MIN_VAL_PER_DOMAIN=20`), weighted-domain sampler, early stopping on macro per-domain loss (excl. Uncategorized).
- Eval: `evaluate_models.py` — masked teacher-forcing loss, base vs adapter, per domain + blind A/B CSVs, scored on the official gold benchmark `hebrew_cs_pilot_dataset_24.json`. Results in `summary.md` (§9 = v6 final), `eval_results_v5/`, `eval_results_v6/`.
- Env: use `.venv\Scripts\python.exe` (Python 3.13). Bare `python` = msys2 3.14 with no ML wheels. Stack: torch 2.6.0+cu124, transformers 5.16.1, peft 0.20, bitsandbytes 0.50.2.

## 4. Changelog of solved issues (newest last)
- **V4 dataset cleaning** — removed CJK chars spliced into Hebrew, empty row, leftover "במצגת" slide refs, duplicate instructions (7,449 → 7,431 rows).
- **Domain-imbalance hypothesis tested & rejected** (v4-balanced, v5): fixed sampler `num_samples` bug, excluded "Uncategorized" from the metric, patience 3→5, per-domain eval floor, versioned `OUTPUT_DIR`. Both balanced runs still lost to the plain 09-10 run → not a sampler problem.
- **Real root cause found (09-14):** train/eval **style mismatch** — old training rationales were short (2-3 generic bullets, no examples); eval rationales long (3-5 dense bullets with worked examples). Fix = regenerate corpus with a richer teacher prompt → `merged_6_courses_new.json`.
- **`train.py` syntax error** (stray `0` after `TrainingArguments,`) — removed.
- **JSON list-coercion bug** — GPT returned a field (usually `rationale`) as a JSON *array* → `.strip()` raised `AttributeError` and crashed a generation run. Fixed in both generators: `generate_pilot_full-dataset.py::coerce_string_fields` (list → `"\n".join`, runs before validation so the numbered-point check turns it into a normal retry) and `generate_pilot_dataset_for_24.py::validate_example` (list → `" ".join`).
- **Obsolete keyword classifier removed** — `classify_domain` / `DOMAIN_KEYWORDS` (imported from `evaluate_models.py`; ~59% of rows fell into "Uncategorized") deleted. Domain now read straight from each row's `domain` key.
- **VRAM / BSOD hardware crashes (v6) — ✅ SOLVED & CONFIRMED (2026-09-20): the full v6 run completed 856/856 steps, including the step-107 eval/save that killed the earlier run, with no crash; Windows logged no Kernel-Power 41 / `nvlddmkm` / bugcheck events for the run.** Two crashes on 09-18: a hard OS freeze, then a BSOD `KMODE_EXCEPTION_NOT_HANDLED` (0x1E, kernel access violation) in `nvlddmkm.sys`.
  - **Root causes (chain):**
    1. **Longer sequences — measured.** `merged_6_courses_new.json` is ~2.3× longer per row than V4 (tok p50 565 / p99 707 / max 822 vs 241 / 331 / 444). At batch 8 the worst-case micro-batch peaks at ≈ **11.1 GB** allocated (extrapolated from 5.47 GB @ batch 1, 6.27 GB @ batch 2) on a 12 GB card that also drives the desktop → **VRAM overcommit**.
    2. **Spill into system RAM — inferred, not directly captured.** During the crashed run Python showed **14.9 GB** and system RAM 81% of 31 GB; under the fixed config Python holds only 0.9–1.6 GB working set. Consistent with WDDM demoting GPU allocations to shared system memory (the Windows shared-GPU-memory counter could not be read to confirm).
    3. **`paged_adamw_8bit` crashed the driver during eval/save — strongest lead, inferred.** The crashed run died while writing its first checkpoint (step 107, 09-18 5:42:49 PM, same second as an `nvlddmkm` Error event): `optimizer.pt` was 0.3 MB vs ~86 MB healthy, no `scheduler.pt` / `rng_state.pth` / `trainer_state.json`. Paged optimizers keep state in CUDA unified memory, which Windows/WDDM can't page; `torch.save` reading it under VRAM pressure is the likely trigger. v4/v5 used the same optimizer without issue but were not under memory pressure.
  - **Mitigations now active in `train.py`:** `PER_DEVICE_BATCH=2` × `GRAD_ACCUM=8` (effective batch 16 unchanged, so LR/schedule stay comparable to v4/v5) · `EVAL_BATCH=2` · `set_per_process_memory_fraction(0.85)` (~10.2 GB cap → catchable OOM instead of driver overcommit) + allocator GC config · worst-case-batch preflight fwd/bwd before training (measured 6.27 GB peak) · `adamw_bnb_8bit` instead of paged · `MAX_SEQ_LEN=832` (nothing dropped; don't go lower — 640 drops ~5% of the long, eval-style rows) · `find_resumable_checkpoint()` replacing `get_last_checkpoint` (skips checkpoints with missing files, a truncated `optimizer.pt`, or a different micro-batch size read from `trainer_state.json`; Trainer 5.16 saves in place, so a crash mid-save leaves partial folders). LoRA / 4-bit config deliberately untouched so v6 stays comparable.
  - Partial crash checkpoint kept as evidence at `out/_crashed/v6-checkpoint-107-partial` (adapter weights intact, optimizer state gone). Est. ~13 s/step, 856 steps ≈ 3 h.
  - Driver-level bugs can't be ruled out from code: consider `nvidia-smi -pl 180` (admin), no GPU overclock/undervolt, closing GPU-using apps during runs; if it recurs, clean driver reinstall (driver was 591.86).
- **v6 evaluation — ✅ SOLVED (2026-09-20): OFFICIAL RESULT on the syllabus-aligned gold benchmark.** `evaluate_models.py` on the v6 adapter (best checkpoint, step 749) vs. base, scored on `hebrew_cs_pilot_dataset_24.json`: overall masked teacher-forcing loss **0.8526 → 0.7024, Δ = +0.1502** (PPL 2.35 → 2.02), **positive in all 6 domains** — Networks +0.178, Cyber +0.178, OS +0.163, Java +0.157, C++ +0.113, Python +0.112 — and better on 23/24 items. Training: 856/856 steps, no early stop, val macro loss fell monotonically 0.8284 → 0.7907. Consistent with the 09-14 diagnosis (train/eval style mismatch; fix = regenerate the corpus with 3–5 point worked-example rationales). Full write-up: `summary.md` §9; artifacts in `eval_results_v6/`.
  - **Benchmark decision (2026-09-20, project owner):** the pilot set is the official gold benchmark because it is generated from the college syllabi — the project's actual target — whereas the old 24-item set is generic CS and deprecated. A planned cross-set run of v6 on the old set (converted copy `hebrew_cs_eval_dataset_24_v6_format.json`, output dir `eval_results_v6_gold24/`) was **aborted and is NOT part of the official results**; the output dir is empty, the converted file is unused (original file untouched).
  - **Scope caveats (don't over-cite):** (a) v4/v5 were scored on the deprecated generic set (base loss 1.072 vs 0.853 here), so their −0.052 is historical context, not a like-for-like baseline for v6; (b) the benchmark is same-syllabus / same-pipeline-family as the corpus, i.e. in-distribution *by design* — it measures syllabus-specific learning, not out-of-syllabus generalization; (c) no exact train/eval instruction overlap (0/24); one near-duplicate (Jaccard 0.87) — excluding overlap-prone items keeps Δ at +0.141…+0.151; (d) N=4/domain; (e) loss ≠ answer quality, `blind_evaluation.csv` not yet human-graded.

## 5. Open items / next
- Human-grade `eval_results_v6/blind_evaluation.csv` (0/48 cells graded) before trusting teacher-forcing loss as the sole target.
- Watch for the model emitting `תשובה:` instead of `תשובה סופית:`.
- Housekeeping: `.env` (API key) is untracked — keep it out of git; `merged_training_dataset_V4_clean.json` shows as deleted in the working tree; all recent work is uncommitted.

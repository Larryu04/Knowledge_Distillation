# Evaluation Summary — v5 (Corrected Balanced-Sampling) QLoRA Adapter vs. Base Model

> **Status update (2026-09-20):** §§1–8 record the **v5** evaluation and are kept as the
> historical record. They are superseded by the v6 run — see
> [§9 Final result — v6](#9-final-result--v6-richer-rationale-corpus). The v5 bottom line and
> the §6 recommendations are no longer current, and the §8 `train.py` syntax error was fixed
> on 2026-09-14.

**Date:** 2026-09-14
**Base model:** `dicta-il/dictalm2.0-instruct` (4-bit NF4)
**Fine-tuned:** base + QLoRA adapter `out/dictalm2-tutor-qlora-v5/adapter`
(v5 = v4-balanced with the `WeightedRandomSampler` `num_samples` bug, the `Uncategorized`
metric pollution, and the noisy-early-stop patience all fixed — see `train.py` and
[[hebrew-distillation-pitfalls]])
**Gold set:** `hebrew_cs_eval_dataset_24.json` (24 items, 6 CS domains, unchanged)
**Script:** `evaluate_models.py` (full run, exit code 0)

**Bottom line: fixing the sampling/early-stopping bugs did not fix the negative
transfer.** Cybersecurity, C++, Java, and Python are all still worse than the base model
— by roughly the same margin as the buggy v4-balanced run. Computer Networks and
Operating Systems are also *not* maintained: both are worse than they were under plain
(unbalanced) sampling. Across three attempts now, the original unbalanced run
(2026-09-10) remains the best-performing adapter on this gold set by a wide margin. The
evidence increasingly points away from "domain imbalance in the sampler" as the root
cause — see §4.

---

## 1. Method (unchanged)

- **Quantitative — masked teacher-forcing loss.** Cross-entropy per example, batch size 1,
  using the exact `make_example_builder` logic from `train.py`: prompt masked to `-100`,
  loss scored over (`Rationale` + `\n\nתשובה סופית: ` + `Output` + EOS). Perplexity = exp(loss).
- **Qualitative — blind test.** One sampled generation per instruction per model, blind
  A/B assignment. Exports `blind_evaluation.csv` (blank scores) and `answer_key.csv`.
- **Memory.** Base model fully released before the adapter model loads.
- **Domain mapping.** Same keyword classifier, same 24-item split as prior runs:
  Computer Networks 5, Cybersecurity 3, Operating Systems 4, C++ 4, Java 4, Python 4.

---

## 2. Categorical Masked Teacher-Forcing Loss — v5

| Domain | N | Base Loss | FT Loss | Δ (Base−FT) | Base PPL | FT PPL |
|---|--:|--:|--:|--:|--:|--:|
| Computer Networks | 5 | 0.9511 | 0.9111 | **+0.0400** | 2.59 | 2.49 |
| Cybersecurity | 3 | 0.9612 | 1.0231 | **−0.0619** | 2.61 | 2.78 |
| Operating Systems | 4 | 0.9416 | 0.9993 | **−0.0577** | 2.56 | 2.72 |
| C++ | 4 | 1.1552 | 1.2259 | **−0.0707** | 3.17 | 3.41 |
| Java | 4 | 1.2760 | 1.3531 | **−0.0771** | 3.58 | 3.87 |
| Python | 4 | 1.1508 | 1.2615 | **−0.1107** | 3.16 | 3.53 |
| **OVERALL (all 24)** | 24 | **1.0722** | **1.1243** | **−0.0521** | 2.92 | 3.08 |

*Δ > 0 ⇒ fine-tuned model assigns higher likelihood to the gold target (better).*

**Direct answer to the question:** No. Cybersecurity, C++, Java, and Python are all
*worse* than the base model, not improved — Cybersecurity is now clearly negative (it was
roughly neutral, +0.0058, in the original run). Computer Networks and Operating Systems
were **not maintained**: Networks' gain is less than half of the original run's, and OS's
regression is nearly 4x larger than the original run's.

---

## 3. Three-way comparison: unbalanced (09-10) vs. buggy-balanced v4 (09-13) vs. fixed-balanced v5 (09-14)

| Domain | Δ unbalanced (09-10) | Δ buggy-balanced v4 (09-13) | Δ fixed-balanced v5 (09-14) |
|---|--:|--:|--:|
| Computer Networks | **+0.0881** | +0.0298 | +0.0400 |
| Cybersecurity | **+0.0058** | −0.0365 | −0.0619 |
| Operating Systems | **−0.0146** | −0.0570 | −0.0577 |
| C++ | **−0.0374** | −0.0819 | −0.0707 |
| Java | **−0.0089** | −0.0551 | −0.0771 |
| Python | −0.0688 | −0.1301 | **−0.1107** |
| OVERALL | **−0.0025** | −0.0524 | −0.0521 |

*(Bold = best of the three for that row.)*

Fixing the sampler/metric bugs moved v5 a little closer to v4-balanced's own numbers in
a couple of domains (Networks, C++, Python improved slightly vs. the buggy run) and
slightly worse in others (Cybersecurity, Java) — but **v5's overall loss (−0.0521) is
statistically indistinguishable from the buggy v4-balanced run's (−0.0524)**, and every
single domain in both balanced attempts is worse than the original unbalanced run. Fixing
the mechanical bugs produced a correctly-executed experiment — it did not produce a
better adapter.

---

## 4. What the v5 training run itself looked like (this part worked as designed)

From `out/dictalm2-tutor-qlora-v5/checkpoint-1358/trainer_state.json`:

- Planned: 1558 steps (2 epochs × 779 steps/epoch — steps/epoch correctly grew from 442
  to 779 once the sampler's `num_samples` bug was fixed, confirming the majority domain is
  no longer being starved of draws).
- `EarlyStoppingCallback` (patience=5) fired at step 1358 (87% of plan); best checkpoint
  restored = step 388 (`eval_macro_domain_loss` = 0.7666, excluding `Uncategorized`).
- Unlike the v4-balanced run's ambiguous single-eval wobble, v5's `macro_domain_loss`
  **rose monotonically and substantially over 5 straight evaluations** after step 388
  (0.7666 → 0.7798 → 0.7907 → 0.7857 → 0.8099 → 0.8129) — a real, sustained internal-val
  degradation, not noise. Patience=5 correctly rode it out and confirmed the trend before
  stopping, rather than reacting to a single bad eval.

**So the mechanics are now sound**: the sampler gives the majority domain its fair
exposure plus genuine oversampling on top for the minority domains, the composite metric
no longer includes the off-target `Uncategorized` bucket, and early stopping is no longer
trigger-happy on noise. The adapter that this correctly-functioning pipeline converged on
is still worse on the external gold set than doing nothing special (plain shuffling,
pooled `eval_loss`) two runs ago.

---

## 5. Interpretation — the imbalance hypothesis looks wrong, not just poorly implemented

Three consistent data points now argue against "sampler-level domain imbalance" as the
real root cause of the original 2026-09-10 regression:

1. Correctly-implemented oversampling (v5) performs about the same as buggy oversampling
   (v4-balanced) — if the *amount* of minority-domain exposure were the lever, fixing the
   under-training bug should have shown a real improvement over v4-balanced. It didn't.
2. Oversampling minority domains made the **majority domains** (Networks, OS) worse too,
   not just held constant while minority domains improved. That's the opposite of what
   the imbalance hypothesis predicts.
3. The internal validation loss (drawn from the same GPT-4o-generated training
   distribution) improved with more training in both balanced runs, while the external,
   independently-curated 24-item gold set got worse. That gap — good in-distribution,
   worse out-of-distribution — is a train/gold-set **style mismatch** signature (phrasing,
   verbosity, formatting baked into the training `Rationale`/`Output` fields), not a
   coverage/imbalance signature. Reweighting the *same fixed pool* of examples repeats the
   same limited minority-domain phrasings more often; it doesn't add new information that
   would close a style gap.

## 6. Recommended next steps

1. **Stop iterating on sampler weights/formulas.** Two corrected attempts land in the same
   place; a third reweighting scheme is unlikely to diverge from this pattern.
2. **Revert to the 2026-09-10 configuration (plain random split, pooled `eval_loss` as
   `metric_for_best_model`) as the working baseline** — it remains the best adapter
   produced so far on every single domain of this gold set.
3. **Human-grade `blind_evaluation.csv` for all three runs before making further
   loss-driven decisions.** Teacher-forcing loss over exact gold phrasing penalizes a
   correct answer worded differently from the reference just as much as a wrong answer —
   it may not be the right optimization target here at all, especially for the
   programming-language domains where there's more than one natural way to phrase an
   explanation.
4. **If minority-domain quality still needs to improve, address it with data, not
   reweighting:** more or better-curated Cybersecurity/C++/Java/Python examples (new
   content), or few-shot/prompt-level scaffolding at inference time, rather than
   resampling the existing ~250–450 rows per domain more aggressively.
5. Keep `OUTPUT_DIR` versioning (v4/v5/…) going forward regardless of which direction is
   chosen next — it's what made this apples-to-apples three-way comparison possible.

---

## 7. Output Artifacts (all overwritten by this run)

| File | Contents |
|---|---|
| `loss_results.csv` | Per-domain + overall loss & perplexity, base vs. v5 adapter |
| `blind_evaluation.csv` | 24 rows, shuffled Answer_A/Answer_B, blank score columns |
| `answer_key.csv` | Secret A/B → model mapping |
| `eval_raw_results.json` | Per-question losses + full generations from both models |
| `eval_run.log` | Full stdout of this run |

---

## 8. Note: unrelated issue found in `train.py`

While reviewing the repo for this evaluation, `train.py` was found to currently contain a
syntax error — line 37 reads `TrainingArguments,0` (a stray `0` character), which makes
the file unparseable (`SyntaxError: invalid syntax`). This must have been introduced
*after* the v5 training run completed (the run clearly executed successfully to produce
the adapter evaluated above). `evaluate_models.py` does not import `train.py`, so this
did not affect this evaluation — but the file cannot currently be run again as-is. Flagging
per your instructions rather than silently editing it; let me know if you'd like it fixed.

---

## 9. Final result — v6 (richer-rationale corpus)

**Date:** 2026-09-20
**Adapter:** `out/dictalm2-tutor-qlora-v6/adapter` (best checkpoint, step 749 of 856)
**Base model:** `dicta-il/dictalm2.0-instruct` (4-bit NF4)
**Training data:** `merged_6_courses_new.json` — 7,200 rows (1,200 per course)
**Eval set (official gold benchmark):** `hebrew_cs_pilot_dataset_24.json` — 24 items, 4 per domain
**Script:** `evaluate_models.py` · results in `eval_results_v6/`

**Official benchmark.** `hebrew_cs_pilot_dataset_24.json` is this project's gold evaluation
set. Its items were generated directly from the Academic College of Ramat Gan course
syllabi, so it measures what the project is for: a tutor aligned to the college's own
syllabus distribution. The earlier `hebrew_cs_eval_dataset_24.json` (used for the v4/v5 reports
in §§1–8) is a hand-curated set of generic computer-science questions that are not derived
from the syllabi. It is **deprecated** as a benchmark; its results are kept as history only.

**Bottom line: on the official syllabus-aligned benchmark, the v6 adapter beats the base
model in all six domains (overall Δ = +0.1502), and training completed on the 12 GB card
with no hardware crashes.** This is the definitive v6 result. It is consistent with the
diagnosis in §5: regenerating the corpus with deep, multi-point rationales gives a clear gain,
where the v4/v5 corpora, scored on the deprecated set, showed negative transfer (overall
Δ −0.052). Because the benchmark changed, the v4/v5 deltas are historical context rather than a
like-for-like baseline (see §9.5).

### 9.1 What changed relative to v5

| | v5 | v6 |
|---|---|---|
| Training data | ~7.4k rows, short 2–3 bullet rationales | 7,200 rows, 3–5 numbered points with a worked example, synthesized `output` |
| Sequence length | tok p50 241 / max 444 | tok p50 565 / max 822 |
| Domain label | keyword classifier (~59% "Uncategorized") | read from each row's `domain` key |
| Micro-batch × accumulation | 8 × 2 | **2 × 8** (effective batch 16, unchanged) |
| Optimizer | `paged_adamw_8bit` | `adamw_bnb_8bit` |
| LoRA / LR / epochs | r=16, α=32, 1e-4, 2 | unchanged |

### 9.2 Masked teacher-forcing loss — v6 vs. base

| Domain | N | Base Loss | FT Loss | Δ (Base−FT) | Base PPL | FT PPL |
|---|--:|--:|--:|--:|--:|--:|
| Computer Networks | 4 | 0.8456 | 0.6677 | **+0.1778** | 2.33 | 1.95 |
| Cybersecurity | 4 | 0.8238 | 0.6453 | **+0.1785** | 2.28 | 1.91 |
| Operating Systems | 4 | 0.8541 | 0.6909 | **+0.1633** | 2.35 | 2.00 |
| Java | 4 | 0.8167 | 0.6598 | **+0.1569** | 2.26 | 1.93 |
| C++ | 4 | 0.7954 | 0.6824 | **+0.1130** | 2.22 | 1.98 |
| Python | 4 | 0.9797 | 0.8680 | **+0.1117** | 2.66 | 2.38 |
| **OVERALL** | 24 | **0.8526** | **0.7024** | **+0.1502** | 2.35 | 2.02 |

*Δ > 0 ⇒ the adapter assigns higher likelihood to the reference answer (better).*

- The adapter is better on **23 of 24 items**; the single exception (one C++ item, Δ −0.001)
  is a tie. Every domain has a positive mean.
- Python — the worst-regressing domain in v5 (Δ −0.111) — is positive in v6, though it and
  C++ show the smallest gains.

### 9.3 Training run

- **856 / 856 steps** (2 epochs); no early stopping. Effective batch 16, ≈ 3–3.5 h wall-clock.
- Validation `macro_domain_loss` fell monotonically over the first seven evaluations
  (step 107 → 749: 0.8284 → 0.7907) and plateaued at 0.7914 at step 856. The best checkpoint
  (step 749) was restored for the saved adapter. Macro and pooled loss agree to within 0.001,
  as expected with a balanced corpus.
- Unlike v5, internal validation and the external eval now move in the same direction.

### 9.4 Hardware stabilization

Two crashes on 2026-09-18 (a hard OS freeze, then a BSOD `KMODE_EXCEPTION_NOT_HANDLED` in
`nvlddmkm.sys`) traced back to the v6 corpus itself:

1. **Longer sequences (measured).** At batch 8 the worst-case micro-batch peaked at ≈ 11 GB
   (extrapolated from 5.47 GB @ batch 1 and 6.27 GB @ batch 2) on a 12 GB card that also
   drives the desktop — VRAM overcommit.
2. **Spill into system RAM (inferred).** The crashed run showed Python at 14.9 GB and system
   RAM at 81%; the fixed configuration holds 0.9–1.6 GB. Consistent with Windows demoting GPU
   allocations to shared memory.
3. **Paged optimizer during checkpoint save (strongest lead, inferred).** The run died while
   writing its first checkpoint (step 107): `optimizer.pt` was 0.3 MB instead of ~86 MB.
   `paged_adamw_8bit` keeps state in CUDA unified memory, which Windows cannot page.

Mitigations now in `train.py`:

| Mitigation | Setting |
|---|---|
| Micro-batch / accumulation | `PER_DEVICE_BATCH=2` × `GRAD_ACCUM=8`; `EVAL_BATCH=2` |
| VRAM cap | `set_per_process_memory_fraction(0.85)` (~10.2 GB) + allocator GC threshold |
| Worst-case preflight | forward+backward on the longest micro-batch before training (measured 6.27 GB peak) |
| Optimizer | `adamw_bnb_8bit` (non-paged) |
| Sequence bound | `MAX_SEQ_LEN=832` (no rows dropped) |
| Crash recovery | `find_resumable_checkpoint()` — skips incomplete, truncated, or batch-size-mismatched checkpoints |

**Outcome:** the full run completed all 856 steps, including the step-107 eval-and-save that
killed the previous run and seven further checkpoint cycles. Windows logged no `Kernel-Power 41`,
`nvlddmkm` error, or bugcheck events for the run's date. A driver-level fault cannot be ruled out
from code alone, so the hardware notes in `project_memory.md` (power limit, no overclock) still apply.

### 9.5 Scope and caveats

Read these before citing the headline number or comparing it with earlier runs.

- **Comparison with v4/v5.** v4/v5 were scored on the deprecated generic set, where base-model
  loss is different (1.072 vs. 0.853 here). Their deltas (e.g. overall −0.052) are historical
  context, not a baseline for v6. The base-vs-adapter delta within v6 is the valid measure.
- **In-distribution by design.** The benchmark is generated from the same syllabi, by the same
  `gpt-4o-mini` pipeline family, as the training corpus. A positive delta therefore shows the
  adapter learned the syllabus-specific style and content better than the base model — the
  project's goal — rather than general computer-science ability. Out-of-syllabus
  generalization is not measured and is not a project target.
- **Overlap check.** No exact instruction matches between the eval set and the training corpus
  (0 / 24). One eval item is a near-duplicate of a training instruction (token-Jaccard 0.87)
  and five exceed 0.6. Excluding items above 0.8 / 0.6 / 0.5 gives mean Δ of +0.151 / +0.141
  / +0.150, so the result does not depend on the overlap.
- **Small N.** Four items per domain: per-domain deltas are directionally reliable but noisy;
  the overall figure (N = 24) is the more stable number.
- **Loss ≠ answer quality.** Teacher-forcing loss rewards matching the reference wording.
  `eval_results_v6/blind_evaluation.csv` has **not yet been human-graded** (0 / 48 score cells filled).

### 9.6 Next steps

1. **Human-grade `eval_results_v6/blind_evaluation.csv`** for base vs. v6 before treating loss
   as the sole target.
2. Watch for the model emitting `תשובה:` instead of the trained `תשובה סופית:`.

### 9.7 Output artifacts

| File | Contents |
|---|---|
| `eval_results_v6/loss_results.csv` | Per-domain + overall loss and perplexity, base vs. v6 adapter |
| `eval_results_v6/blind_evaluation.csv` | 24 rows, shuffled Answer_A/Answer_B, blank score columns |
| `eval_results_v6/answer_key.csv` | A/B → model mapping |
| `eval_results_v6/eval_raw_results.json` | Per-question losses and full generations from both models |
| `out/dictalm2-tutor-qlora-v6/adapter` | Final adapter (best checkpoint, step 749) |

---

## 10. Result — v7 (12-course corpus)

**Date:** 2026-09-28
**Adapter:** `out/dictalm2-tutor-qlora-v7/adapter` (best checkpoint, step 1704 of 1710)
**Base model:** `dicta-il/dictalm2.0-instruct` (4-bit NF4)
**Training data:** `merged_12_courses.json` — 14,400 rows (1,200 per course × 12)
**Eval set (official gold benchmark):** `hebrew_cs_pilot_dataset_48.json` — 48 items, 4 per domain × 12
**Script:** `evaluate_models.py` · results in `eval_results_v7/`

**Bottom line: the v7 adapter beats the base model in all 12 domains (overall Δ = +0.2974,
loss 0.8292 → 0.5319, PPL 2.29 → 1.70) and on every one of the 48 items.** The six new
domains, which no earlier adapter had seen, gain as much as the original six do.

### 10.1 What changed relative to v6

| | v6 | v7 |
|---|---|---|
| Training data | 7,200 rows, 6 courses | 14,400 rows, 12 courses (same generator / format) |
| Sequence length | tok p50 565 / max 822 | tok p50 610 / p99 742 / max 866 |
| `MAX_SEQ_LEN` | 832 | **872** (832 would have dropped 7 rows) |
| Train sampler | weighted (inverse-sqrt domain frequency) | **plain shuffle** — corpus is exactly balanced (1,200/domain), all weights = 1.0, so sampling with replacement would only leave ~37% of rows unseen per epoch |
| `PREAMBLE` | names 6 subject areas | names all 12 (identical in `train.py` / `inference.py` / `evaluate_models.py`) |
| Steps | 856 (2 epochs) | 1,710 (2 epochs), eval/save every 213 |
| LoRA / LR / epochs / batch | r=16, α=32, 1e-4, 2, eff. 16 (2 × 8) | unchanged |

### 10.2 Masked teacher-forcing loss — v7 vs. base

| Domain | N | Base Loss | FT Loss | Δ (Base−FT) | Base PPL | FT PPL |
|---|--:|--:|--:|--:|--:|--:|
| Algorithms *(new)* | 4 | 0.9465 | 0.6203 | **+0.3262** | 2.58 | 1.86 |
| Operating Systems | 4 | 0.8443 | 0.5225 | **+0.3218** | 2.33 | 1.69 |
| Databases *(new)* | 4 | 0.8309 | 0.5092 | **+0.3217** | 2.30 | 1.66 |
| Automata Theory *(new)* | 4 | 0.8200 | 0.5031 | **+0.3169** | 2.27 | 1.65 |
| Java | 4 | 0.8648 | 0.5527 | **+0.3122** | 2.37 | 1.74 |
| Computer Architecture *(new)* | 4 | 0.9361 | 0.6344 | **+0.3016** | 2.55 | 1.89 |
| Software Engineering *(new)* | 4 | 0.7599 | 0.4639 | **+0.2960** | 2.14 | 1.59 |
| Python | 4 | 0.8294 | 0.5392 | **+0.2902** | 2.29 | 1.71 |
| Computer Networks | 4 | 0.8026 | 0.5131 | **+0.2895** | 2.23 | 1.67 |
| Data Structures *(new)* | 4 | 0.7557 | 0.4797 | **+0.2760** | 2.13 | 1.62 |
| C++ | 4 | 0.7535 | 0.4897 | **+0.2638** | 2.12 | 1.63 |
| Cybersecurity | 4 | 0.8069 | 0.5547 | **+0.2523** | 2.24 | 1.74 |
| **OVERALL** | 48 | **0.8292** | **0.5319** | **+0.2974** | 2.29 | 1.70 |

*Δ > 0 ⇒ the adapter assigns higher likelihood to the reference answer (better).*

- Better on **48 / 48 items**; all domain deltas lie in a narrow band (+0.25 … +0.33).
- New six domains: mean Δ +0.306; original six: mean Δ +0.288.
- The two hardest domains for the base model (Algorithms 0.95, Computer Architecture 0.94)
  are among the largest gains.

### 10.3 Training run

- **1,710 / 1,710 steps** (2 epochs); no early stopping; no crash. Worst-case preflight
  (2 × 872 tokens): 6.37 GB peak allocated, 6.71 GB reserved, under the 10.2 GB cap.
- Validation `macro_domain_loss` per eval (every 213 steps): 0.8224 → 0.8011 → 0.7876 →
  0.7748 → 0.7767 → 0.7725 → 0.7707 → **0.7695** (step 1704, restored as the final adapter).
  Gains flatten after ~step 850 (the second epoch adds only ~0.005), so more epochs are
  unlikely to help much. Train loss 1.075 → 0.668.
- Per-domain val loss at step 1704 ranges 0.721 (Java) … 0.838 (Algorithms).

### 10.4 Scope and caveats

- **v6's +0.150 (§9) and v7's +0.297 are not like-for-like.** The 48-item benchmark shares
  **no** questions with the 24-item set, and base loss differs (0.829 vs. 0.853). The valid
  adapter-vs-adapter comparison is §10.7 (both adapters on the same 48 items).
- **Overlap check.** 0 / 48 exact instruction matches with the training corpus; max
  within-domain token-Jaccard 0.667 (1 item > 0.6, 2 > 0.5). Excluding them: mean Δ +0.296
  (> 0.6 removed) / +0.294 (> 0.5 removed) — the result does not depend on the overlap.
- **In-distribution by design**, same as §9.5: the benchmark and corpus come from the same
  syllabi and `gpt-4o-mini` pipeline family.
- **Small N.** Four items per domain; the overall figure (N = 48) is the stable number.
- **Loss ≠ answer quality.** `eval_results_v7/blind_evaluation.csv` has not been human-graded.
- **Generation format.** 47 / 48 v7 answers follow the numbered-rationale → final-line
  structure, but the final line starts `תשובה:`, **never** the trained `תשובה סופית:`
  (0 / 48; v6 was also 0 / 24). One Operating Systems answer has no final line. No CJK or
  Cyrillic characters in any generation. Spot-checks show occasional term typos
  (e.g. "OIS" for OSI, "SYN-AK").

### 10.5 Next steps

1. Human-grade `eval_results_v7/blind_evaluation.csv` (48 rows).
2. Decide on the `תשובה:` vs. `תשובה סופית:` prefix: either accept `תשובה:` (change
   `FINAL_ANSWER_PREFIX` and any parser) or investigate why the trained prefix never appears.

### 10.6 Output artifacts

| File | Contents |
|---|---|
| `eval_results_v7/loss_results.csv` | Per-domain + overall loss and perplexity, base vs. v7 adapter |
| `eval_results_v7/blind_evaluation.csv` | 48 rows, shuffled Answer_A/Answer_B, blank score columns |
| `eval_results_v7/answer_key.csv` | A/B → model mapping |
| `eval_results_v7/eval_raw_results.json` | Per-question losses and full generations from both models |
| `out/dictalm2-tutor-qlora-v7/adapter` | Final adapter (best checkpoint, step 1704) |
| `eval_results_v7/v6_vs_v7_loss_comparison.csv` | v6 vs. v7 side by side on the 48 items (§10.7), built by `compare_loss_results.py` |
| `eval_results_v6_on48/` | v6 adapter scored on the 48-item benchmark (loss CSV, blind CSVs, raw JSON) |

### 10.7 Direct comparison — v6 vs. v7 on the same 48-item benchmark

**Date:** 2026-09-28 · v6 re-scored with
`evaluate_models.py --adapter-dir out/dictalm2-tutor-qlora-v6/adapter --out-dir eval_results_v6_on48/`;
merged by `compare_loss_results.py` → `eval_results_v7/v6_vs_v7_loss_comparison.csv`.
Both runs score the same frozen base model; per-item base losses are identical (max diff 0.0),
so this comparison is like-for-like.

**Bottom line: v7 scaled to 12 domains without losing ground on the original 6. It beats v6
on every one of the 48 items: +0.124 mean loss gain on the new domains and +0.044 on the
original ones.** This is the baseline evidence for the 12-course expansion.

| Domain | Group | Base | v6 | v7 | Δ v6 vs. base | Δ v7 vs. base | **Gain v7 over v6** |
|---|---|--:|--:|--:|--:|--:|--:|
| Computer Networks | original | 0.8026 | 0.5569 | 0.5131 | +0.2457 | +0.2895 | **+0.0438** |
| Cybersecurity | original | 0.8069 | 0.6064 | 0.5547 | +0.2005 | +0.2523 | **+0.0517** |
| Operating Systems | original | 0.8443 | 0.5656 | 0.5225 | +0.2787 | +0.3218 | **+0.0431** |
| C++ | original | 0.7535 | 0.5186 | 0.4897 | +0.2350 | +0.2638 | **+0.0288** |
| Java | original | 0.8648 | 0.5954 | 0.5527 | +0.2694 | +0.3122 | **+0.0427** |
| Python | original | 0.8294 | 0.5942 | 0.5392 | +0.2352 | +0.2902 | **+0.0550** |
| Data Structures | new | 0.7557 | 0.5705 | 0.4797 | +0.1852 | +0.2760 | **+0.0908** |
| Algorithms | new | 0.9465 | 0.7596 | 0.6203 | +0.1869 | +0.3262 | **+0.1393** |
| Computer Architecture | new | 0.9361 | 0.7377 | 0.6344 | +0.1984 | +0.3016 | **+0.1032** |
| Databases | new | 0.8309 | 0.6273 | 0.5092 | +0.2036 | +0.3217 | **+0.1181** |
| Automata Theory | new | 0.8200 | 0.7089 | 0.5031 | +0.1111 | +0.3169 | **+0.2057** |
| Software Engineering | new | 0.7599 | 0.5493 | 0.4639 | +0.2106 | +0.2960 | **+0.0854** |
| **Original 6 (mean)** | | 0.8169 | 0.5729 | 0.5286 | +0.2441 | +0.2883 | **+0.0442** |
| **New 6 (mean)** | | 0.8415 | 0.6589 | 0.5351 | +0.1826 | +0.3064 | **+0.1238** |
| **OVERALL (48)** | | 0.8292 | 0.6159 | 0.5319 | +0.2134 | +0.2974 | **+0.0840** |

*Gain = v6 loss − v7 loss; > 0 ⇒ v7 assigns higher likelihood to the reference answer.*

**New domains — the expansion worked.** v7's gain over v6 is ~3× larger on the new six
(+0.124) than on the original six (+0.044), and the largest per-domain gains are all new
domains: Automata Theory +0.206, Algorithms +0.139, Databases +0.118, Computer
Architecture +0.103. v7 improves on v6 on 24 / 24 new-domain items.

**Original domains — no catastrophic forgetting.** Halving the share of original-domain data
(from 100% to 50% of the corpus) did not cost the original domains anything: v7 is better
than v6 on all 6 of them and on 24 / 24 of their items. The original-6 mean loss went
0.5729 → 0.5286.

**Important nuance: v6 was *not* weak on the new domains.** Without ever training on them,
v6 already beat the base model there by +0.183 (every one of the 24 new-domain items),
i.e. about 60% of v7's +0.306. Most of v6's improvement is therefore domain-independent:
learning the corpus's Hebrew tutor style and the numbered-rationale → final-answer format,
which the teacher-forcing loss rewards on any domain. The new data adds real
domain-specific gain on top of that — largest where v6 transferred worst (Automata Theory:
v6 only +0.111, v7 +0.317) — but it should not be described as taking v6 from "can't answer"
to "can answer".

**Confounds to keep in mind.**
- v7 trained for twice as many optimizer steps (1,710 vs. 856). Part of the +0.044 gain on
  the original domains may come from more total training (cross-domain transfer and style
  reinforcement), not only from the new data.
- Same caveats as §10.4: N = 4 per domain, in-distribution benchmark, loss ≠ human-judged
  answer quality (neither `eval_results_v6_on48/` nor `eval_results_v7/` blind CSV is graded).
- Generation format is unchanged: v6 also ends 48 / 48 answers with `תשובה:`, never
  `תשובה סופית:`; neither adapter shows CJK/Cyrillic drift.

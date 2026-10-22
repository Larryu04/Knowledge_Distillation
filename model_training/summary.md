# Evaluation Summary — v5 (Corrected Balanced-Sampling) QLoRA Adapter vs. Base Model

> **Project status (2026-10-01): model training is complete across all 23 domains.** The final
> adapter is **v8** (`out/dictalm2-tutor-qlora-v8/adapter`, 27,600 rows / 23 courses). On the
> 92-item, 23-domain benchmark it beats the base model on 92 / 92 items (loss 0.8598 → 0.5584)
> and beats v7 by +0.078 on the 11 new domains, with a small −0.011 regression on the 12
> prior ones — see [§11](#11-final-result--v8-23-course-corpus). The **CoreCS** desktop app
> (Tauri UI ↔ Rust core ↔ llama.cpp sidecar, sibling repo `C:\Users\Larry\corecs`) is
> runs the tutor fully offline: verified end-to-end at ~93 tokens/s on Vulkan, and the v8
> model is now exported (`corecs-v8-q4_k_m.gguf`) and loaded in the app. The standalone NSIS
> installer builds (18.5 MB, model shipped separately). Remaining: verify a run of the
> installed app (§11.8).
>
> **Final evaluation (2026-10-12):** in a pairwise, reference-guided LLM-as-a-judge benchmark
> on the same 92 items (`gpt-4o-mini` judge, A/B position-swapped), v8 **wins or ties 84.8%**
> of comparisons against its own base model (53 W / 25 T / 14 L). It loses clearly to larger
> and newer models (DictaLM-3.0 12B Instruct, DictaLM-3.0 24B Thinking, GPT-4o), as expected
> for a 7B model. See [§12](#12-final-benchmark--pairwise-llm-as-a-judge-v8-vs-base-and-larger-models).
>
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

---

## 11. Final result — v8 (23-course corpus)

**Date:** 2026-10-01
**Adapter:** `out/dictalm2-tutor-qlora-v8/adapter` (best checkpoint, step 2454 of 3278)
**Base model:** `dicta-il/dictalm2.0-instruct` (4-bit NF4)
**Training data:** `merged_all_courses.json` — 27,600 rows (1,200 per course × 23)
**Eval set (official gold benchmark):** `hebrew_cs_pilot_dataset_92.json` — 92 items, 4 per domain × 23
**Scripts:** `evaluate_models.py` (loss only, `--skip-generation`) · `compare_loss_results.py`
→ `eval_results_v8/v6_vs_v7_vs_v8_comparison.csv`

**Bottom line: v8 learned the 11 new domains and kept nearly all of its prior knowledge.** It
is the best adapter overall (loss 0.5584 vs. v7 0.5899 and v6 0.6433; base 0.8598) and the best
on all 11 new domains (+0.078 mean gain over v7, 42 / 44 items). On the 12 prior domains it is
slightly behind v7 (−0.011 mean, i.e. ~2% higher loss) but still retains ~97% of v7's gain over
the base model there and remains clearly ahead of v6.

### 11.1 What changed relative to v7

| | v7 | v8 |
|---|---|---|
| Training data | 14,400 rows, 12 courses | 27,600 rows, 23 courses (+11 new, same generator / format, `gpt-4o-mini`) |
| `PREAMBLE` | names 12 subject areas | names all 23 (identical in `train.py` / `inference.py` / `evaluate_models.py`); ~88 tokens longer |
| Sequence length (incl. preamble) | tok p50 610 / p99 742 / max 866 | tok p50 696 / p99 821 / max 1079 |
| `MAX_SEQ_LEN` | 872 | **960** (872 would drop 43 rows; 960 drops 1 outlier) |
| Steps | 1,710 (2 epochs), eval/save every 213 | 3,278 (2 epochs), eval/save every 409 |
| Benchmark | 48 items, 12 domains | **92 items, 23 domains** (`generate_pilot_dataset_for_92.py`) |
| LoRA / LR / epochs / batch | r=16, α=32, 1e-4, 2, eff. 16 (2 × 8) | unchanged |

New domains: Discrete Mathematics, Logic and Set Theory, Graph Theory, Computability and
Complexity, Introduction to Cryptography, Introduction to AI, Introduction to Machine Learning,
Deep Learning, AI-Based Software Development, Information Systems Foundations, Computational
Biology Applications.

### 11.2 Training run

- **3,278 / 3,278 steps** (2 epochs); no early stopping.
- Validation `macro_domain_loss` per eval (every 409 steps): 0.8162 → 0.7998 → 0.7921 →
  0.7879 → 0.7859 → **0.7834** (step 2454, restored as the final adapter) → 0.7844 → 0.7838.
  Flat from ~step 2000; the last ~800 steps added nothing. Train loss 1.084 → 0.631.
- Not comparable to v7's 0.7695: the val split now covers 23 domains and the prompt is longer.

### 11.3 Evaluation protocol — "fair" re-scoring

All three adapters were scored on the same 92 items with the **same 23-course `PREAMBLE`**
(v6 and v7 into `eval_results_v6_fair/` and `eval_results_v7_fair/`, v8 into
`eval_results_v8/`). Per-item base-model losses are identical across the three runs (max
difference 0.0), so the comparison is like-for-like. Because the preamble and item set
differ from §10, the v7 numbers here are **not** the same as v7's §10 numbers.

### 11.4 Results — v6 vs. v7 vs. v8 (92 items)

| Domain | Group | Base | v6 | v7 | v8 | Gain v8 over v7 | Gain v8 over v6 | Best |
|---|---|--:|--:|--:|--:|--:|--:|---|
| Computer Networks | prior | 0.8974 | 0.6355 | 0.6014 | 0.6234 | −0.0219 | +0.0121 | v7 |
| Cybersecurity | prior | 0.8319 | 0.5909 | 0.5701 | 0.5373 | **+0.0329** | +0.0536 | v8 |
| Operating Systems | prior | 0.9014 | 0.6205 | 0.5819 | 0.5965 | −0.0146 | +0.0240 | v7 |
| C++ | prior | 0.8031 | 0.5796 | 0.5474 | 0.5640 | −0.0166 | +0.0157 | v7 |
| Java | prior | 0.9241 | 0.5808 | 0.5555 | 0.5770 | −0.0215 | +0.0038 | v7 |
| Python | prior | 0.7620 | 0.5108 | 0.4961 | 0.5002 | −0.0040 | +0.0107 | v7 |
| Data Structures | prior | 0.7948 | 0.6196 | 0.5023 | 0.5138 | −0.0115 | +0.1058 | v7 |
| Algorithms | prior | 0.8816 | 0.7527 | 0.5714 | 0.5829 | −0.0115 | +0.1698 | v7 |
| Computer Architecture | prior | 0.9356 | 0.7409 | 0.6456 | 0.6475 | −0.0019 | +0.0934 | v7 |
| Databases | prior | 0.9184 | 0.6272 | 0.5114 | 0.5281 | −0.0166 | +0.0991 | v7 |
| Automata Theory | prior | 0.8774 | 0.6980 | 0.5535 | 0.5763 | −0.0228 | +0.1217 | v7 |
| Software Engineering | prior | 0.8250 | 0.5800 | 0.4985 | 0.5165 | −0.0180 | +0.0635 | v7 |
| Discrete Mathematics | new | 0.7639 | 0.6081 | 0.5754 | 0.5163 | **+0.0592** | +0.0918 | v8 |
| Logic and Set Theory | new | 0.7671 | 0.5972 | 0.5719 | 0.4395 | **+0.1324** | +0.1577 | v8 |
| Graph Theory | new | 0.8618 | 0.7161 | 0.6501 | 0.5881 | **+0.0620** | +0.1280 | v8 |
| Computability and Complexity | new | 0.9456 | 0.7563 | 0.7267 | 0.5995 | **+0.1272** | +0.1568 | v8 |
| Introduction to Cryptography | new | 0.8138 | 0.6063 | 0.5848 | 0.4874 | **+0.0975** | +0.1189 | v8 |
| Introduction to AI | new | 0.7096 | 0.5510 | 0.5181 | 0.4761 | **+0.0419** | +0.0748 | v8 |
| Introduction to Machine Learning | new | 0.8511 | 0.6798 | 0.6590 | 0.5587 | **+0.1003** | +0.1211 | v8 |
| Deep Learning | new | 0.9891 | 0.7656 | 0.7263 | 0.6418 | **+0.0845** | +0.1238 | v8 |
| AI-Based Software Development | new | 0.9797 | 0.7134 | 0.6959 | 0.6402 | **+0.0557** | +0.0731 | v8 |
| Information Systems Foundations | new | 0.8629 | 0.5883 | 0.5691 | 0.5388 | **+0.0303** | +0.0494 | v8 |
| Computational Biology Applications | new | 0.8788 | 0.6772 | 0.6554 | 0.5943 | **+0.0612** | +0.0830 | v8 |
| **Prior 12 (mean)** | | 0.8627 | 0.6281 | 0.5529 | 0.5636 | **−0.0107** | +0.0644 | v7 |
| **New 11 (mean)** | | 0.8567 | 0.6599 | 0.6303 | 0.5528 | **+0.0775** | +0.1071 | v8 |
| **OVERALL (92)** | | 0.8598 | 0.6433 | 0.5899 | 0.5584 | **+0.0315** | +0.0849 | v8 |

*Loss = masked teacher-forcing loss (lower is better). Gain = older loss − v8 loss; > 0 ⇒ v8 is better.*

Item-level counts (from `eval_raw_results.json`):

| | v8 better than base | v8 better than v7 | v8 better than v6 |
|---|--:|--:|--:|
| Prior 12 domains (48 items) | 48 / 48 | 15 / 48 | 40 / 48 |
| New 11 domains (44 items) | 44 / 44 | 42 / 44 | 44 / 44 |

### 11.5 Analysis

**New domains — the expansion worked.** v8 is the best adapter on all 11 new domains. Mean
loss on them fell from 0.6303 (v7) to 0.5528 (−12%), and v8 beats v7 on 42 of 44 items. The
largest gains are mostly in the formal / theoretical courses: Logic and Set Theory (+0.132), Computability and Complexity (+0.127), Introduction to Machine
Learning (+0.100), and Introduction to Cryptography (+0.097). The smallest gains are
Information Systems Foundations (+0.030) and Introduction to AI (+0.042). On the new domains v8 is now on par with the prior ones (0.553 vs.
0.564 mean loss).

**Prior domains — mild dilution, no catastrophic forgetting.** v8 is slightly worse than v7 on
11 of 12 prior domains (only Cybersecurity improved, +0.033), with a mean regression of −0.011
loss (~2%), and it beats v7 on only 15 of 48 items. The effect is small but consistent in
direction, so it should be read as real, not noise: the prior courses fell from 100% to 52% of
the corpus. In context it is minor: v8 keeps a +0.299 gain over the base model on these domains
(v7: +0.310, i.e. **~97% retained**), still beats the base model on 48 / 48 items, and is still
clearly better than v6 (+0.064). The largest drops (Automata Theory −0.023, Computer Networks
−0.022, Java −0.022) are each smaller than any single new-domain gain.

**Overall.** v8 is the best adapter: overall loss 0.5584 (Δ vs. base +0.301, PPL 2.36 → 1.75),
+0.031 over v7 and +0.085 over v6, and better than base on all 92 items. Scaling from 12 to 23
courses traded a ~2% loss increase on the old domains for a ~12% decrease on the new ones,
which is the right trade for a tutor that must cover the whole program.

### 11.6 Scope and caveats

- **Preamble handicap for older adapters.** v6 and v7 were scored with the 23-course preamble
  they were not trained on. This is the fair way to compare them on one prompt, but it may cost
  them a little; if so, v8's prior-domain regression versus v7's *native* prompt is slightly
  larger than −0.011.
- **Steps confound.** v8 ran ~1.9× more optimizer steps than v7 (3,278 vs. 1,710), but the
  prior domains received roughly the same number of updates in both runs (their share halved),
  which matches the small regression rather than a gain.
- **Overlap check.** 0 / 92 exact instruction matches with the training corpus; max
  within-domain token-Jaccard 0.579 (3 items > 0.5, none > 0.6).
- **In-distribution by design**, same as §9.5 / §10.4: benchmark and corpus come from the same
  syllabi and the `gpt-4o-mini` pipeline family.
- **Small N.** Four items per domain; group means (N = 48 / 44) and the overall figure
  (N = 92) are the stable numbers.
- **Loss ≠ answer quality.** This run was loss-only (`--skip-generation`): no blind A/B sheets
  were produced, so v8's generation format (incl. the `תשובה:` vs. `תשובה סופית:` prefix, §10.4)
  and answer quality have not been checked yet.

### 11.7 Output artifacts

| File | Contents |
|---|---|
| `eval_results_v8/v6_vs_v7_vs_v8_comparison.csv` | Three-way comparison (§11.4), built by `compare_loss_results.py` |
| `eval_results_v8/loss_results.csv`, `eval_raw_results.json` | v8 vs. base, per domain / per item |
| `eval_results_v7_fair/`, `eval_results_v6_fair/` | v7 / v6 re-scored on the 92 items with the 23-course preamble |
| `hebrew_cs_pilot_dataset_92.json` | Official gold benchmark (92 items, 23 domains) |
| `out/dictalm2-tutor-qlora-v8/adapter` | Final adapter (best checkpoint, step 2454) |

### 11.8 Project completion status

| Component | Status |
|---|---|
| Training corpus (23 courses, 27,600 rows) | ✅ done |
| v8 adapter, trained and benchmarked on all 23 domains | ✅ done (final model) |
| CoreCS app: UI ↔ Rust core ↔ llama.cpp `llama-server` sidecar | ✅ verified end-to-end with the v7 GGUF (~93 tokens/s, Vulkan, RTX 4070 SUPER) |
| Standalone NSIS installer | ✅ builds (`CoreCS_0.1.0_x64-setup.exe`, 18.5 MB; model distributed separately) |
| v8 → GGUF export and load in CoreCS | ✅ done 2026-10-01 — `corecs-v8-q4_k_m.gguf` (Q4_K_M, 4.37 GB) + profile with the 23-subject preamble, selectable in the model picker |
| Installed-app run (from `setup.exe`) | ⏳ not yet verified |
| Generation check: LLM-as-a-judge benchmark on generated answers | ✅ done 2026-10-12 (§12) |
| Human validation of the judge (20-question blind sheet) | ⏳ open |

The ~93 tokens/s figure was measured with the v7 GGUF; v8 has the identical architecture and
quantization, so the same speed is expected, but it has not been re-measured separately.

v8 was exported with (from this repo, with its `.venv`):

```powershell
python ..\corecs\scripts\export_gguf.py --adapter out/dictalm2-tutor-qlora-v8/adapter --name corecs-v8
```

---

## 12. Final benchmark — pairwise LLM-as-a-judge (v8 vs. base and larger models)

**Date:** 2026-10-11 → 2026-10-12
**Benchmark:** `hebrew_cs_pilot_dataset_92.json` (92 items, 4 per domain × 23), same gold set as §11
**Scripts:** `generate_answers.py` (answers) → `judge_eval.py` (verdicts) → `plot_metrics.py` (figures)
**Outputs:** `eval_results_judge/` (answers, verdicts, `judge_summary.csv`, `judge_items.csv`), `figures/`

**Bottom line.** Judged on generated answers rather than teacher-forced loss, **v8 wins or ties
84.8% of head-to-head comparisons against its own base model** (DictaLM-2.0-instruct 7B without
the adapter): 53 wins, 25 ties, 14 losses. This is the first evaluation of this project that scores
the answers the tutor actually produces, and it confirms that the loss gains of §9–§11
translate into answers a grader prefers. Against larger and newer models (DictaLM-3.0 12B
Instruct, DictaLM-3.0 24B Thinking, GPT-4o), v8 loses clearly (win-or-tie 12.0%, 2.2% and 4.3%).
That is the expected ceiling for a 7B model, and §12.5 discusses what these comparisons do and
do not show.

### 12.1 Why a judge-based evaluation

Masked teacher-forcing loss (§1, §9–§11) has two limitations for a final evaluation:

1. **It cannot compare models with different tokenizers.** Loss and perplexity are measured per
   token, so the values for DictaLM-2.0, DictaLM-3.0 and GPT-4o are on incomparable scales.
2. **It does not measure answer quality.** It rewards reproducing the reference wording under
   teacher forcing, not a correct answer generated freely.

Pairwise comparison of generated answers, graded against the gold reference, avoids both
problems: every model is scored on the same text-level output, independent of its tokenizer.

### 12.2 Protocol

**Contestants.** Every model received the identical prompt: the 23-subject `PREAMBLE` from
`train.py` / `inference.py`, followed by the question, as a single user turn.

| Model | Parameters | Runtime | Generation cap |
|---|---|---|---|
| **v8** (DictaLM-2.0-instruct + v8 QLoRA adapter) | 7B | transformers, 4-bit NF4, fully on GPU | 1,024 new tokens |
| DictaLM-2.0-instruct, no adapter (sanity check) | 7B | transformers, 4-bit NF4, fully on GPU | 1,024 new tokens |
| DictaLM-3.0-Nemotron-12B-Instruct | 12B | llama.cpp `llama-server`, Q4_K_M GGUF, fully on GPU | 1,024 tokens |
| DictaLM-3.0-24B-Thinking | 24B | llama.cpp `llama-server`, Q4_K_M GGUF, split GPU / system RAM | 2,048 reasoning + 1,024 answer tokens |
| GPT-4o | proprietary | OpenAI API | 1,024 tokens |

Sampling: temperature 0.3, top-p 0.9, and a fixed per-item seed for all models. The local 7B models
also use `inference.py`'s repetition guards (`repetition_penalty` 1.15, `no_repeat_ngram_size` 4).
Dicta has not released a 24B *Instruct* model; the 24B is available only as Base or
Thinking. The Thinking variant was used, its reasoning block was discarded, and only the final
answer was judged.

**Judge.** `gpt-4o-mini` at temperature 0, in JSON mode. The judge is given the question, the gold
reference answer (the item's `rationale` plus a `תשובה סופית:` line built from its `output`) and
two anonymous candidate answers, A and B. It is instructed to rank by (1) correctness against the
reference, (2) completeness, and (3) clarity in correct Hebrew. It is told explicitly to ignore
length, formatting, and the wording of the final-answer line, and not to reward phrasing that
merely resembles the reference. It returns `A`, `B` or `tie`, with a short justification.

**Position swapping.** Every pair is judged twice, once with v8 as A and once with v8 as B.
In each order v8 receives +1 for a win, 0 for a tie and −1 for a loss. The item outcome is the
sign of the sum. A verdict that reverses when the order is swapped therefore counts as a
**tie**, so the result cannot be driven by a preference for one position. The cost is
4 × 92 × 2 = 736 judge calls in total. An empty answer would have counted as a loss without
calling the judge, but no model produced one.

### 12.3 Results

| v8 vs. | Params | v8 wins | Ties | v8 losses | Win rate | Win-or-tie rate |
|---|---|--:|--:|--:|--:|--:|
| **DictaLM-2.0-instruct (no fine-tuning)** | 7B | **53** | 25 | 14 | **57.6%** | **84.8%** |
| DictaLM-3.0 Nemotron-12B-Instruct | 12B | 4 | 7 | 81 | 4.3% | 12.0% |
| DictaLM-3.0 24B-Thinking | 24B | 0 | 2 | 90 | 0.0% | 2.2% |
| GPT-4o | proprietary | 0 | 4 | 88 | 0.0% | 4.3% |

*N = 92 items per opponent; each item was judged in both A/B orders. Source:
`eval_results_judge/judge_summary.csv`; per-item outcomes are in `judge_items.csv`.*

**v8 vs. its base model, by domain group:**

| Group | Wins | Ties | Losses | Win-or-tie |
|---|--:|--:|--:|--:|
| Prior 12 domains (48 items) | 29 | 14 | 5 | 89.6% |
| New 11 domains (44 items) | 24 | 11 | 9 | 79.5% |

### 12.4 Primary result: the fine-tuning works

Against the identical base model, v8 is preferred more than four times as often as it is
beaten (53 vs. 14), and it wins or ties 78 of 92 questions. The base model and v8 share the
same weights, tokenizer, quantization, prompt and sampling settings. The only difference is
the v8 QLoRA adapter, so the improvement can be attributed to the fine-tuning itself. The
advantage holds in both domain groups. It is larger on the 12 domains that have been in the
corpus since v7 (89.6%) than on the 11 added in v8 (79.5%), which is consistent with the
longer exposure of the earlier domains across successive training rounds.

The result is not explained by the judge preferring longer answers. v8's answers are *shorter*
than the base model's on 79 of 92 items (median 862 vs. 1,323 characters), yet the judge still
prefers v8. The judge can therefore reward a concise correct answer over a longer one.

### 12.5 Comparison with larger and newer models

v8 loses clearly to all three larger opponents. We read this as follows:

- **An expected ceiling for a 7B model.** The opponents are 1.7× to 3.4× larger (or, for GPT-4o,
  a frontier-scale proprietary model). They are also a newer model generation: DictaLM-3.0 is
  built on Mistral-Small-3.1 and Nemotron bases, while v8's base is from the Mistral-7B era. One
  of them also reasons explicitly before answering. A domain adapter trained on 27,600 rows does
  not close a gap of that size, and this benchmark sets realistic expectations for what a 7B
  tutor can do.
- **Evidence against a self-preference bias in the judge.** v8 was distilled from data generated
  by `gpt-4o-mini`, the same model that acts as judge, so a judge that favoured familiar style
  would have favoured v8. It did not: the judge consistently preferred the stronger models. This
  reduces the main concern about using a same-family judge. It does not establish that the judge
  is accurate (see §12.6).
- **Not a controlled scaling study.** The opponents differ from v8 in parameter count, base
  model, training data and inference method all at once, and they are quantized differently (Q4_K_M
  GGUF vs. 4-bit NF4). The results are *consistent with* the advantage expected from larger
  and newer models, but they do not isolate the effect of scale. The fair framing for v8 is
  its position in the cost/quality trade-off: it is the only contestant that runs fully offline
  inside the CoreCS app at interactive speed on a consumer GPU.

### 12.6 Scope and caveats

- **Length confound with the larger models.** v8 writes the shortest answers of all contestants
  (median 862 characters, vs. 1,390 for GPT-4o, 1,503 for the 12B and 2,283 for the 24B). It is
  shorter than GPT-4o and the 24B on 92 / 92 items and shorter than the 12B on 88 / 92. LLM judges
  are known to favour longer answers despite instructions, so part of the gap may reflect
  verbosity rather than correctness. §12.4 shows the judge *can* prefer the shorter answer,
  but the size of this effect against the larger models has not been measured.
- **The judge is not yet human-validated.** A 20-question blind A/B sheet
  (`judge_eval.py --make-blind`, with an agreement report via `--agreement`) is planned and has
  not been graded yet. Until then, the verdicts are a single automated grader's opinion.
- **Ties come only from position disagreements.** The judge never returned `tie` directly. Every
  tie in §12.3 is an item where the verdict reversed when the A/B order was swapped (25 against
  the base model, 7 / 2 / 4 against the larger models). Across all 736 verdicts the judge picked
  position A 53% of the time, a mild position bias that the swap neutralizes.
- **Truncation.** Some answers hit their generation cap: base 7B 4 / 92, 12B 7 / 92, 24B 9 / 92
  (reasoning plus answer budget). v8 and GPT-4o: 0. Truncated answers were judged as they were.
  One base-model answer (item 0) was produced during the pipeline smoke test, under the earlier
  512-token cap. All other local answers used the 1,024-token cap.
- **Final-answer prefix.** v8 ends 89 / 92 answers with `תשובה:` and never with the trained
  `תשובה סופית:` (same as v6 / v7, §10.4). All three larger models follow the preamble's
  `תשובה סופית:` instruction on 86–90 / 92 items. The judge was told to ignore this difference.
  No answer from any model contains CJK or Cyrillic characters, so v8 shows no language drift.
- **Single sample, small N.** One sampled answer per model per item (temperature 0.3, fixed
  seed). Four items per domain, so per-domain judge results are not reported. The group and
  overall rates are the stable numbers.
- **In-distribution benchmark.** As in §11.6, the gold set comes from the same syllabi and the
  same `gpt-4o-mini` generation pipeline as v8's training data. This favours v8 relative to the
  general-purpose opponents, which makes the large-model results a conservative estimate of
  the gap.
- **Use of `gpt-4o`.** The project rule is that all OpenAI calls use `gpt-4o-mini`. GPT-4o was
  admitted as an exception, approved by the project owner, as a *contestant* only: 92
  generation calls, throttled under the account's 30k TPM limit. The judge and all data
  generation remain `gpt-4o-mini`.

### 12.7 Run statistics

| Model | Mean time / answer | Generation speed | Total |
|---|--:|--:|--:|
| v8 (7B, transformers 4-bit) | 35.6 s | — | 0.9 h |
| Base 7B (transformers 4-bit) | 27.4 s | — | 0.7 h |
| 12B Instruct (llama.cpp, all layers on GPU) | 12.0 s | ~57 tokens/s | 0.3 h |
| 24B Thinking (llama.cpp, GPU + RAM offload) | 222.5 s | ~6.5 tokens/s | 5.7 h |
| GPT-4o (API, throttled) | 3.8 s | — | 0.1 h |

The 7B models were run through `transformers` + `bitsandbytes` for parity with the training and
loss-evaluation stack. Deployed v8 runs much faster in CoreCS (llama.cpp, ~93 tokens/s).

### 12.8 Figures for presentation

`plot_metrics.py` renders presentation-ready charts (300-dpi PNG plus SVG; light theme by default,
`--theme dark` for dark slides) into `figures/`:

| File | Content |
|---|---|
| `loss_progression` | Overall gold-set loss, Base → v6 → v7 → v8 (0.860 → 0.643 → 0.590 → 0.558) |
| `prior_vs_new_domains` | Mean loss on the 12 prior vs. 11 new domains for each version |
| `judge_results` | v8's win / tie / loss share against each opponent (§12.3) |

### 12.9 Output artifacts

| File | Contents |
|---|---|
| `eval_results_judge/answers/<model>.jsonl` | One answer per item per model, with timing, token count and finish reason |
| `eval_results_judge/answers/llama_server.log` | llama.cpp server log for the 12B / 24B runs |
| `eval_results_judge/judgments/<opponent>.jsonl` | Raw judge verdicts (both orders) and justifications |
| `eval_results_judge/judge_items.csv` | Per item and opponent: outcome, position flip, answer lengths |
| `eval_results_judge/judge_summary.csv` | The table in §12.3 (input to `plot_metrics.py`) |
| `figures/` | Presentation charts (§12.8) |

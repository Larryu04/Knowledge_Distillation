# Offline Knowledge Distillation: Hebrew Academic CS Tutor

A research project on offline knowledge distillation that trains a specialized, resource-efficient language model to act as an academic tutor for undergraduate Computer Science students (B.Sc. in Computer Science). The tutor answers Hebrew questions with a structured, numbered, step-by-step rationale followed by a concise final answer, grounded in the syllabi of the Academic College of Ramat Gan.

> **Latest milestone: v7.** The tutor now covers **12 Computer Science domains** (up from 6), trained on **14,400 rows** generated concurrently with `gpt-4o-mini`, and evaluated on a new **48-item gold benchmark**. The v7 QLoRA adapter beats the base model in all 12 domains and beats the v6 adapter on every one of the 48 benchmark items, with no catastrophic forgetting on the original six domains.

---

## Highlights (v7)

| | |
|---|---|
| **Domains** | 12 (expanded from 6) |
| **Training corpus** | `merged_12_courses.json`: 14,400 rows (1,200 per course x 12), generated concurrently via `gpt-4o-mini` |
| **Gold benchmark** | `hebrew_cs_pilot_dataset_48.json`: 48 items, 4 per domain x 12, derived from the course syllabi |
| **Adapter** | `dictalm2-tutor-qlora-v7`: QLoRA, r=16, alpha=32, 1,710 steps (2 epochs), best checkpoint at step 1704 |
| **v7 vs. base** | Masked loss 0.8292 -> 0.5319 (**+0.2974**), perplexity 2.29 -> 1.70; better in 12/12 domains and 48/48 items |
| **v7 vs. v6 (same 48 items)** | **+0.124** average gain on the 6 new domains, **+0.044** on the 6 original domains, 48/48 items improved |

### Covered Domains

| Original six | New in v7 |
|---|---|
| Computer Networks | Data Structures |
| Cybersecurity | Algorithms |
| Operating Systems | Computer Architecture |
| C++ (Fundamentals) | Databases |
| Java (OOP) | Automata Theory |
| Python (Advanced Programming) | Software Engineering |

---

## Technical Architecture & Methodology

### 1. Knowledge Distillation Setup
* **Teacher Model:** `gpt-4o-mini` was used offline to generate a synthetic, syllabus-grounded instruction dataset. Every row carries a numbered 3-5 point rationale that includes a worked example, plus a synthesized final answer.
* **Student Model:** `dicta-il/dictalm2.0-instruct` (Mistral-7B architecture with an extended Hebrew vocabulary and native tokenizer).
* **Supervision Format:** Hard-label sequence-to-sequence supervision. The model is penalized exclusively on target outputs (rationale + final answer) via strict loss masking.

### 2. Dataset Generation & Scale-Up (6 -> 12 Domains)
* **Concurrent generation:** Each of the 12 courses is generated as its own process (`generate_pilot_full-dataset.py`), 1,200 rows per course, using `gpt-4o-mini` exclusively.
* **Strict schema:** Every row has exactly five string fields: `course_name`, `domain`, `instruction`, `rationale`, `output`.
* **Validation gate:** Minimum instruction and answer length, 3-5 numbered rationale points with a worked example, out-of-syllabus term filtering, and near-duplicate instruction rejection. Generation checkpoints incrementally and resumes cleanly after interruptions.
* **Merging:** `merge_datasets.py` validates keys and domains, de-duplicates across courses (0 cross-course duplicates found), and guards against a short run, producing `merged_12_courses.json` with exactly 1,200 rows per domain.

### 3. Parameter-Efficient Fine-Tuning (QLoRA)
* **Quantization:** 4-bit NormalFloat (NF4) base model loading with double quantization and bfloat16 compute precision via `bitsandbytes`.
* **Adapter Rank & Alpha:** Low-rank adaptation (r=16, alpha=32, dropout 0.05) targeting attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP (`gate_proj`, `up_proj`, `down_proj`) blocks.
* **Vocabulary Preservation:** `embed_tokens` and `lm_head` remain **frozen** during training to preserve the base model's native Hebrew representations.
* **Optimization:** Learning rate 1e-4 with cosine schedule, 2 epochs, effective batch size 16 (micro-batch 2 x gradient accumulation 8), non-paged 8-bit AdamW, gradient clipping at `max_grad_norm = 0.3`.
* **Regularization & Stability:**
  * NEFTune noise injection (alpha=5) on input embeddings.
  * Right-padding with `<unk>` as a distinct `pad_token` (no EOS/PAD collisions) and an explicit EOS in every label.
  * Stratified per-domain validation split with early stopping on the macro per-domain loss.
  * Plain shuffling on the exactly balanced 12-domain corpus (domain-weighted sampling is used only when domain counts differ).
* **12 GB GPU safety (RTX 4070 Super):** VRAM cap at 85% of the card, worst-case-batch preflight before training, `MAX_SEQ_LEN=872` (no rows dropped), and crash-safe checkpoint resumption that skips incomplete or mismatched checkpoints.

### 4. Masked Cross-Entropy Loss
Loss calculation applies strictly to the tutor's response:
* Conversation framing uses the native chat template (`apply_chat_template`), with a shared preamble naming all 12 subject areas, kept identical across training, inference, and evaluation.
* All prompt tokens (system preamble + user question) are masked to `-100`, disabling gradient backpropagation over the query span.
* Explicit `eos_token_id` boundaries eliminate non-terminating decoding loops.

---

## Evaluation

### Gold Benchmark
The official benchmark is `hebrew_cs_pilot_dataset_48.json`: **48 items, 4 per domain across all 12 domains**, generated from the college syllabi so that it measures what the project is for, a tutor aligned to the actual course material. It replaces the earlier 24-item, 6-domain benchmark.

Evaluation is performed by `evaluate_models.py`: masked teacher-forcing loss and perplexity for the base model versus the adapter, reported per domain and overall, plus blind A/B answer sheets for human grading. A positive delta means the adapter assigns higher likelihood to the reference answer.

### v7 vs. Base Model

| Domain | Base Loss | v7 Loss | Delta |
|---|--:|--:|--:|
| Algorithms *(new)* | 0.9465 | 0.6203 | +0.3262 |
| Operating Systems | 0.8443 | 0.5225 | +0.3218 |
| Databases *(new)* | 0.8309 | 0.5092 | +0.3217 |
| Automata Theory *(new)* | 0.8200 | 0.5031 | +0.3169 |
| Java | 0.8648 | 0.5527 | +0.3122 |
| Computer Architecture *(new)* | 0.9361 | 0.6344 | +0.3016 |
| Software Engineering *(new)* | 0.7599 | 0.4639 | +0.2960 |
| Python | 0.8294 | 0.5392 | +0.2902 |
| Computer Networks | 0.8026 | 0.5131 | +0.2895 |
| Data Structures *(new)* | 0.7557 | 0.4797 | +0.2760 |
| C++ | 0.7535 | 0.4897 | +0.2638 |
| Cybersecurity | 0.8069 | 0.5547 | +0.2523 |
| **Overall (48 items)** | **0.8292** | **0.5319** | **+0.2974** |

The adapter improves on all 12 domains and on all 48 items.

### v7 vs. v6 Baseline (Same 48 Items)
To make a like-for-like comparison, the v6 adapter (trained on the original 6 domains only) was re-scored on the same 48-item benchmark. Both runs share an identical frozen base model, so per-item base losses match exactly.

| Group | v6 Loss | v7 Loss | Gain of v7 over v6 |
|---|--:|--:|--:|
| Original 6 domains | 0.5729 | 0.5286 | **+0.044** |
| New 6 domains | 0.6589 | 0.5351 | **+0.124** |
| **Overall (48 items)** | **0.6159** | **0.5319** | **+0.084** |

* **The expansion worked.** The average gain on the new domains (+0.124) is roughly three times that on the original domains. The largest gains are all in new domains: Automata Theory (+0.206), Algorithms (+0.139), Databases (+0.118), and Computer Architecture (+0.103).
* **Zero catastrophic forgetting.** Even though the original six domains dropped from 100% to 50% of the training corpus, v7 is better than v6 on all six of them and on all 24 of their items.

### Training Run
* 1,710 / 1,710 steps, no early stopping, no crashes; peak worst-case memory 6.37 GB allocated.
* Validation macro-domain loss fell from 0.8224 to 0.7695 (best checkpoint at step 1704), flattening after roughly step 850.

### Scope and Caveats
* **In-distribution by design:** the benchmark and training corpus come from the same syllabi and generation pipeline family, so the results demonstrate syllabus-specific learning rather than general out-of-syllabus ability.
* **Leakage check:** 0 of 48 exact instruction matches with the training corpus; results are unchanged (+0.294 to +0.296) after excluding the most similar items.
* **Extra training steps:** v7 trained for twice as many optimizer steps as v6, so part of its gain on the original domains may stem from additional training rather than the new data alone.
* **Prior transfer:** v6 already outperformed the base model on the unseen domains (+0.183), reflecting transferable style and format learning; the new data adds domain-specific gains on top.
* **Loss is not answer quality:** teacher-forcing loss rewards matching the reference wording. The blind A/B sheets in `eval_results_v7/` have not yet been human-graded.
* **Small sample:** four items per domain; the overall figure (N = 48) is the most stable number.

---

## Repository Structure

```text
├── train.py                          # QLoRA training pipeline with sequence masking
├── inference.py                      # Interactive REPL & single-turn query CLI
├── evaluate_models.py                # Base vs. adapter masked-loss evaluation + blind A/B sheets
├── compare_loss_results.py           # Side-by-side v6 vs. v7 loss comparison
├── generate_pilot_full-dataset.py    # Concurrent per-course corpus generation (gpt-4o-mini)
├── generate_pilot_dataset_for_48.py  # Generator for the 48-item gold benchmark
├── merge_datasets.py                 # Validation, de-duplication, and merging of per-course data
├── merged_12_courses.json            # Current training corpus (14,400 rows, 12 domains)
├── merged_6_courses_new.json         # Previous v6 corpus (7,200 rows, 6 domains)
├── hebrew_cs_pilot_dataset_48.json   # Official gold benchmark (48 items, 12 domains)
├── raw_courses_data/                 # Per-course generated data (12 files, 1,200 rows each)
├── eval_results_v5/ ... v7/          # Evaluation artifacts per adapter version
├── eval_results_v6_on48/             # v6 adapter scored on the 48-item benchmark
├── summary.md                        # Detailed evaluation reports (v5 through v7)
├── project_memory.md                 # Project context, constraints, and changelog
├── requirements.txt                  # Locked Python dependencies
└── .gitignore                        # Excludes checkpoints, local venv, secrets, and IDE configs
```

## Usage

```bash
# Train the v7 adapter (writes to out/dictalm2-tutor-qlora-v7)
python train.py

# Evaluate base vs. adapter on the 48-item benchmark
python evaluate_models.py

# Query the tutor interactively
python inference.py
```

## Next Steps
1. Human-grade the blind A/B sheets in `eval_results_v7/` to confirm that the loss gains translate into better answers.
2. Resolve the final-answer prefix mismatch: generations end with a final-answer line that uses a shorter prefix than the one seen in training. Either accept the shorter prefix or investigate why the trained one is not reproduced.

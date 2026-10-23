# Offline Knowledge Distillation: Hebrew Academic CS Tutor

A research project on offline knowledge distillation that trains a specialized, resource-efficient language model to act as an academic tutor for undergraduate Computer Science students (B.Sc. in Computer Science). The tutor answers Hebrew questions with a structured, numbered, step-by-step rationale followed by a concise final answer, grounded in the syllabi of the Academic College of Ramat Gan.

> Part of the [CoreCS monorepo](../README.md): this folder holds the research and training
> pipeline, [`../corecs/`](../corecs/) the offline desktop app that ships the model, and
> [`../GLOBAL/`](../GLOBAL/) the lecturer platform that generalizes the pipeline.

> **Final model: v8.** The tutor covers **23 Computer Science subjects**, trained on **27,600 rows** (1,200 per course) generated with `gpt-4o-mini`, and evaluated on a **92-item, 23-domain gold benchmark**. v8 beats the base model on all 92 items, is the best adapter on all 11 newly added subjects, and keeps ~97% of its gain on the 12 earlier ones. In a pairwise LLM-as-a-judge benchmark on generated answers, v8 **wins or ties 84.8%** of comparisons against its own base model. The model runs fully offline in **CoreCS**, a native desktop app (Tauri UI + Rust core + llama.cpp) at **~93 tokens/s** on an RTX 4070 SUPER, packaged as a standalone Windows installer.

---

## Highlights (v8)

| | |
|---|---|
| **Subjects** | 23 (6 in v6 -> 12 in v7 -> 23 in v8) |
| **Training corpus** | `merged_all_courses.json`: 27,600 rows (1,200 per course x 23), generated via `gpt-4o-mini` |
| **Gold benchmark** | `hebrew_cs_pilot_dataset_92.json`: 92 items, 4 per domain x 23, derived from the course syllabi |
| **Adapter** | `dictalm2-tutor-qlora-v8`: QLoRA, r=16, alpha=32, 3,278 steps (2 epochs), best checkpoint at step 2454 |
| **v8 vs. base** | Masked loss 0.8598 -> 0.5584 (**+0.301**), perplexity 2.36 -> 1.75; better on 92/92 items |
| **v8 vs. v7 (same 92 items, same prompt)** | **+0.078** on the 11 new subjects (42/44 items), **-0.011** on the 12 prior subjects, **+0.031** overall |
| **v8 vs. base (LLM judge)** | 53 wins / 25 ties / 14 losses on 92 generated answers: **84.8% win-or-tie** (`gpt-4o-mini` judge, position-swapped) |
| **Deployment** | CoreCS offline desktop app, Q4_K_M GGUF via llama.cpp (Vulkan), ~93 tokens/s, 18.5 MB NSIS installer |

### Covered Subjects

| v6 (original six) | Added in v7 | Added in v8 |
|---|---|---|
| Computer Networks | Data Structures | Discrete Mathematics |
| Cybersecurity | Algorithms | Logic and Set Theory |
| Operating Systems | Computer Architecture | Graph Theory |
| C++ (Fundamentals) | Databases | Computability and Complexity |
| Java (OOP) | Automata Theory | Introduction to Cryptography |
| Python (Advanced Programming) | Software Engineering | Introduction to AI |
| | | Introduction to Machine Learning |
| | | Deep Learning |
| | | AI-Based Software Development |
| | | Information Systems Foundations |
| | | Computational Biology Applications |

---

## Technical Architecture & Methodology

### 1. Knowledge Distillation Setup
* **Teacher Model:** `gpt-4o-mini` was used offline to generate a synthetic, syllabus-grounded instruction dataset. Every row carries a numbered 3-5 point rationale that includes a worked example, plus a synthesized final answer.
* **Student Model:** `dicta-il/dictalm2.0-instruct` (Mistral-7B architecture with an extended Hebrew vocabulary and native tokenizer).
* **Supervision Format:** Hard-label sequence-to-sequence supervision. The model is penalized exclusively on target outputs (rationale + final answer) via strict loss masking.

### 2. Dataset Generation & Scale-Up (6 -> 12 -> 23 Subjects)
* **Concurrent generation:** Each course is generated as its own process (`generate_pilot_full-dataset.py`, batch-launched by `run_batches.ps1`), 1,200 rows per course, using `gpt-4o-mini` exclusively.
* **Strict schema:** Every row has exactly five string fields: `course_name`, `domain`, `instruction`, `rationale`, `output`.
* **Validation gate:** Minimum instruction and answer length, 3-5 numbered rationale points with a worked example, out-of-syllabus term filtering, and near-duplicate instruction rejection. Generation checkpoints incrementally and resumes cleanly after interruptions.
* **Merging:** `merge_datasets.py` validates keys and domains for all 23 per-course files in `raw_courses_data/`, checks for cross-course duplicates (0 found), and guards against a short run, producing `merged_all_courses.json` with exactly 1,200 rows per domain.

### 3. Parameter-Efficient Fine-Tuning (QLoRA)
* **Quantization:** 4-bit NormalFloat (NF4) base model loading with double quantization and bfloat16 compute precision via `bitsandbytes`.
* **Adapter Rank & Alpha:** Low-rank adaptation (r=16, alpha=32, dropout 0.05) targeting attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP (`gate_proj`, `up_proj`, `down_proj`) blocks.
* **Vocabulary Preservation:** `embed_tokens` and `lm_head` remain **frozen** during training to preserve the base model's native Hebrew representations.
* **Optimization:** Learning rate 1e-4 with cosine schedule, 2 epochs, effective batch size 16 (micro-batch 2 x gradient accumulation 8), non-paged 8-bit AdamW, gradient clipping at `max_grad_norm = 0.3`.
* **Regularization & Stability:**
  * NEFTune noise injection (alpha=5) on input embeddings.
  * Right-padding with `<unk>` as a distinct `pad_token` (no EOS/PAD collisions) and an explicit EOS in every label.
  * Stratified per-domain validation split with early stopping on the macro per-domain loss.
  * Plain shuffling on the exactly balanced 23-domain corpus (domain-weighted sampling is used only when domain counts differ).
* **12 GB GPU safety (RTX 4070 Super):** VRAM cap at 85% of the card, worst-case-batch preflight before training, `MAX_SEQ_LEN=960` (1 outlier row dropped out of 27,600), and crash-safe checkpoint resumption that skips incomplete or mismatched checkpoints.

### 4. Masked Cross-Entropy Loss
Loss calculation applies strictly to the tutor's response:
* Conversation framing uses the native chat template (`apply_chat_template`), with a shared preamble naming all 23 subject areas, kept byte-identical across training, inference, and evaluation.
* All prompt tokens (preamble + user question) are masked to `-100`, disabling gradient backpropagation over the query span.
* Explicit `eos_token_id` boundaries eliminate non-terminating decoding loops.

---

## Evaluation

### Gold Benchmark
The official benchmark is `hebrew_cs_pilot_dataset_92.json`: **92 items, 4 per domain across all 23 domains**, generated from the college syllabi so that it measures what the project is for, a tutor aligned to the actual course material. It replaces the earlier 48-item, 12-domain benchmark.

Evaluation is performed by `evaluate_models.py`: masked teacher-forcing loss and perplexity for the base model versus an adapter, reported per domain and overall, plus optional blind A/B answer sheets for human grading. `compare_loss_results.py` merges the per-adapter results into one side-by-side table.

### Fair Three-Way Comparison: v6 vs. v7 vs. v8
All three adapters were scored on the same 92 items with the **same 23-subject preamble**. They share an identical frozen base model, so per-item base losses match exactly. Lower loss is better.

| Group | Base | v6 | v7 | v8 | Gain of v8 over v7 |
|---|--:|--:|--:|--:|--:|
| Prior 12 subjects (48 items) | 0.8627 | 0.6281 | 0.5529 | 0.5636 | **-0.011** |
| New 11 subjects (44 items) | 0.8567 | 0.6599 | 0.6303 | 0.5528 | **+0.078** |
| **Overall (92 items)** | **0.8598** | **0.6433** | **0.5899** | **0.5584** | **+0.031** |

* **The 23-subject expansion worked.** v8 is the best adapter on all 11 new subjects and beats v7 on 42 of 44 of their items. The largest gains are Logic and Set Theory (+0.132), Computability and Complexity (+0.127), Introduction to Machine Learning (+0.100), and Introduction to Cryptography (+0.097).
* **No catastrophic forgetting.** On the 12 prior subjects v8 is slightly behind v7 (about 2% higher loss; Cybersecurity is the one prior subject where v8 improved), a mild dilution effect from those subjects dropping to 52% of the corpus. v8 still retains ~97% of v7's gain over the base model there, beats the base model on all 48 of their items, and remains well ahead of v6.
* **Best overall.** v8 has the lowest overall loss and beats the base model on all 92 items.

Full per-domain table: `eval_results_v8/v6_vs_v7_vs_v8_comparison.csv` and `summary.md` §11.

### Training Run (v8)
* 3,278 / 3,278 steps, no early stopping.
* Validation macro-domain loss fell from 0.8162 to 0.7834 (best checkpoint at step 2454) and was flat from roughly step 2000 onward.

### Evaluation Methodology: LLM-as-a-Judge
Teacher-forcing loss has two limits as a final metric. It cannot compare models with different tokenizers, because loss and perplexity are measured per token. It also rewards reproducing the reference wording rather than generating a correct answer. The final evaluation therefore compares **generated answers** directly:

* **Reference-guided grading.** For each of the 92 gold items, a judge model (`gpt-4o-mini`, temperature 0, JSON output) receives the question, the gold reference answer and two anonymous candidate answers. It ranks them by correctness against the reference, then completeness, then clarity in Hebrew. It is instructed to ignore length, formatting and the exact wording of the final-answer line.
* **Double-blind A/B with position swapping.** Neither candidate is identified to the judge, and every pair is judged twice with the A/B order reversed. v8 scores +1 / 0 / −1 per order, and the sign of the sum decides the item. A verdict that reverses with the order counts as a tie, so position bias cannot affect the result.
* **Identical conditions.** All contestants receive the same 23-subject preamble and question, with temperature 0.3, top-p 0.9 and a fixed per-item seed.
* **Contestants.** v8 is compared with its own base model (DictaLM-2.0-instruct 7B, the sanity check) and with three larger or newer models: DictaLM-3.0-Nemotron-12B-Instruct, DictaLM-3.0-24B-Thinking (reasoning discarded, final answer judged) and GPT-4o.

### Final Benchmark Results (LLM-as-a-Judge)

| v8 vs. | Params | Wins | Ties | Losses | Win-or-tie |
|---|---|--:|--:|--:|--:|
| **DictaLM-2.0-instruct (no fine-tuning)** | 7B | **53** | 25 | 14 | **84.8%** |
| DictaLM-3.0 Nemotron-12B-Instruct | 12B | 4 | 7 | 81 | 12.0% |
| DictaLM-3.0 24B-Thinking | 24B | 0 | 2 | 90 | 2.2% |
| GPT-4o | proprietary | 0 | 4 | 88 | 4.3% |

* **The fine-tuning works.** Same weights, tokenizer, prompt and sampling; the adapter is the only difference. Against that baseline, v8 is preferred more than four times as often as it loses. This holds even though v8's answers are shorter than the base model's on 79 of 92 items, so the preference is not explained by length.
* **Larger, newer models remain ahead.** v8 loses clearly to models 1.7× to 3.4× its size and to GPT-4o, which is the expected ceiling for a 7B model. Because v8 was distilled from `gpt-4o-mini` data, a judge biased toward that style would have favoured v8; it did not, which counts against a self-preference bias. These opponents differ from v8 in scale, base model, training data and inference method, so the comparison shows a gap consistent with model scale and generation but does not isolate scale itself.
* **Open validation.** Agreement between the judge and a human grader has not yet been measured. Note also that v8 writes the shortest answers of all contestants, so length bias may contribute to the gap with the larger models.

Full protocol, per-group breakdown and caveats: `summary.md` §12.

### Scope and Caveats
* **In-distribution by design:** the benchmark and training corpus come from the same syllabi and generation pipeline family, so the results demonstrate syllabus-specific learning rather than general out-of-syllabus ability.
* **Leakage check:** 0 of 92 exact instruction matches with the training corpus; maximum within-domain token-Jaccard similarity 0.579.
* **Shared prompt:** v6 and v7 were scored with the 23-subject preamble, which they were not trained on, so that all three adapters see the identical prompt.
* **Loss is not answer quality:** teacher-forcing loss rewards matching the reference wording. The LLM-as-a-judge benchmark above complements it with an answer-level evaluation. That judge has not yet been validated against human grading.
* **Small sample:** four items per domain; the group means and the overall figure (N = 92) are the most stable numbers.

---

## Evaluation Pipeline & Tooling

The judge benchmark is a three-stage pipeline. Each stage writes to disk, and the expensive first two stages can be interrupted and resumed.

### `generate_answers.py` — answer generation
Produces one answer per gold item for each contestant, using the identical prompt (the preamble is imported from `inference.py`).
* **Local 7B models (v8, base):** loaded through `transformers` in 4-bit NF4, with the same generation settings as `inference.py` (1,024-token cap, repetition guards).
* **Local GGUF models via `llama-server`:** the script starts and stops CoreCS's bundled llama.cpp server, pins it to the discrete GPU (Vulkan device 0, not the integrated GPU) and talks to its OpenAI-compatible endpoint. The 12B Instruct model (Q4_K_M, 7.5 GB) fits fully in the 12 GB of VRAM. The 24B Thinking model (Q4_K_M, 14.3 GB) does not, so only part of its layers run on the GPU and the rest are offloaded to system RAM (~6.5 tokens/s, about 6 hours for 92 items).
* **Reasoning models:** thinking is capped with `--reasoning-budget 2048`. The server separates the reasoning trace from the answer, and any residual `<think>…</think>` block is stripped, so only the final answer reaches the judge.
* **API throttling (GPT-4o):** a sliding 60-second token window keeps usage under a 22k tokens/minute budget, below the account's 30k TPM limit. Before each call the script reserves the prompt plus the full completion budget. It also waits at least 2 s between calls and retries rate-limit and server errors with exponential backoff.
* **Continuous saving:** every answer is appended to `eval_results_judge/answers/<model>.jsonl` and flushed to disk immediately, with timing, token count and finish reason. A rerun skips completed items, so a crash, reboot or Ctrl-C loses at most one answer.

### `judge_eval.py` — LLM-as-a-judge
* Runs the position-swapped, reference-guided comparison of v8 against every opponent whose answers exist, using `gpt-4o-mini` with 4 parallel workers and retry/backoff.
* Caches every verdict in `eval_results_judge/judgments/<opponent>.jsonl`, so reruns only request missing verdicts.
* Writes `judge_summary.csv` (wins / ties / losses per opponent, position-flip counts) and `judge_items.csv` (per-item outcome and answer lengths).
* `--make-blind` exports a stratified 20-question blind A/B sheet for human grading, with a separate key file. `--agreement` then reports how often the human grader and the judge agree.

### `plot_metrics.py` — presentation figures
Renders publication-quality charts (300-dpi PNG plus SVG) into `figures/`, with `--theme light` (default) or `--theme dark`:
* `loss_progression`: overall gold-set loss across Base → v6 → v7 → v8.
* `prior_vs_new_domains`: grouped bars for the 12 prior vs. 11 new domains per version.
* `judge_results`: a 100% stacked bar of v8's win / tie / loss share against each opponent, read from `judge_summary.csv`.

---

## Deployment: CoreCS Desktop App

The trained tutor ships as **CoreCS**, a fully offline Windows desktop application in [`corecs/`](../corecs/) at the monorepo root:

* **Architecture:** Tauri 2 shell with a React/TypeScript UI (Hebrew RTL), a Rust core (SQLite chat history, model discovery, engine lifecycle), and a bundled llama.cpp `llama-server` sidecar running on Vulkan.
* **Model format:** the QLoRA adapter is merged into the base model and quantized to a Q4_K_M GGUF by `corecs/scripts/export_gguf.py`, together with an optional profile JSON carrying the training preamble and sampling settings. The app is model-agnostic: any new GGUF appears in its model picker without code changes.
* **Performance:** ~93 tokens/s generation on an RTX 4070 SUPER (measured end-to-end in the app; v8 uses the same architecture and quantization).
* **Shipped model:** v8 is exported as `corecs-v8-q4_k_m.gguf` and loaded in the app.
* **Installer:** `npm run tauri build -- --bundles nsis` produces a standalone 18.5 MB `setup.exe` (app + inference engine); the ~4.4 GB model file is distributed separately.

Export a trained adapter for the app (from this `model_training/` folder; this is how v8 was exported):

```powershell
python ..\corecs\scripts\export_gguf.py --adapter out/dictalm2-tutor-qlora-v8/adapter --name corecs-v8
```

---

## Repository Structure

```text
model_training/
├── train.py                          # QLoRA training pipeline with sequence masking
├── inference.py                      # Interactive REPL & single-turn query CLI
├── evaluate_models.py                # Base vs. adapter masked-loss evaluation + blind A/B sheets
├── compare_loss_results.py           # Side-by-side comparison of N adapters (v6 vs. v7 vs. v8)
├── generate_answers.py               # Benchmark answers from all contestants (local HF, llama-server GGUF, API)
├── judge_eval.py                     # Position-swapped, reference-guided LLM-as-a-judge (gpt-4o-mini)
├── plot_metrics.py                   # Presentation figures (loss progression, domain groups, judge results)
├── generate_pilot_full-dataset.py    # Concurrent per-course corpus generation (gpt-4o-mini)
├── generate_pilot_dataset_for_92.py  # Generator for the 92-item gold benchmark
├── run_batches.ps1                   # Batch launcher for per-course generation runs
├── merge_datasets.py                 # Validation, de-duplication, and merging of per-course data
├── merged_all_courses.json           # Final training corpus (27,600 rows, 23 domains)
├── merged_12_courses.json            # Previous v7 corpus (14,400 rows, 12 domains)
├── merged_6_courses_new.json         # Previous v6 corpus (7,200 rows, 6 domains)
├── hebrew_cs_pilot_dataset_92.json   # Official gold benchmark (92 items, 23 domains)
├── hebrew_cs_pilot_dataset_48.json   # Previous benchmark (48 items, 12 domains)
├── raw_courses_data/                 # Per-course generated data (23 files, 1,200 rows each)
├── eval_results_v8/                  # v8 results + the three-way comparison CSV
├── eval_results_v6_fair/, v7_fair/   # v6 / v7 re-scored on the 92 items with the 23-subject preamble
├── eval_results_v5/ ... v7/          # Historical evaluation artifacts per adapter version
├── eval_results_judge/               # Judge benchmark: answers, verdicts, summary CSVs
├── figures/                          # Generated presentation charts (PNG + SVG)
├── summary.md                        # Detailed evaluation reports (v5 through v8, judge benchmark)
├── project_memory.md                 # Project context, constraints, and changelog
├── requirements.txt                  # Locked Python dependencies
├── out/                              # (git-ignored) adapters and checkpoints, e.g. out/dictalm2-tutor-qlora-v8/
├── models/                           # (git-ignored) downloaded GGUFs for the judge benchmark
└── logs/, archive/                   # (git-ignored) run logs and retired artifacts
```

Ignore rules for all three sub-projects live in the monorepo's root [`.gitignore`](../.gitignore).

## Setup

```powershell
cd model_training
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
$env:OPENAI_API_KEY = "sk-..."    # only for dataset generation and the judge benchmark
```

## Usage

All commands run from `model_training/` with the venv active.

```bash
# Train the v8 adapter (writes to out/dictalm2-tutor-qlora-v8)
python train.py

# Evaluate base vs. v8 on the 92-item benchmark (-> eval_results_v8/)
python evaluate_models.py
# Re-score an older adapter with the current preamble (-> eval_results_v7_fair/)
python evaluate_models.py --adapter-dir out/dictalm2-tutor-qlora-v7/adapter

# Merge the per-adapter results into one comparison table
python compare_loss_results.py

# Query the tutor interactively
python inference.py
```

LLM-as-a-judge benchmark (requires `OPENAI_API_KEY`; GGUF files are downloaded to `models/`, which is git-ignored):

```powershell
# 1. Generate answers (each run is resumable)
python generate_answers.py --model v8
python generate_answers.py --model base7b
python generate_answers.py --model dicta12b --gguf models\DictaLM-3.0-Nemotron-12B-Instruct-Q4_K_M.gguf
python generate_answers.py --model dicta24b --gguf models\DictaLM-3.0-24B-Thinking-Q4_K_M.gguf
python generate_answers.py --model gpt4o

# 2. Judge v8 against every opponent (-> eval_results_judge/judge_summary.csv)
python judge_eval.py
python judge_eval.py --make-blind     # 20-question sheet for human validation
python judge_eval.py --agreement      # after grading the sheet

# 3. Render the presentation figures (-> figures/)
python plot_metrics.py
```

## Next Steps
1. Validate the judge: grade the 20-question blind sheet and report human–judge agreement.
2. Verify a run of the installed CoreCS app (from `setup.exe`) with the v8 model.
3. Resolve the final-answer prefix mismatch: v6, v7 and v8 end answers with `תשובה:` rather than the trained `תשובה סופית:` (v8: 89/92 answers, 0 with the trained prefix).

# Offline Knowledge Distillation: Hebrew Academic CS Tutor

A research project focused on offline knowledge distillation to train a specialized, resource-efficient language model serving as an academic tutor for undergraduate Computer Science students (B.Sc. in Computer Science).

The tutor provides structured, step-by-step logical explanations followed by concise final answers for core CS domains: **Operating Systems**, **Computer Networking**, and **Cybersecurity**.

---

## Technical Architecture & Methodology

### 1. Knowledge Distillation Setup
* **Teacher Model:** GPT-4o was utilized offline to generate synthetically curated, high-accuracy instruction datasets (`merged_training_dataset_V4_clean.json`) incorporating core curriculum concepts, architectural edge cases, and standard academic rationales.
* **Student Model:** `dicta-il/dictalm2.0-instruct` (Mistral-7B architecture base with an extended Hebrew vocabulary and native tokenizer).
* **Supervision Format:** Hard-label sequence-to-sequence supervision. The model is penalized exclusively on target outputs (rationale + final answer) via strict loss masking.

### 2. Parameter-Efficient Fine-Tuning (QLoRA)
* **Quantization:** 4-bit NormalFloat (NF4) base model loading with double quantization and bfloat16 compute precision via `bitsandbytes`.
* **Adapter Rank & Alpha:** Low-rank adaptation ($r=8$, $\alpha=16$) targeting key linear projections across Attention (`q_proj`, `k_proj`, `v_proj`, `o_proj`) and MLP (`gate_proj`, `up_proj`, `down_proj`) blocks.
* **Vocabulary Preservation:** `embed_tokens` and `lm_head` remain strictly **frozen** (`requires_grad = False`) during training to preserve the base model's native Hebrew semantic representations without catastrophic forgetting.
* **Regularization & Stability:** 
  * NEFTune noise injection ($\alpha=5$) on input embeddings for better generalization on compact domain data.
  * Gradient clipping at `max_grad_norm = 0.3`.
  * Right-padding for training using `<unk>` as a distinct `pad_token` (preventing EOS/PAD collisions).

### 3. Masked Cross-Entropy Loss
Loss calculation applies strictly to the tutor's response:
* Dynamic conversation framing formats the student turn using the native chat template (`apply_chat_template`).
* All prompt tokens (system preamble + user question) are masked to `-100`, disabling gradient backpropagation over the query span.
* Generation is reinforced with explicit `eos_token_id` boundaries to eliminate non-terminating decoding loops.

---

## Repository Structure

```text
├── train.py                              # QLoRA training pipeline with sequence masking
├── inference.py                          # Interactive REPL & single-turn query CLI
├── merged_training_dataset_V4_clean.json # Distilled training dataset
├── requirements.txt                      # Locked Python dependencies
└── .gitignore                            # Excludes checkpoints, local venv, and IDE configs

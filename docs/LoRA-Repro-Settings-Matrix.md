# LoRA Reproduction Settings Matrix (RoBERTa-base + GPT-2 Small/Medium)

## 1. RoBERTa-base – GLUE Classification Tasks

This table includes the LoRA paper’s settings for RoBERTa-base, plus our intended reduced-resource reproduction.

| Category            | LoRA Paper Value                                                              | Feasible for Our Project                     | Unknown/Unspecified                                  | Notes                                           |
|---------------------|-------------------------------------------------------------------------------|----------------------------------------------|------------------------------------------------------|-------------------------------------------------|
| Model               | RoBERTa-base (125M)                                                           | Same                                         | —                                                    | —                                               |
| Datasets            | MNLI, SST-2, MRPC, CoLA, QNLI, QQP, RTE, STS-B                                | MNLI (main), optionally MRPC/RTE             | —                                                    | MNLI is the standard benchmark to start with.   |
| Max seq length      | **512** (main RoBERTa-base runs) <br> **128** (restricted adapter comparison) | 128–512 (your choice; 128 saves VRAM)        | —                                                    | Use 128 for consistency with adapter baselines. |
| Batch size          | **16**                                                                        | 16                                           | —                                                    | Good for 1 GPU V100.                            |
| Epochs              | **30** (MNLI)                                                                 | **3–10**                                     | —                                                    | We cannot afford 30 epochs; 3–5 is standard.    |
| Optimizer           | AdamW                                                                         | Same                                         | —                                                    | —                                               |
| Learning rate       | **5e-4**                                                                      | 5e-4                                         | —                                                    | —                                               |
| Warmup              | 6%                                                                            | Optional                                     | —                                                    | Not essential for reproduction.                 |
| LoRA ranks          | r = **8** (paper)                                                             | r = **1, 4, 8, 16** (your planned ablations) | —                                                    | We *should* run r = 1, 4, 8, 16.                |
| LoRA α              | α = r or fixed (8)                                                            | α = r OR α=8                                 | Paper hand-waves this; α largely learning-rate-like. | —                                               |
| LoRA target modules | **Wq and Wv only**                                                            | Same                                         | —                                                    | Don’t apply LoRA to FFN layers.                 |
| Trainable params    | ~300k                                                                         | ~75k–600k                                    | —                                                    | Depends on r.                                   |
| Metrics             | Accuracy                                                                      | Same                                         | —                                                    | MNLI splits: matched vs mismatched.             |
| Seeds               | 5                                                                             | 1–3                                          | —                                                    | Lower seeds = acceptable for project.           |

**Summary for RoBERTa-base**

```txt
batch_size = 16
max_seq_length = 128
learning_rate = 5e-4
epochs = 3
lora_r ∈ {1, 4, 8, 16}
lora_alpha = lora_r
modules = {q_proj, v_proj}
optimizer = AdamW
seeds = 3
```

## 2. GPT-2 Small / Medium – E2E NLG

This table includes the LoRA paper’s settings for GPT-2 Medium, plus our intended reduced-resource reproduction.

| Category            | LoRA Paper Value (GPT-2 Medium)                  | Feasible for Our Project           | Unknown/Unspecified       | Notes                                |
|---------------------|--------------------------------------------------|------------------------------------|---------------------------|--------------------------------------|
| Model               | GPT-2 Medium (345M)                              | GPT-2 small (124M) + medium (345M) | —                         | We can run small & medium easily.    |
| Dataset             | E2E NLG                                          | Same                               | —                         | Perfect dataset for text generation. |
| Max seq length      | Not stated explicitly; E2E is short (~80 tokens) | 128–256                            | Exact training seq length | 128 is fine.                         |
| Batch size          | **8**                                            | 4–8                                | —                         | Use 4 on small GPU memory.           |
| Epochs              | **5**                                            | 3–5                                | —                         | Matches paper.                       |
| Learning rate       | **2e-4**                                         | Same                               | —                         | Recommended by paper.                |
| Optimizer           | AdamW                                            | Same                               | —                         | —                                    |
| LoRA rank           | **r = 4** (paper)                                | r ∈ {1, 4, 8, 16}                  | —                         | Needed for ablations.                |
| LoRA α              | **32** (paper)                                   | 16–32                              | No theoretical guidance   | Use α=32 if r=4; otherwise α=r.      |
| LoRA target modules | q_proj + v_proj                                  | Same                               | —                         | As in paper.                         |
| Metrics             | BLEU, NIST, METEOR, ROUGE-L, CIDEr               | Same                               | —                         | Provided by the dataset.             |
| Seeds               | 3                                                | 1–3                                | —                         | Paper averages 3 runs.               |

**Summary for GPT-2 Experiments**

```txt
epochs = 5
batch_size = 8
learning_rate = 2e-4
lora_r ∈ {1, 4, 8, 16}
lora_alpha = 32 (for r=4) or alpha = r
target_modules = {q_proj, v_proj}
seq_length = 128
optimizer = AdamW
seeds = 3
```

## 3. Required Ablation Studies

| Ablation                   | Required by Paper?                 | Required by Our Proposal? | What to Vary                                                        |
|----------------------------|------------------------------------|---------------------------|---------------------------------------------------------------------|
| LoRA rank r                | **Yes**                            | Yes                       | r = 1, 4, 8, 16                                                     |
| Which matrices adapted     | **Yes** (Wq vs Wv vs combinations) | Optional                  | {Wq}, {Wv}, {Wq+Wv}                                                 |
| Hyperparameter sensitivity | Yes                                | Yes                       | LR, α                                                               |
| Comparison to baselines    | Yes                                | Yes                       | Full fine-tune, partial fine-tune, "bias-only", optionally adapters |

## 4. Things the Paper Does NOT Specify Clearly

| Category                                  | Not Specified | How to Handle                                                  |
|-------------------------------------------|---------------|----------------------------------------------------------------|
| Exact random seeds for each model         | ❌             | Use 1–3 seeds.                                                 |
| Exact training sequence lengths for GPT-2 | ❌             | Use 128.                                                       |
| Exact warmup ratio for GPT-2              | ❌             | Use 500 warmup steps (paper uses 500).                         |
| Exact evaluation cadence (per steps)      | ❌             | Use default HF Trainer (`eval_steps=None`, eval at epoch end). |
| Hardware details for RoBERTa-base         | ❌             | Not needed; use V100s on Wahab cluster.                        |


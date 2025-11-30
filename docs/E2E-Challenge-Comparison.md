| Study         | Model       | Method        | Avg Trainable Params (M) | Avg BLEU  | Avg ROUGE-L | Avg METEOR |
|---------------|-------------|---------------|--------------------------|------|-------------|------------|
| LoRA        | GPT-2 L     | Adapter L       | 0.88                     | 69.1 | 71.4   | 46.3  |
| LoRA        | GPT-2 L     | FT              | 774.03                   | 68.5 | 69.9   | 46.0  |
| LoRA        | GPT-2 L     | LoRA            | 0.77                     | 70.4 | 72.0   | 46.8  |
| LoRA        | GPT-2 L     | PreLayer        | 0.77                     | 70.3 | 71.7   | 46.2  |
| LoRA        | GPT-2 M     | Adapter L       | 0.37                     | 66.3 | 69.8   | 45.0  |
| LoRA        | GPT-2 M     | FT              | 354.92                   | 68.2 | 71.0   | 46.2  |
| LoRA        | GPT-2 M     | LoRA            | 0.35                     | 70.4 | 71.8   | 46.8  |
| LoRA        | GPT-2 M     | PreLayer        | 0.35                     | 69.7 | 71.4   | 46.0  |
| Reproduction| GPT-2 Medium| BitFit          | 0.3M                     | 12.8 | 34.7   | 56.3  |
| Reproduction| GPT-2 Medium| Full FT         | 354.8M                   | 12.9 | 31.4   | 54.4  |
| Reproduction| GPT-2 Medium| LoRA            | 2.0M                     | 13.1 | 39.2   | 55.8  |
| Reproduction| GPT-2 Small | BitFit          | 0.1M                     | 12.6 | 33.5   | 54.7  |
| Reproduction| GPT-2 Small | Full FT         | 124.4M                   | 13.5 | 32.6   | 55.5  |
| Reproduction| GPT-2 Small | LoRA            | 0.7M                     | 13.2 | 33.4   | 55.6  |

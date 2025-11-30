| study        | model        | method   | trainable parameters   |   mnli |   mrpc |
|:-------------|:-------------|:---------|:-----------------------|-------:|-------:|
| LoRA         | RoBERTa BASE | FT       | 125.0M                 |   87.6 |   90.2 |
| Reproduction | RoBERTa BASE | FT       | 125.0M                 |   87.8 |   91.2 |
| LoRA         | RoBERTa BASE | BitFit   | 0.1M                   |   84.7 |   92.7 |
| Reproduction | RoBERTa BASE | BitFit   | 0.1M                   |   84.4 |   87   |
| LoRA         | RoBERTa BASE | Adpt D   | 0.3M                   |   87.1 |   88.5 |
| LoRA         | RoBERTa BASE | Adpt D   | 0.9M                   |   87.3 |   88.4 |
| LoRA         | RoBERTa BASE | LoRA     | 0.3M                   |   87.5 |   89.7 |
| Reproduction | RoBERTa BASE | LoRA     | 0.8M                   |   86.8 |   88.2 |

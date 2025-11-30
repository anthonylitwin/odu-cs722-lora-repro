| Model       | Method       | Avg Trainable Params (M) | Avg BLEU  | Avg ROUGE-L | Avg METEOR | Sources       |
|------------|--------------|--------------------------|-----------|-------------|------------|---------------|
| GPT-2 L    | Adapter_L    | 0.88                     | 69.100000 | 71.400000   | 46.300000  | [LoRA]        |
| GPT-2 L    | Adapter_L^r  | 23.00                    | 68.900000 | 71.300000   | 46.000000  | [LoRA]        |
| GPT-2 L    | FT           | 774.03                   | 68.500000 | 69.900000   | 46.000000  | [LoRA]        |
| GPT-2 L    | LoRA         | 0.77                     | 70.400000 | 72.000000   | 46.800000  | [LoRA]        |
| GPT-2 L    | PreLayer     | 0.77                     | 70.300000 | 71.700000   | 46.200000  | [LoRA]        |
| GPT-2 M    | Adapter_H    | 11.09                    | 67.300000 | 70.800000   | 46.200000  | [LoRA]        |
| GPT-2 M    | Adapter_L    | 0.37                     | 66.300000 | 69.800000   | 45.000000  | [LoRA]        |
| GPT-2 M    | Adapter_L^r  | 11.09                    | 68.900000 | 71.300000   | 46.000000  | [LoRA]        |
| GPT-2 M    | FT           | 354.92                   | 68.200000 | 71.000000   | 46.200000  | [LoRA]        |
| GPT-2 M    | FT^Top2      | 25.19                    | 68.100000 | 70.800000   | 46.000000  | [LoRA]        |
| GPT-2 M    | LoRA         | 0.35                     | 70.400000 | 71.800000   | 46.800000  | [LoRA]        |
| GPT-2 M    | PreLayer     | 0.35                     | 69.700000 | 71.400000   | 46.000000  | [LoRA]        |
| GPT-2 Medium | BitFit     | NaN                      | 12.780213 | 34.697201   | 56.345169  | [Reproduction] |
| GPT-2 Medium | Full FT    | NaN                      | 12.904269 | 31.382245   | 54.371138  | [Reproduction] |
| GPT-2 Medium | LoRA       | NaN                      | 13.148866 | 39.191030   | 55.795895  | [Reproduction] |
| GPT-2 Small  | BitFit     | NaN                      | 12.640254 | 33.458374   | 54.661926  | [Reproduction] |
| GPT-2 Small  | Full FT    | NaN                      | 13.532101 | 32.570455   | 55.493638  | [Reproduction] |
| GPT-2 Small  | LoRA       | NaN                      | 13.228193 | 33.445101   | 55.576680  | [Reproduction] |

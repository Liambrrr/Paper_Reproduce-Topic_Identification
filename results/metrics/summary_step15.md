# Step 15: Model-wise Summary (per subset)

_Per-metric distributions are computed across topics within each subset._

## Group A

| Model | Cosine: label↔centroid | Mean cosine: label↔docs | Cosine: label↔mean(top-words) | Jaccard: label↔top-words |
|---|---|---|---|---|
| `meta-llama_llama-3.1-8b-instruct` | N/A | N/A | 0.564 ± 0.116 | 0.587 [0.087] | 0.287–0.745 (n=20) | 0.107 ± 0.085 | 0.083 [0.143] | 0.000–0.300 (n=20) |
| `meta-llama_llama-3.3-70b-instruct` | N/A | N/A | 0.586 ± 0.136 | 0.612 [0.165] | 0.398–0.724 (n=4) | 0.044 ± 0.050 | 0.042 [0.087] | 0.000–0.091 (n=4) |
| `qwen_qwen2.5-coder-7b-instruct` | N/A | N/A | N/A | N/A |
| `qwen_qwen3-vl-30b-a3b-instruct` | N/A | N/A | 0.562 ± 0.140 | 0.592 [0.143] | 0.169–0.797 (n=29) | 0.080 ± 0.058 | 0.083 [0.055] | 0.000–0.182 (n=29) |

> Format: mean ± std | median [IQR] | min–max (n)

## Group B

| Model | Cosine: label↔centroid | Mean cosine: label↔docs | Cosine: label↔mean(top-words) | Jaccard: label↔top-words |
|---|---|---|---|---|
| `meta-llama_llama-3.1-8b-instruct` | N/A | N/A | 0.432 ± 0.032 | 0.437 [0.049] | 0.391–0.462 (n=4) | 0.066 ± 0.086 | 0.042 [0.133] | 0.000–0.182 (n=4) |
| `meta-llama_llama-3.3-70b-instruct` | N/A | N/A | 0.767 ± 0.000 | 0.767 [0.000] | 0.767–0.767 (n=1) | 0.091 ± 0.000 | 0.091 [0.000] | 0.091–0.091 (n=1) |
| `qwen_qwen2.5-coder-7b-instruct` | N/A | N/A | N/A | N/A |
| `qwen_qwen3-vl-30b-a3b-instruct` | N/A | N/A | 0.388 ± 0.054 | 0.368 [0.103] | 0.346–0.449 (n=3) | 0.088 ± 0.091 | 0.083 [0.182] | 0.000–0.182 (n=3) |

> Format: mean ± std | median [IQR] | min–max (n)

## Group C

| Model | Cosine: label↔centroid | Mean cosine: label↔docs | Cosine: label↔mean(top-words) | Jaccard: label↔top-words |
|---|---|---|---|---|
| `meta-llama_llama-3.1-8b-instruct` | N/A | N/A | 0.764 ± 0.087 | 0.764 [0.124] | 0.702–0.826 (n=2) | 0.091 ± 0.129 | 0.091 [0.182] | 0.000–0.182 (n=2) |
| `meta-llama_llama-3.3-70b-instruct` | N/A | N/A | 0.841 ± 0.000 | 0.841 [0.000] | 0.841–0.841 (n=1) | 0.091 ± 0.000 | 0.091 [0.000] | 0.091–0.091 (n=1) |
| `qwen_qwen2.5-coder-7b-instruct` | N/A | N/A | N/A | N/A |
| `qwen_qwen3-vl-30b-a3b-instruct` | N/A | N/A | 0.516 ± 0.219 | 0.553 [0.433] | 0.281–0.714 (n=3) | 0.088 ± 0.091 | 0.083 [0.182] | 0.000–0.182 (n=3) |

> Format: mean ± std | median [IQR] | min–max (n)

## Group D

| Model | Cosine: label↔centroid | Mean cosine: label↔docs | Cosine: label↔mean(top-words) | Jaccard: label↔top-words |
|---|---|---|---|---|
| `meta-llama_llama-3.1-8b-instruct` | N/A | N/A | 0.540 ± 0.201 | 0.540 [0.374] | 0.323–0.832 (n=7) | 0.091 ± 0.108 | 0.083 [0.200] | 0.000–0.273 (n=7) |
| `meta-llama_llama-3.3-70b-instruct` | N/A | N/A | 0.665 ± 0.204 | 0.714 [0.381] | 0.367–0.849 (n=5) | 0.085 ± 0.088 | 0.083 [0.171] | 0.000–0.200 (n=5) |
| `qwen_qwen2.5-coder-7b-instruct` | N/A | N/A | N/A | N/A |
| `qwen_qwen3-vl-30b-a3b-instruct` | N/A | N/A | 0.505 ± 0.162 | 0.520 [0.309] | 0.250–0.696 (n=13) | 0.075 ± 0.072 | 0.083 [0.136] | 0.000–0.182 (n=13) |

> Format: mean ± std | median [IQR] | min–max (n)

# MOOC Forum Topic Analysis -- LLM Part

## Overview

Implementation of the instruction Steps **10–17** corresponds to Python files **step8–step15** in the current workflow.  
Steps **1–7** were adapted from [XqFeng-Josie/AI_Education_Paper_Replicate](https://github.com/XqFeng-Josie/AI_Education_Paper_Replicate/tree/main/MOOC_Forum_Topic).

## Quick Start

### 1. Environment Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run Pipeline

```bash
python step1_preprocess_data.py
python step2_text_preprocessing.py
python step3_generate_embeddings.py
python step4_train_bertopic.py 
python step4_visualize_grid_search.py
python step6_traditional_models.py
python step7_paper_comparison.py
python step8_export_per_topic_artifacts.py
python step9_build_llm_prompts.py
python step10_rum_llms.py
python step11_validate_normalize.py
python step12_label_vs_centroid.py
python step13_label_vs_docs.py
python step14_label_vs_keywords.py
python step15_aggregate_report.py

# Step 6 with specific models
python step6_traditional_models.py "LDA,LSI"
```
## Step Instruction
Step 11 - Prompt and decoding settings (constant across models)
Use a single instruction prompt that (a) shows the subset label (A/B/C/D), (b) lists the ordered top-10 keywords, (c) includes the representative snippets, and (d) requests one concise human-readable label with ≤4 words. Require strict JSON output (e.g., {"label": "...", "rationale": "..."}), and run deterministically (temperature=0, top_p=1, max_new_tokens≈32). 
Your prompt can be similar to this (please change the word “Biology” to our context), but you can try several variations by paraphrasing to get better results. (Prompt is cited by: Kozlowski, D., Pradier, C., & Benz, P. (2024). Generative AI for automatic topic labelling. arXiv preprint arXiv:2408.07003.)

Step 12 - Run LLMs for multiple times and store raw outputs
Use Llama 3.1-8B, Llama 3.3-70B-Instruct, Qwen2.5-7B, Qwen3-30B models. For every topic in each subset, call each model once (then we can run them 3 times) and store outputs to separate files (e.g., labels_A_llama_70B.jsonl, labels_A_Qwen_30B.jsonl, …). 
Keep model outputs separate for evaluation.

Step 13 - Output validation and normalization
Parse JSON; drop responses that violate the schema, re-prompt once if needed. 
Normalize labels minimally (trim spaces, lowercase, strip leading/trailing punctuation) while preserving wording (do not stem/lemmatize the label itself). 
Enforce the ≤4-word constraint; if violated, keep the first four tokens.

Step 14 - Compute cosine similarity between the label and the topic centroid
For each topic, compute a topic centroid embedding by averaging the document embeddings of that topic (use the same sentence-embedding model already adopted in the study for consistency which is all-MiniLM-L6-v2). 
Compute the label’s text embedding. 
Record cosine similarity(label, topic-centroid) per topic and per LLM.

Step 15 - Compute average cosine similarity between the label and representative documents
For each topic, embed the representative snippets selected in Step 10. 
Compute cosine similarity between the label embedding and each snippet embedding, then take the mean (also retain min/max). 
Record per topic and per LLM.

Step 16 - Compare the label with top-N keywords
For each topic, (a) compute an embedding-based comparison by averaging the embeddings of the top-10 keywords into a single vector, then compute cosine similarity(label, mean(top-word embeddings))
(b) compute a token-overlap comparison via Jaccard similarity between the set of lowercased/lemmatized label tokens and the set of top-10 keywords. 
Record both per topic and per LLM.

Step 17 - Aggregate and summarize per model (reported separately for A,B,C,D)
For each LLM and subset, report distribution summaries across topics for the three comparisons above (cosine with centroid; mean cosine with representative documents; cosine with mean(top-word embeddings) and Jaccard with top-words): mean, standard deviation, median, interquartile range, minimum, maximum.

## Experiment Results

### Group A

| Model | Cosine: label↔centroid | Mean cosine: label↔docs | Cosine: label↔mean(top-words) | Jaccard: label↔top-words |
|---|---|---|---|---|
| `meta-llama_llama-3.1-8b-instruct_op1` | 0.589 ± 0.150 | 0.600 [0.221] | 0.111–0.852 (n=161) | 0.457 ± 0.125 | 0.464 [0.170] | 0.087–0.801 (n=161) | 0.554 ± 0.154 | 0.566 [0.204] | 0.020–0.848 (n=161) | 0.095 ± 0.086 | 0.083 [0.182] | 0.000–0.400 (n=161) |
| `meta-llama_llama-3.3-70b-instruct_op1` | 0.595 ± 0.142 | 0.601 [0.206] | 0.223–0.853 (n=161) | 0.461 ± 0.117 | 0.468 [0.155] | 0.169–0.794 (n=161) | 0.598 ± 0.150 | 0.613 [0.223] | 0.175–0.877 (n=161) | 0.103 ± 0.081 | 0.091 [0.140] | 0.000–0.300 (n=161) |
| `qwen_qwen-2.5-7b-instruct_op1` | 0.583 ± 0.157 | 0.605 [0.237] | 0.170–0.853 (n=161) | 0.452 ± 0.130 | 0.463 [0.179] | 0.130–0.794 (n=161) | 0.599 ± 0.162 | 0.611 [0.241] | 0.133–0.905 (n=161) | 0.099 ± 0.078 | 0.091 [0.143] | 0.000–0.300 (n=161) |
| `qwen_qwen3-vl-30b-a3b-instruct_op1` | 0.605 ± 0.142 | 0.612 [0.210] | 0.117–0.898 (n=161) | 0.471 ± 0.120 | 0.477 [0.154] | 0.087–0.768 (n=161) | 0.568 ± 0.143 | 0.574 [0.190] | 0.146–0.868 (n=161) | 0.095 ± 0.079 | 0.083 [0.182] | 0.000–0.300 (n=161) |

> Format: mean ± std | median [IQR] | min–max (n)

### Group B

| Model | Cosine: label↔centroid | Mean cosine: label↔docs | Cosine: label↔mean(top-words) | Jaccard: label↔top-words |
|---|---|---|---|---|
| `meta-llama_llama-3.1-8b-instruct_op1` | 0.592 ± 0.155 | 0.596 [0.234] | 0.235–0.881 (n=25) | 0.464 ± 0.147 | 0.440 [0.247] | 0.194–0.757 (n=25) | 0.481 ± 0.153 | 0.450 [0.200] | 0.251–0.835 (n=25) | 0.096 ± 0.088 | 0.083 [0.174] | 0.000–0.300 (n=25) |
| `meta-llama_llama-3.3-70b-instruct_op1` | 0.579 ± 0.133 | 0.601 [0.240] | 0.361–0.814 (n=25) | 0.457 ± 0.128 | 0.453 [0.209] | 0.290–0.768 (n=25) | 0.503 ± 0.119 | 0.492 [0.143] | 0.251–0.767 (n=25) | 0.093 ± 0.076 | 0.091 [0.049] | 0.000–0.300 (n=25) |
| `qwen_qwen-2.5-7b-instruct_op1` | 0.563 ± 0.137 | 0.540 [0.231] | 0.362–0.823 (n=25) | 0.441 ± 0.124 | 0.400 [0.189] | 0.290–0.673 (n=25) | 0.512 ± 0.135 | 0.529 [0.136] | 0.251–0.777 (n=25) | 0.077 ± 0.061 | 0.091 [0.091] | 0.000–0.200 (n=25) |
| `qwen_qwen3-vl-30b-a3b-instruct_op1` | 0.571 ± 0.105 | 0.589 [0.149] | 0.333–0.724 (n=25) | 0.448 ± 0.099 | 0.438 [0.167] | 0.273–0.683 (n=25) | 0.456 ± 0.112 | 0.449 [0.142] | 0.251–0.665 (n=25) | 0.088 ± 0.073 | 0.083 [0.052] | 0.000–0.300 (n=25) |

> Format: mean ± std | median [IQR] | min–max (n)

### Group C

| Model | Cosine: label↔centroid | Mean cosine: label↔docs | Cosine: label↔mean(top-words) | Jaccard: label↔top-words |
|---|---|---|---|---|
| `meta-llama_llama-3.1-8b-instruct_op1` | 0.632 ± 0.191 | 0.671 [0.316] | 0.294–0.977 (n=22) | 0.494 ± 0.202 | 0.489 [0.279] | 0.087–0.951 (n=22) | 0.570 ± 0.191 | 0.543 [0.264] | 0.143–0.899 (n=22) | 0.094 ± 0.085 | 0.083 [0.182] | 0.000–0.300 (n=22) |
| `meta-llama_llama-3.3-70b-instruct_op1` | 0.613 ± 0.188 | 0.636 [0.260] | 0.123–0.915 (n=22) | 0.475 ± 0.189 | 0.489 [0.190] | 0.099–0.863 (n=22) | 0.619 ± 0.169 | 0.616 [0.278] | 0.243–0.899 (n=22) | 0.103 ± 0.086 | 0.091 [0.182] | 0.000–0.300 (n=22) |
| `qwen_qwen-2.5-7b-instruct_op1` | 0.584 ± 0.199 | 0.630 [0.232] | 0.141–0.915 (n=22) | 0.458 ± 0.182 | 0.435 [0.184] | 0.126–0.863 (n=22) | 0.606 ± 0.176 | 0.625 [0.287] | 0.228–0.899 (n=22) | 0.081 ± 0.076 | 0.091 [0.091] | 0.000–0.200 (n=22) |
| `qwen_qwen3-vl-30b-a3b-instruct_op1` | 0.631 ± 0.166 | 0.676 [0.213] | 0.201–0.921 (n=22) | 0.491 ± 0.181 | 0.489 [0.184] | 0.162–0.897 (n=22) | 0.593 ± 0.167 | 0.592 [0.218] | 0.281–0.888 (n=22) | 0.105 ± 0.102 | 0.083 [0.182] | 0.000–0.300 (n=22) |

> Format: mean ± std | median [IQR] | min–max (n)

### Group D

| Model | Cosine: label↔centroid | Mean cosine: label↔docs | Cosine: label↔mean(top-words) | Jaccard: label↔top-words |
|---|---|---|---|---|
| `meta-llama_llama-3.1-8b-instruct_op1` | 0.571 ± 0.145 | 0.562 [0.176] | 0.195–0.865 (n=94) | 0.456 ± 0.128 | 0.463 [0.200] | 0.147–0.733 (n=94) | 0.531 ± 0.167 | 0.531 [0.264] | 0.169–0.859 (n=94) | 0.093 ± 0.078 | 0.083 [0.182] | 0.000–0.300 (n=94) |
| `meta-llama_llama-3.3-70b-instruct_op1` | 0.560 ± 0.153 | 0.569 [0.218] | 0.150–0.865 (n=94) | 0.446 ± 0.131 | 0.437 [0.195] | 0.133–0.762 (n=94) | 0.571 ± 0.162 | 0.565 [0.244] | 0.234–0.877 (n=94) | 0.090 ± 0.083 | 0.091 [0.182] | 0.000–0.300 (n=94) |
| `qwen_qwen-2.5-7b-instruct_op1` | 0.531 ± 0.152 | 0.525 [0.225] | 0.144–0.876 (n=94) | 0.423 ± 0.127 | 0.423 [0.207] | 0.147–0.702 (n=94) | 0.581 ± 0.168 | 0.574 [0.217] | 0.178–0.935 (n=94) | 0.087 ± 0.077 | 0.091 [0.167] | 0.000–0.300 (n=94) |
| `qwen_qwen3-vl-30b-a3b-instruct_op1` | 0.560 ± 0.148 | 0.556 [0.225] | 0.189–0.912 (n=94) | 0.444 ± 0.126 | 0.451 [0.183] | 0.167–0.794 (n=94) | 0.537 ± 0.162 | 0.539 [0.248] | 0.177–0.893 (n=94) | 0.089 ± 0.079 | 0.083 [0.182] | 0.000–0.300 (n=94) |

> Format: mean ± std | median [IQR] | min–max (n)

---
  
## 📖 Reference
Khodeir, N., & Elghannam, F. (2024). Efficient topic identification for urgent MOOC Forum posts using BERTopic and traditional topic modeling techniques. *Education and Information Technologies*. Springer.
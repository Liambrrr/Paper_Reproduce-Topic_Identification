# Weekly Progress Report

**Date:** Nov 3, 2025
**Project:** Paper_Reproduce-Topic_Identification  
**Author:** Liam Tang  

---

## Overview

Implementation of the instruction Steps **10–17** corresponds to Python files **step8–step15** in the current workflow.  
Steps **1–7** were adapted from [XqFeng-Josie/AI_Education_Paper_Replicate](https://github.com/XqFeng-Josie/AI_Education_Paper_Replicate/tree/main/MOOC_Forum_Topic).

---
## Progress Summary Week 2
current results:

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

### Problems from last week

#### 1. Model Updates
The AWS API also does not support non-instruct version of llama-3.1-7b, qwen-2.5-8b and qwen-3-30b, so no change was made on the way of calling cloud models.

#### 2. Validation and Normalization
Had problem with not finished rationale, so rationale is limited to at most 10 words. Also variations of prompts were attempted.

##### Variation 1 - Original
System prompt:
```text
You label topics for MOOC forum posts. 
Return a STRICT JSON object with exactly the keys 'label' and 'rationale'. 
Constraints:
- 'label' must be a concise, human-readable topic name of at most 4 words.
- Do not include quotes, numbering, brackets, or punctuation in the label.
- 'rationale' is a short sentence explaining why the label fits (≤ 10 words).
- Respond with ONLY the JSON object. No extra text before or after.
```
User Prompt:
```text
Subset: {group} (urgent MOOC forum posts)
Topic ID: {topic_id} | Topic Size: {topic_size}
Top keywords (ordered):
 - {keyword_1}, {keyword_2}, ..., {keyword_n}
Representative snippets (truncated):
 [1] {snippet_1}
 [2] {snippet_2}
 ...
Task: Based on the keywords and snippets, provide a concise human-readable label (<=4 words) that best describes this topic.
If multiple plausible labels exist, choose the most general, course-agnostic phrasing.
Return JSON ONLY in the form: {"label": "<≤4 words>", "rationale": "<≤10 words>"}
```

##### Variation 2 - minimal
System prompt:
```text
You label topics for MOOC forum posts.
Return ONLY this strict JSON object:
{"label": "<≤4 words>", "rationale": "<≤10 words>"}.
Rules:
- "label": concise, human-readable, ≤4 words, no quotes/brackets/punctuation.
- "rationale": ≤10 words, short reason for fit.
No extra text before/after the JSON.
```
User Prompt:
```text
Corpus: MOOC forum posts, subset {group}
Topic ID: {topic_id} | Topic size: {topic_size}
Keywords (ordered):
[ {keyword_1}, {keyword_2}, ..., {keyword_n} ]
Representative snippets (truncated):
[1] {snippet_1}
[2] {snippet_2}
 ...
Task: Extract a short topic name (≤4 words) and a ≤10-word rationale.
Output JSON ONLY: {"label": "<≤4 words>", "rationale": "<≤10 words>"}
```

##### Variation 3 - paraphrase of original
System prompt:
```text
You are labeling topics for a MOOC forum corpus.
Output must be ONLY this JSON: {"label": "<≤4 words>", "rationale": "<≤10 words>"}.
Constraints:
- label: ≤4 words, general, course-agnostic, no punctuation/quotes/brackets.
- rationale: ≤10 words, briefly why the label fits.
No additional commentary.
```
User Prompt:
```text
I have a corpus of MOOC forum posts with many topics.
This topic is described by the following keywords:
[ {keyword_1}, {keyword_2}, ..., {keyword_n} ]
Consider these representative snippets (truncated):
[1] {snippet_1}
[2] {snippet_2}
 ...
Based on the information above, produce a short topic label (one to four words)
that best represents the topic, and a ≤10-word rationale.
Format: {"label": "<≤4 words>", "rationale": "<≤10 words>"}

[Context] Subset: {group} | Topic ID: {topic_id} | Topic Size: {topic_size}
```

### Current Problems/ Things to be guided

llama-3.3-70b-instruct only outperforms 3.1-8b-instruct for group A(general topics), but the difference in performance is tiny. Based on the report table it seems that llama-3.3-70b-instruct performs better with larger input size.

---
## Progress Summary Week 1

### 1. Model Updates
Since some of the original models are unavailable or too large to host locally, the following **OpenRouter-accessible models** were used:

- `meta-llama/llama-3.1-8b-instruct`
- `qwen/qwen2.5-7b-instruct`   
- `qwen/qwen3-vl-30b-a3b-instruct`
Temporary results are at results/metrics/summary_step15.md.
---

### 2. Step 11 – Validation and Normalization

- Out of ~1,200 total samples, most are rejected.  
- **Re-prompting was disabled** (`--reprompt 0`).  
- The primary cause of rejection was **incomplete JSON outputs** — models often produced truncated rationales due to token limits.

#### Example (Rejected)
```json
"response": {
  "text": "{\n  \"label\": \"MOOC Course Issues\",\n  \"rationale\": \"This topic involves problems and questions related to a massive open online course, including",
  "finish_reason": "length",
  "usage": {"prompt_tokens": 489, "completion_tokens": 32, "total_tokens": 521}
}
```
#### Example (Accepted)
```json
"response": {
  "text": "{\n  \"label\": \"Article Access Issues\",\n  \"rationale\": \"Topic revolves around accessing articles, subscription requirements, and related problems.\" \n}",
  "finish_reason": "stop",
  "usage": {"prompt_tokens": 328, "completion_tokens": 31, "total_tokens": 359}
}
```
### Questions and Clarifications

#### 1. Model Access
Where can llama-3.1-8b, llama-3.3-70b-instruct, qwen/qwen2.5-7b, qwen/qwen3-30b models be accessed via cloud or lightweight endpoints (without local download)?
Pulling these models locally consumes too much storage and memory on my laptop.

#### 2. Prompt Variations
Haven't tried variations of prompts. How many variations should be sufficient?
Current prompt template:
- system prompt:
    - You label topics for MOOC forum posts. 
    - Return a STRICT JSON object with exactly the keys 'label' and 'rationale'. 
    - Constraints:
        - 'label' must be a concise, human-readable topic name of at most 4 words.
        - Do not include quotes, numbering, brackets, or punctuation in the label.
        - 'rationale' is 1–2 short sentences explaining why the label fits.
        - Respond with ONLY the JSON object. No extra text before or after.
- user prompt:
    - Subset: (A/B/C/D) (urgent MOOC forum posts)
    - Topic ID:  | Topic Size: 
    - Top keywords (ordered):
    - Representative snippets (truncated):

    Task: Based on the keywords and snippets, provide a concise human-readable label (≤4 words)
    that best describes this topic.
    Return JSON ONLY in the form: {"label": "<≤4 words>", "rationale": "<short reason>"}

#### 3. Validation Criteria
•	For responses rejected due to "finish_reason": "length",
→ Should they be considered valid if the partial JSON structure is correct up to the cutoff?
•	When re-prompting invalid cases,
→ Does “re-prompt” mean re-sending the same prompt or re-writing the prompt (e.g., simpler version)?

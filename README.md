# Weekly Progress Report

**Date:** Nov 3, 2025
**Project:** Paper_Reproduce-Topic_Identification  
**Author:** Liam Tang  

---

## Overview

Implementation of the instruction Steps **10–17** corresponds to Python files **step8–step15** in the current workflow.  
Steps **1–7** were adapted from [XqFeng-Josie/AI_Education_Paper_Replicate](https://github.com/XqFeng-Josie/AI_Education_Paper_Replicate/tree/main/MOOC_Forum_Topic).

---

## Progress Summary

### 1. Model Updates
Since some of the original models are unavailable or too large to host locally, the following **OpenRouter-accessible models** were used:

- `meta-llama/llama-3.1-8b-instruct`
- `qwen/qwen2.5-7b-instruct`   
- `qwen/qwen3-vl-30b-a3b-instruct`

---

### 2. Step 11 – Validation and Normalization

- Out of ~1,200 total samples, **only 92 were classified as clean**, while ~**1,100 were rejected**.  
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
## Questions and Clarifications

### 1. Model Access
Where can llama-3.1-8b, llama-3.3-70b-instruct, qwen/qwen2.5-7b, qwen/qwen3-30b models be accessed via cloud or lightweight endpoints (without local download)?
Pulling these models locally consumes too much storage and memory on my laptop.

### 2. Prompt Variations
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

### 3. Validation Criteria
•	For responses rejected due to "finish_reason": "length",
→ Should they be considered valid if the partial JSON structure is correct up to the cutoff?
•	When re-prompting invalid cases,
→ Does “re-prompt” mean re-sending the same prompt or re-writing the prompt (e.g., simpler version)?
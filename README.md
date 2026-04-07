---
Title :  Loan Underwriting Env
emoji :  🏦
colorFrom :  blue
colorTo : green
sdk : docker
app_port : 7860
tags :
  - openenv
  - finance
  - loan-underwriting
  - mortgage
  - rl-environment
pinned : false
---


## 📌 Overview

Mortgage underwriting is a high-stakes, rule-governed decision process performed daily by human officers at banks and lenders. This environment simulates that exact workflow, giving AI agents the same data a real underwriter sees and scoring them on the same criteria a real lender would use.

An agent must:
- Compute **DTI** (Debt-to-Income) and **LTV** (Loan-to-Value) ratios
- Apply policy rules to **approve**, **reject**, or **escalate** each applicant
- Assign accurate **interest rates** for approved applicants
- Identify all applicable **risk flags**
- Manage **portfolio constraints** (capital budgets, risk concentration caps)

This makes it an ideal benchmark for structured financial reasoning — rules are explicit and deterministic, but edge cases (borderline DTI, fraud detection, thin files) challenge even frontier models.

---

## 🎯 Tasks

| # | Name | Difficulty | Applicants | Key Challenge |
|---|------|-----------|-----------|---------------|
| `task_1_easy` | Single Applicant Underwriting | 🟢 Easy | 1 | DTI/LTV math + correct decision |
| `task_2_medium` | Batch with Capital Constraints | 🟡 Medium | 6 | Portfolio budget + risk cap |
| `task_3_hard` | Edge Case Portfolio | 🔴 Hard | 8 | Fraud, thin files, borderline DTI |

### Scoring Breakdown

<details>
<summary><b>Task 1 — Easy (click to expand)</b></summary>

| Component | Weight | Criteria |
|-----------|--------|----------|
| Decision correctness | **0.50** | Exact match: `approve / reject / escalate` |
| Interest rate accuracy | **0.25** | Within ±0.5% of ground truth (approve only) |
| Risk flag recall | **0.20** | Fraction of ground-truth flags identified |
| Risk flag precision | **0.05** | Penalty for false-positive flags |

</details>

<details>
<summary><b>Task 2 — Medium (click to expand)</b></summary>

| Component | Weight | Criteria |
|-----------|--------|----------|
| Decision accuracy | **0.40** | Per-applicant correct decision rate |
| Capital budget | **0.25** | Full credit if under budget; proportional if over |
| Risk cap | **0.20** | Full credit if ≤2 high-DTI approvals |
| Interest rate accuracy | **0.10** | For correctly approved applicants |
| Fraud safety bonus | **0.05** | All mandatory-escalation cases caught |

</details>

<details>
<summary><b>Task 3 — Hard (click to expand)</b></summary>

| Component | Weight | Criteria |
|-----------|--------|----------|
| Decision accuracy | **0.30** | Per-applicant correct decision rate |
| Escalation F1 | **0.30** | Precision + recall on escalate class |
| Risk flag recall | **0.20** | Average flag recall across all applicants |
| Safety | **0.10** | Penalises false approvals of hard-reject cases |
| Fraud detection | **0.10** | Recall on fraud/unverified/prior-default |

</details>

---

## 📊 Baseline Scores

> Measured with `gpt-4o`, `seed=42`, `temperature=0`

| Task | Score | Notes |
|------|-------|-------|
| `task_1_easy` | `0.85 – 0.95` | Occasional rate rounding errors |
| `task_2_medium` | `0.75 – 0.88` | Budget constraint requires portfolio optimisation |
| `task_3_hard` | `0.65 – 0.80` | Edge cases challenge even frontier models |
| **Mean** | **`0.75 – 0.87`** | |

> 🎲 Random agent: `~0.10–0.30` &nbsp;&nbsp;|&nbsp;&nbsp; 📈 All-escalate agent: `~0.45–0.65`

---

## 🔭 Observation Space

Returned by `POST /reset` and `GET /state`:

```json
{
  "task_id": "task_1_easy",
  "step": 0,
  "max_steps": 1,
  "done": false,
  "message": "Episode started. Submit decisions for all applicants in context.applicants.",
  "context": {
    "applicants": [
      {
        "applicant_id": "APP-4821-00",
        "annual_income": 95000,
        "monthly_debt": 1200,
        "credit_score": 710,
        "loan_amount": 320000,
        "property_value": 400000,
        "employment_years": 4.5,
        "employment_type": "salaried",
        "prior_default": false,
        "fraud_flag": false,
        "income_verified": true
      }
    ],
    "policy": "ESCALATE if fraud_flag OR income_verified=false OR prior_default OR 0.40<=DTI<=0.45..."
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `applicant_id` | `string` | Unique ID — copy exactly in your decision |
| `annual_income` | `float` | Annual income in USD |
| `monthly_debt` | `float` | Total monthly debt obligations |
| `credit_score` | `int` | FICO score (580–800 range) |
| `loan_amount` | `float` | Requested loan amount |
| `property_value` | `float` | Appraised property value |
| `employment_years` | `float` | Years at current employer |
| `employment_type` | `string` | `salaried`, `self_employed`, or `contract` |
| `prior_default` | `bool` | Prior loan default on record |
| `fraud_flag` | `bool` | Fraud indicator triggered |
| `income_verified` | `bool` | Income documentation verified |

---

## ⚡ Action Space

Submitted to `POST /step`:

```json
{
  "task_id": "task_1_easy",
  "decisions": [
    {
      "applicant_id": "APP-4821-00",
      "decision": "approve",
      "interest_rate": 7.18,
      "risk_flags": ["high_ltv"],
      "reasoning": "DTI=0.1516 LTV=0.8000 => approve. Rate=6.60."
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `applicant_id` | `string` | Must match exactly from observation |
| `decision` | `string` | `approve`, `reject`, or `escalate` |
| `interest_rate` | `float \| null` | Required for approve; **must be null** otherwise |
| `risk_flags` | `string[]` | All applicable flags (see below) |
| `reasoning` | `string` | Step-by-step justification |

### Risk Flags

| Flag | Trigger Condition |
|------|-------------------|
| `high_dti` | DTI > 0.36 |
| `low_credit_score` | credit_score < 680 |
| `high_ltv` | LTV > 0.80 |
| `short_employment` | employment_years < 2.0 |
| `self_employed_income` | employment_type == `self_employed` |
| `prior_default` | prior_default == true |
| `fraud_flag` | fraud_flag == true |
| `unverified_income` | income_verified == false |

---

## 📐 Underwriting Rules

```
DTI = (monthly_debt × 12) / annual_income
LTV = loan_amount / property_value

━━━ Priority 1 — ESCALATE (human review required) ━━━
  • fraud_flag = true
  • income_verified = false
  • prior_default = true
  • 0.40 ≤ DTI ≤ 0.45  (borderline zone)

━━━ Priority 2 — REJECT (if not escalating) ━━━
  • DTI > 0.45
  • credit_score < 620
  • LTV > 0.97

━━━ Priority 3 — APPROVE (all other cases) ━━━
  interest_rate = round(6.5 + max(0, (DTI−0.28)×4) + max(0, (720−credit_score)×0.01), 2)
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker
- API key for any OpenAI-compatible provider (OpenAI, Groq, Together AI, etc.)

### 1. Clone & Install

```bash
git clone https://huggingface.co/spaces/bhattyuvraj22/loan-underwriting-env
cd loan-underwriting-env
pip install -r requirements.txt
```

### 2. Start the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

### 3. Verify It's Running

```bash
curl http://localhost:7860/health
# {"status":"ok","env":"loan-underwriting-env","version":"1.0.0"}

curl http://localhost:7860/tasks
# Lists all 3 tasks
```

### 4. Run Baseline Inference

**With Groq (free tier available):**
```bash
export HF_TOKEN=gsk_your_groq_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export ENV_URL=http://localhost:7860

python inference.py
```

**With OpenAI:**
```bash
export HF_TOKEN=sk-your_openai_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export ENV_URL=http://localhost:7860

python inference.py
```

**For reproducibility:**
```bash
python inference.py --seed 123
```

---

## 🐳 Docker

```bash
# Build
docker build -t loan-underwriting-env .

# Run
docker run -p 7860:7860 loan-underwriting-env

# Verify
curl http://localhost:7860/health

# Run inference against the container
HF_TOKEN=your_key \
API_BASE_URL=https://api.groq.com/openai/v1 \
MODEL_NAME=llama-3.3-70b-versatile \
ENV_URL=http://localhost:7860 \
python inference.py
```

---

## 🌐 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Root health check — required for HF Space ping |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks with metadata |
| `POST` | `/reset` | Start new episode, returns Observation |
| `POST` | `/step` | Submit decisions, returns reward + info |
| `GET` | `/state?task_id=...` | Get current episode state |

> 📖 Interactive docs available at `http://localhost:7860/docs`

### Full Episode — curl Example

```bash
# Step 1: Reset
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1_easy", "seed": 42}' | python -m json.tool

# Step 2: Submit decisions (copy applicant_id from reset response)
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{
    "task_id": "task_1_easy",
    "decisions": [{
      "applicant_id": "APP-XXXX-00",
      "decision": "approve",
      "interest_rate": 7.18,
      "risk_flags": ["high_ltv"],
      "reasoning": "DTI=0.15 LTV=0.80 => approve"
    }]
  }' | python -m json.tool
```

---

## 🏗️ Project Structure

```
loan-underwriting-env/
│
├── 📄 main.py              # FastAPI app — all HTTP endpoints
├── 📄 inference.py         # Baseline inference script (OpenAI-compatible)
├── 📄 openenv.yaml         # OpenEnv spec — observation/action space definitions
├── 📄 requirements.txt     # Python dependencies
├── 📄 pyproject.toml       # Project metadata + entry points
├── 📄 Dockerfile           # Container definition
├── 📄 uv.lock              # Locked dependency tree
│
├── 📁 env/
│   ├── models.py           # Typed Pydantic models (Observation, AgentAction, Reward)
│   ├── state.py            # Session manager (reset / step / state logic)
│   ├── underwriting.py     # Applicant generator + ground truth computation
│   └── graders/
│       ├── grader1.py      # Easy — decision + rate + flags
│       ├── grader2.py      # Medium — batch + constraints + safety
│       └── grader3.py      # Hard — escalation F1 + fraud detection
│
└── 📁 server/
    └── app.py              # Entry point for multi-mode deployment
```

---

## 🧠 Reward Design

**Partial progress signal** — Every scoring component is independent. An agent that gets decisions right but misses risk flags still earns `0.50–0.75`, not zero. This gives a meaningful learning signal at every skill level.

**Anti-trivial-strategy design** — A lazy "escalate everything" strategy scores only `~0.45–0.65` because decision accuracy penalises incorrect escalations of should-approve applicants. A correct agent scores `0.90+`.

**Safety penalties** — False approvals of hard-reject and fraud applicants incur explicit deductions *on top of* the decision accuracy loss, strongly incentivising conservative handling of risky cases.

---

## 📋 Environment Checklist

| Requirement | Status |
|-------------|--------|
| Real-world task simulation | ✅ Mortgage underwriting |
| OpenEnv spec compliant (`step`, `reset`, `state`) | ✅ |
| 3+ tasks with graders (easy → hard) | ✅ |
| Scores 0.0 – 1.0 with partial credit | ✅ |
| Baseline inference script (`inference.py`) | ✅ |
| HF Space deploys + returns 200 | ✅ |
| Dockerfile builds and runs | ✅ |
| `openenv validate` passes | ✅ |



---

<div align="center">

Built for the **OpenEnv Challenge** · Powered by [FastAPI](https://fastapi.tiangolo.com) · Hosted on [Hugging Face Spaces](https://huggingface.co/spaces)

</div>
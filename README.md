---
title: Loan Underwriting OpenEnv
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
  - finance
  - real-world
---

# Loan Underwriting OpenEnv

A **real-world OpenEnv environment** where an AI agent acts as a mortgage loan underwriting officer — evaluating applicant financial profiles and making `approve / reject / escalate` decisions, assigning interest rates, and identifying risk flags.

This environment models a genuine task performed daily by human underwriters at banks and mortgage lenders. It is suitable for training and evaluating LLM-based agents on structured financial reasoning.

---

## Environment Description & Motivation

Mortgage underwriting is a high-stakes, rule-governed decision process that requires:
- Numerical reasoning (DTI, LTV calculation)
- Policy compliance (hard thresholds for rejection)
- Risk judgment (escalation for ambiguous or fraudulent cases)
- Portfolio management (capital budget and risk concentration limits)

These properties make it an excellent benchmark for LLM agents: the rules are explicit and deterministic, but edge cases (borderline DTI, fraud detection, thin files) require careful reasoning that separates strong agents from weak ones.

---

## Action & Observation Spaces

### Observation (returned by `POST /reset` and `GET /state`)

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
    "policy": "ESCALATE if fraud_flag OR income_verified=false OR prior_default OR 0.40<=DTI<=0.45. REJECT if DTI>0.45 OR credit_score<620 OR LTV>0.97. APPROVE otherwise."
  }
}
```

**Applicant fields:**

| Field | Type | Description |
|---|---|---|
| `applicant_id` | string | Unique ID — copy exactly in your decision |
| `annual_income` | float | Annual income in USD |
| `monthly_debt` | float | Total monthly debt obligations |
| `credit_score` | int | FICO score (580–800 range) |
| `loan_amount` | float | Requested loan amount |
| `property_value` | float | Appraised property value |
| `employment_years` | float | Years at current employer |
| `employment_type` | string | `salaried`, `self_employed`, or `contract` |
| `prior_default` | bool | Prior loan default on record |
| `fraud_flag` | bool | Fraud indicator triggered |
| `income_verified` | bool | Income documentation verified |

### Action (submitted to `POST /step`)

```json
{
  "task_id": "task_1_easy",
  "decisions": [
    {
      "applicant_id": "APP-4821-00",
      "decision": "approve",
      "interest_rate": 7.18,
      "risk_flags": ["high_ltv", "low_credit_score"],
      "reasoning": "DTI=0.1516 LTV=0.8000 => approve. Rate=6.60. Flags: high_ltv"
    }
  ]
}
```

**Decision fields:**

| Field | Type | Description |
|---|---|---|
| `applicant_id` | string | Must match exactly from observation |
| `decision` | string | `approve`, `reject`, or `escalate` |
| `interest_rate` | float \| null | Required for approve; **must be null** otherwise |
| `risk_flags` | string[] | All applicable flags (see below) |
| `reasoning` | string | Agent's step-by-step justification |

**Valid risk flags:**

| Flag | Condition |
|---|---|
| `high_dti` | DTI > 0.36 |
| `low_credit_score` | credit_score < 680 |
| `high_ltv` | LTV > 0.80 |
| `short_employment` | employment_years < 2.0 |
| `self_employed_income` | employment_type == "self_employed" |
| `prior_default` | prior_default == true |
| `fraud_flag` | fraud_flag == true |
| `unverified_income` | income_verified == false |

---

## Underwriting Rules

```
DTI = (monthly_debt × 12) / annual_income
LTV = loan_amount / property_value

Priority 1 — ESCALATE (human review required):
  fraud_flag = true
  income_verified = false
  prior_default = true
  0.40 ≤ DTI ≤ 0.45  (borderline zone)

Priority 2 — REJECT (hard disqualifiers, checked only if not escalating):
  DTI > 0.45
  credit_score < 620
  LTV > 0.97

Priority 3 — APPROVE (all other cases):
  interest_rate = round(6.5 + max(0, (DTI−0.28)×4) + max(0, (720−credit_score)×0.01), 2)
```

---

## Tasks

### Task 1 — Easy: Single Applicant Underwriting

Evaluate **1 applicant**. Apply the underwriting rules, assign an interest rate if approved, and identify all risk flags.

**Scoring (max 1.0):**

| Component | Weight | Description |
|---|---|---|
| Decision correctness | 0.50 | Exact match: approve / reject / escalate |
| Interest rate accuracy | 0.25 | Within ±0.5% of ground truth (approve only) |
| Risk flag recall | 0.20 | Fraction of ground-truth flags identified |
| Risk flag precision | 0.05 | Penalty for false-positive flags |

---

### Task 2 — Medium: Batch Underwriting with Capital Constraints

Evaluate **6 applicants** while respecting:
- `capital_budget`: total approved `loan_amount` must not exceed this value
- `risk_cap = 2`: at most 2 approved applicants may have DTI > 0.36

**Scoring (max 1.0):**

| Component | Weight | Description |
|---|---|---|
| Decision accuracy | 0.40 | Per-applicant correct decision rate |
| Capital budget | 0.25 | Full credit if under budget; proportional if over |
| Risk cap | 0.20 | Full credit if ≤2 high-DTI approvals; proportional otherwise |
| Interest rate accuracy | 0.10 | For correctly approved applicants |
| Fraud safety bonus | 0.05 | Escalated all mandatory-escalation applicants |

---

### Task 3 — Hard: Edge Case Portfolio with Escalation

Evaluate **8 applicants** including forced edge cases:
- At least 1 with `fraud_flag = true`
- At least 1 with `income_verified = false` (thin file)
- At least 1 with borderline DTI in `[0.40, 0.45]`
- Edge cases are shuffled — the agent cannot rely on position

**Scoring (max 1.0):**

| Component | Weight | Description |
|---|---|---|
| Decision accuracy | 0.30 | Per-applicant correct decision rate |
| Escalation F1 | 0.30 | Precision + recall on the escalate class |
| Risk flag recall | 0.20 | Average flag recall across all applicants |
| Safety | 0.10 | Penalises false approvals of hard-reject applicants |
| Fraud detection | 0.10 | Recall on fraud/unverified/prior-default escalation |

---

## Baseline Scores

Scores measured with `gpt-4o`, `seed=42`, `temperature=0`:

| Task | Baseline Score | Notes |
|---|---|---|
| `task_1_easy` | ~0.85–0.95 | Occasional rate rounding errors |
| `task_2_medium` | ~0.75–0.88 | Budget constraint requires portfolio optimisation |
| `task_3_hard` | ~0.65–0.80 | Edge cases challenge even frontier models |
| **Mean** | **~0.75–0.87** | |

A random agent scores approximately `0.10–0.30`. An all-escalate agent scores approximately `0.45–0.65`.

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker (for containerised deployment)
- An API key for any OpenAI-compatible provider (OpenAI, Groq, Together, etc.)

### Local Development

```bash
# 1. Clone the repo
git clone https://huggingface.co/spaces/YOUR_HF_USERNAME/loan-underwriting-env
cd loan-underwriting-env

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the environment server
uvicorn main:app --host 0.0.0.0 --port 7860 --reload

# 4. Verify it's running
curl http://localhost:7860/health
curl http://localhost:7860/tasks
```

### Run Baseline Inference (Groq — free tier available)

```bash
export HF_TOKEN=gsk_your_groq_key
export API_BASE_URL=https://api.groq.com/openai/v1
export MODEL_NAME=llama-3.3-70b-versatile
export ENV_URL=http://localhost:7860

python inference.py
python inference.py --seed 123   # reproducibility check
```

### Run Baseline Inference (OpenAI)

```bash
export HF_TOKEN=sk-your_openai_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
export ENV_URL=http://localhost:7860

python inference.py
```

### Docker

```bash
# Build
docker build -t loan-underwriting-env .

# Run
docker run -p 7860:7860 loan-underwriting-env

# Verify
curl http://localhost:7860/health

# Run inference against Docker container
export ENV_URL=http://localhost:7860
export HF_TOKEN=your_key
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o
python inference.py
```

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root health check — required for HF Space ping |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List all tasks with metadata |
| `POST` | `/reset` | Start new episode, returns Observation |
| `POST` | `/step` | Submit decisions, returns reward + info |
| `GET` | `/state?task_id=...` | Get current episode state |

Interactive API docs available at `http://localhost:7860/docs` when running locally.

### Example: Full Episode via curl

```bash
# 1. Reset
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_1_easy", "seed": 42}' | python -m json.tool

# 2. Step (copy applicant_id from reset response)
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

## Project Structure

```
.
├── main.py              # FastAPI application (all HTTP endpoints)
├── inference.py         # Baseline inference script (OpenAI-compatible)
├── openenv.yaml         # OpenEnv spec with observation/action space definitions
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── .dockerignore        # Excludes venv, __pycache__, .git from image
└── env/
    ├── __init__.py
    ├── models.py        # Typed Pydantic models (Observation, AgentAction, Reward, ...)
    ├── state.py         # Session manager (reset/step/state logic)
    ├── underwriting.py  # Applicant generator + ground truth computation
    └── graders/
        ├── __init__.py
        ├── grader1.py   # Easy task grader  (decision + rate + flags)
        ├── grader2.py   # Medium task grader (batch + constraints + safety)
        └── grader3.py   # Hard task grader  (escalation F1 + fraud detection)
```

---

## Reward Design Notes

**Partial progress signal:** Every scoring component is independent, so an agent that gets decisions right but misses risk flags still receives meaningful reward (0.50–0.75 range rather than 0). This provides a learning signal even for imperfect agents.

**Anti-trivial-strategy design:** A lazy "escalate everything" strategy scores approximately 0.45–0.65 (below a correct agent's 0.90+) because decision accuracy penalises incorrect escalations of should-approve applicants.

**Safety penalties:** False approvals of hard-reject and fraud applicants incur explicit score deductions beyond the decision accuracy loss, incentivising conservative handling of risky cases.

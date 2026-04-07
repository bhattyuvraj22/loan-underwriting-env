"""
Applicant generator and ground-truth computer for Loan Underwriting OpenEnv.
All logic here is the single source of truth — graders import compute_ground_truth.
"""
import random
from .models import ApplicantProfile

BASE_RATE = 6.5  # base interest rate percent

# Valid risk flag names (used by graders and inference prompt)
RISK_FLAGS = [
    "high_dti",
    "low_credit_score",
    "high_ltv",
    "short_employment",
    "self_employed_income",
    "prior_default",
    "fraud_flag",
    "unverified_income",
]


def make_applicant(
    rng: random.Random,
    force_fraud: bool = False,
    force_thin: bool = False,
    force_borderline: bool = False,
    index: int = 0,
) -> ApplicantProfile:
    """
    Generate a random ApplicantProfile.
    force_fraud     — sets fraud_flag=True (for hard task)
    force_thin      — sets income_verified=False (for hard task)
    force_borderline — sets DTI firmly in [0.40, 0.45] (for hard task)
    index           — used to guarantee unique applicant_id within a batch
    """
    credit_score = rng.randint(580, 800)
    annual_income = rng.randint(40_000, 180_000)
    monthly_debt = rng.randint(200, 3_000)
    loan_amount = rng.randint(80_000, 600_000)
    property_value = loan_amount * rng.uniform(1.0, 1.4)
    employment_years = round(rng.uniform(0.5, 15.0), 1)
    employment_type = rng.choice(["salaried", "self_employed", "contract"])
    prior_default = rng.random() < 0.1
    fraud_flag = force_fraud or (rng.random() < 0.05)
    income_verified = not force_thin and (rng.random() < 0.9)

    if force_borderline:
        # FIX: Use round() not int() so DTI stays firmly in [0.40, 0.45]
        # int() truncation can push DTI just below 0.40, missing the escalation trigger
        target_dti = rng.uniform(0.401, 0.449)  # avoid exact boundaries for safety
        monthly_debt = round(annual_income * target_dti / 12)
        # Verify and clamp: if rounding pushed us out, adjust by +1
        actual_dti = (monthly_debt * 12) / annual_income
        if actual_dti < 0.40:
            monthly_debt += 1

    # Unique applicant_id: combine randint with index to prevent collisions in batch
    uid = rng.randint(1000, 9999)
    applicant_id = f"APP-{uid:04d}-{index:02d}"

    return ApplicantProfile(
        applicant_id=applicant_id,
        annual_income=annual_income,
        monthly_debt=monthly_debt,
        credit_score=credit_score,
        loan_amount=loan_amount,
        property_value=property_value,
        employment_years=employment_years,
        employment_type=employment_type,
        prior_default=prior_default,
        fraud_flag=fraud_flag,
        income_verified=income_verified,
    )


def compute_dti(applicant: ApplicantProfile) -> float:
    return (applicant.monthly_debt * 12) / applicant.annual_income


def compute_ltv(applicant: ApplicantProfile) -> float:
    return applicant.loan_amount / applicant.property_value


def compute_interest_rate(dti: float, credit_score: int) -> float:
    """Compute risk-adjusted interest rate for approved loans."""
    spread = max(0.0, (dti - 0.28) * 4) + max(0.0, (720 - credit_score) * 0.01)
    return round(BASE_RATE + spread, 2)


def compute_ground_truth(applicant: ApplicantProfile) -> dict:
    """
    Compute the deterministic correct decision for an applicant.
    This is the single source of truth used by all graders.

    Returns dict with keys:
      decision, interest_rate, risk_flags, dti, ltv
    """
    dti = compute_dti(applicant)
    ltv = compute_ltv(applicant)

    # Priority 1: Mandatory escalation triggers
    must_escalate = (
        applicant.fraud_flag
        or not applicant.income_verified
        or applicant.prior_default
        or (0.40 <= dti <= 0.45)
    )

    # Priority 2: Hard rejection (only checked if not escalating)
    hard_reject = (
        dti > 0.45
        or applicant.credit_score < 620
        or ltv > 0.97
    )

    if must_escalate:
        decision = "escalate"
    elif hard_reject:
        decision = "reject"
    else:
        decision = "approve"

    # Risk flags (independent of decision — always evaluate all)
    flags = []
    if dti > 0.36:
        flags.append("high_dti")
    if applicant.credit_score < 680:
        flags.append("low_credit_score")
    if ltv > 0.80:
        flags.append("high_ltv")
    if applicant.employment_years < 2:
        flags.append("short_employment")
    if applicant.employment_type == "self_employed":
        flags.append("self_employed_income")
    if applicant.prior_default:
        flags.append("prior_default")
    if applicant.fraud_flag:
        flags.append("fraud_flag")
    if not applicant.income_verified:
        flags.append("unverified_income")

    # Interest rate: only meaningful for approved loans
    rate = compute_interest_rate(dti, applicant.credit_score) if decision == "approve" else None

    return {
        "decision": decision,
        "interest_rate": rate,
        "risk_flags": flags,
        "dti": round(dti, 4),
        "ltv": round(ltv, 4),
    }

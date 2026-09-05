"""
Generator for benchmark datasets with curated ground-truth data quality,
target leakage, time drift, and slice fairness issues.
"""

import os
import numpy as np
import pandas as pd


def generate_benchmark_datasets(output_dir: str = "data/samples"):
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Telecom Churn with Target Leakage
    n_churn = 1200
    customer_id = [f"CUST-{10000 + i}" for i in range(n_churn)]
    gender = np.random.choice(["Male", "Female"], size=n_churn)
    age = np.random.randint(18, 75, size=n_churn)
    tenure_months = np.random.randint(1, 72, size=n_churn)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, size=n_churn), 2)
    contract_type = np.random.choice(["Month-to-Month", "One-Year", "Two-Year"], size=n_churn, p=[0.55, 0.25, 0.20])
    tech_support_calls = np.random.poisson(lam=1.5, size=n_churn)

    # True churn probability based on tenure, monthly_charges, contract
    churn_logit = -1.5 - 0.04 * tenure_months + 0.02 * monthly_charges + (contract_type == "Month-to-Month") * 0.8
    churn_prob = 1 / (1 + np.exp(-churn_logit))
    churn = (np.random.rand(n_churn) < churn_prob).astype(int)

    # INJECTED TARGET LEAKAGE: refund_issued_after_churn (92% correlated with churn)
    refund_issued_after_churn = np.where(
        churn == 1,
        np.random.choice([1, 0], size=n_churn, p=[0.92, 0.08]),
        np.random.choice([1, 0], size=n_churn, p=[0.02, 0.98])
    )

    # INJECTED PROXY LEAKAGE: cancellation_survey_score
    cancellation_survey_score = np.where(churn == 1, np.random.normal(1.2, 0.4, size=n_churn), np.nan)

    # INJECTED INFORMATIVE MISSINGNESS: total_charges missing for 12%
    total_charges = monthly_charges * tenure_months + np.random.normal(0, 10, size=n_churn)
    total_charges[total_charges < 0] = 0
    # Make missingness correlate with contract_type
    missing_mask = (contract_type == "Month-to-Month") & (np.random.rand(n_churn) < 0.25)
    total_charges[missing_mask] = np.nan

    churn_df = pd.DataFrame({
        "customer_id": customer_id,
        "gender": gender,
        "age": age,
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges,
        "total_charges": np.round(total_charges, 2),
        "contract_type": contract_type,
        "tech_support_calls": tech_support_calls,
        "refund_issued_after_churn": refund_issued_after_churn,
        "cancellation_survey_score": np.round(cancellation_survey_score, 1),
        "churn": churn
    })
    churn_path = os.path.join(output_dir, "telecom_churn_leakage.csv")
    churn_df.to_csv(churn_path, index=False)

    # 2. Credit Risk with Distribution Drift
    n_credit = 1500
    app_id = [f"APP-{20000 + i}" for i in range(n_credit)]
    dates_base = pd.date_range("2023-01-01", periods=1000, freq="8h")
    dates_curr = pd.date_range("2024-01-01", periods=500, freq="16h")
    all_dates = list(dates_base.strftime("%Y-%m-%d %H:%M")) + list(dates_curr.strftime("%Y-%m-%d %H:%M"))

    # Baseline distribution (first 1000)
    income_base = np.random.normal(65000, 15000, size=1000)
    credit_score_base = np.random.normal(710, 45, size=1000)
    dti_base = np.random.normal(0.28, 0.08, size=1000)

    # Shifted / Drifted distribution (last 500)
    income_curr = np.random.normal(48000, 18000, size=500)  # Significant drop in income
    credit_score_curr = np.random.normal(640, 60, size=500) # Lower credit scores
    dti_curr = np.random.normal(0.42, 0.12, size=500)       # Elevated debt-to-income

    all_income = np.clip(np.concatenate([income_base, income_curr]), 15000, 200000)
    all_credit_score = np.clip(np.concatenate([credit_score_base, credit_score_curr]), 300, 850)
    all_dti = np.clip(np.concatenate([dti_base, dti_curr]), 0.05, 0.85)

    loan_amount = np.round(np.random.uniform(5000, 40000, size=n_credit), 0)
    employment_years = np.random.randint(0, 25, size=n_credit)
    region = np.random.choice(["North", "South", "East", "West"], size=n_credit)

    risk_logit = 2.0 - (all_credit_score - 600) * 0.015 + all_dti * 4.0 - (all_income / 100000) * 1.5
    default_prob = 1 / (1 + np.exp(-risk_logit))
    default_risk = (np.random.rand(n_credit) < default_prob).astype(int)

    credit_df = pd.DataFrame({
        "application_id": app_id,
        "application_date": all_dates,
        "annual_income": np.round(all_income, 2),
        "credit_score": np.round(all_credit_score, 0),
        "debt_to_income_ratio": np.round(all_dti, 3),
        "loan_amount": loan_amount,
        "employment_years": employment_years,
        "region": region,
        "default_risk": default_risk
    })
    credit_path = os.path.join(output_dir, "credit_risk_drift.csv")
    credit_df.to_csv(credit_path, index=False)

    # 3. Titanic Quality Audit Dataset
    n_titanic = 891
    p_id = list(range(1, n_titanic + 1))
    pclass = np.random.choice([1, 2, 3], size=n_titanic, p=[0.24, 0.21, 0.55])
    sex = np.random.choice(["male", "female"], size=n_titanic, p=[0.64, 0.36])
    age = np.random.normal(30, 14, size=n_titanic)
    age = np.clip(age, 1, 80)
    # Inject 20% missing in age
    age[np.random.rand(n_titanic) < 0.20] = np.nan

    sibsp = np.random.choice([0, 1, 2, 3, 4], size=n_titanic, p=[0.68, 0.23, 0.05, 0.02, 0.02])
    parch = np.random.choice([0, 1, 2], size=n_titanic, p=[0.76, 0.16, 0.08])
    
    # Fare with extreme outliers
    base_fare = np.where(pclass == 1, 85, np.where(pclass == 2, 25, 12)) + np.random.exponential(15, size=n_titanic)
    # Inject 4% extreme outliers (e.g. fare > 500)
    outlier_idx = np.random.choice(n_titanic, size=int(n_titanic * 0.04), replace=False)
    base_fare[outlier_idx] = np.random.uniform(450, 950, size=len(outlier_idx))

    embarked = np.random.choice(["S", "C", "Q"], size=n_titanic, p=[0.72, 0.19, 0.09])

    surv_logit = 1.0 - 1.2 * (pclass - 1) + (sex == "female") * 2.4 - (np.nan_to_num(age, nan=30) / 40)
    surv_prob = 1 / (1 + np.exp(-surv_logit))
    survived = (np.random.rand(n_titanic) < surv_prob).astype(int)

    titanic_df = pd.DataFrame({
        "passenger_id": p_id,
        "pclass": pclass,
        "sex": sex,
        "age": np.round(age, 1),
        "sibsp": sibsp,
        "parch": parch,
        "fare": np.round(base_fare, 2),
        "embarked": embarked,
        "survived": survived
    })
    titanic_path = os.path.join(output_dir, "titanic_quality_audit.csv")
    titanic_df.to_csv(titanic_path, index=False)

    # 4. Recruitment Bias Slices Dataset
    n_rec = 1000
    cand_id = [f"CAND-{i+1:04d}" for i in range(n_rec)]
    gender_rec = np.random.choice(["Male", "Female", "Non-Binary"], size=n_rec, p=[0.55, 0.40, 0.05])
    experience_years = np.random.randint(1, 15, size=n_rec)
    technical_score = np.clip(np.random.normal(70, 15, size=n_rec), 20, 100)
    dept = np.random.choice(["Engineering", "Sales", "Design", "Marketing"], size=n_rec)
    
    # Subgroup disparity
    hire_logit = -3.0 + 0.04 * technical_score + 0.15 * experience_years + (gender_rec == "Female") * 0.2
    hire_prob = 1 / (1 + np.exp(-hire_logit))
    hired = (np.random.rand(n_rec) < hire_prob).astype(int)

    rec_df = pd.DataFrame({
        "candidate_id": cand_id,
        "gender": gender_rec,
        "department": dept,
        "experience_years": experience_years,
        "technical_score": np.round(technical_score, 1),
        "hired": hired
    })
    rec_path = os.path.join(output_dir, "recruitment_bias_slices.csv")
    rec_df.to_csv(rec_path, index=False)

    print(f"Generated 4 benchmark datasets in {output_dir}")


if __name__ == "__main__":
    generate_benchmark_datasets()

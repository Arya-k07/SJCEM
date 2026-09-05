"""
Deterministic Artifact Generator: Builds reproducible Jupyter Notebooks, Scikit-Learn pipelines,
HTML Decision summaries, and JSON contracts for confirmed human decisions.
"""

import json
from typing import Any, Dict, List, Optional
from dataset_autopilot.schemas import (
    AuditReport,
    DecisionCandidate,
    ExperimentVariantResult,
)


def generate_jupyter_notebook(
    candidate: DecisionCandidate,
    report: AuditReport,
    selected_strategy_id: Optional[str] = None
) -> str:
    """
    Generate a valid, runnable Jupyter Notebook (.ipynb JSON string)
    reproducing the exact baseline, strategy comparison, and final decision pipeline.
    """
    strategy_id = selected_strategy_id or (
        candidate.user_decision.selected_strategy_id if candidate.user_decision else (
            candidate.available_strategies[0] if candidate.available_strategies else "BASELINE"
        )
    )
    affected_col = candidate.affected_columns[0] if candidate.affected_columns else "feature"
    target_col = report.dataset_profile.detected_target or "target"
    is_classification = report.dataset_profile.target_type in ["binary_classification", "multiclass_classification"]

    cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"# Decision Lab Reproducibility Notebook: {candidate.decision_id}\n",
                f"**Dataset**: `{report.dataset_name}`  \n",
                f"**Target**: `{target_col}` ({report.dataset_profile.target_type or 'classification'})  \n",
                f"**Question**: *{candidate.question}*  \n",
                f"**Selected Strategy**: `{strategy_id}`  \n",
                f"**Generated**: `{report.created_at}`  \n\n",
                "---\n",
                "### Context & Hypothesis\n",
                f"{candidate.rationale}\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 1. Environment & Library Setup\n",
                "import numpy as np\n",
                "import pandas as pd\n",
                "from sklearn.model_selection import StratifiedKFold, KFold\n",
                "from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor\n",
                "from sklearn.impute import SimpleImputer\n",
                "from sklearn.preprocessing import StandardScaler, OrdinalEncoder\n",
                "from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, r2_score, mean_absolute_error\n\n",
                "# Set deterministic seed\n",
                "RANDOM_SEED = 42\n",
                "np.random.seed(RANDOM_SEED)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# 2. Load Dataset\n",
                f"# Replace with path to '{report.dataset_name}'\n",
                f"df = pd.read_csv('{report.dataset_name}')\n",
                f"print(f'Loaded dataset with shape: {{df.shape}}')\n",
                "df.head()\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Baseline Data Preparation & Oracle Model\n",
                "We evaluate the model using stratified 5-fold cross-validation with identical parameters.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"TARGET_COL = '{target_col}'\n",
                f"AFFECTED_COLS = {json.dumps(candidate.affected_columns)}\n\n",
                "# Drop invalid target rows\n",
                "valid_mask = df[TARGET_COL].notna()\n",
                "df_clean = df.loc[valid_mask].copy()\n",
                f"y = df_clean[TARGET_COL].values if not {is_classification} else pd.factorize(df_clean[TARGET_COL])[0]\n\n",
                "# Candidate predictive features\n",
                "feature_cols = [c for c in df_clean.columns if c != TARGET_COL]\n",
                "print(f'Active feature count: {len(feature_cols)}')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                f"### 4. Implementation of Selected Strategy: `{strategy_id}`\n",
                "Transform features with strict within-fold isolation to prevent data leakage.\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "def transform_fold(train_df, val_df, strategy):\n",
                "    cols = [c for c in train_df.columns if c != TARGET_COL]\n",
                "    if strategy in ['DROP_FEATURE', 'REMOVE_SUSPECT_FEATURE']:\n",
                "        cols = [c for c in cols if c not in AFFECTED_COLS]\n",
                "    elif strategy == 'PRUNE_CORRELATED_GROUP' and len(AFFECTED_COLS) > 1:\n",
                "        cols = [c for c in cols if c not in AFFECTED_COLS[1:]]\n",
                "    \n",
                "    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(train_df[c])]\n",
                "    cat_cols = [c for c in cols if c not in num_cols]\n",
                "    \n",
                "    train_parts, val_parts = [], []\n",
                "    if num_cols:\n",
                "        tr_num = train_df[num_cols].copy()\n",
                "        v_num = val_df[num_cols].copy()\n",
                "        if strategy == 'WINSORIZE_OUTLIERS':\n",
                "            for c in AFFECTED_COLS:\n",
                "                if c in num_cols:\n",
                "                    low = float(tr_num[c].quantile(0.01))\n",
                "                    high = float(tr_num[c].quantile(0.99))\n",
                "                    tr_num[c] = tr_num[c].clip(lower=low, upper=high)\n",
                "                    v_num[c] = v_num[c].clip(lower=low, upper=high)\n",
                "        imputer = SimpleImputer(strategy='median')\n",
                "        tr_imp = imputer.fit_transform(tr_num)\n",
                "        v_imp = imputer.transform(v_num)\n",
                "        scaler = StandardScaler()\n",
                "        train_parts.append(scaler.fit_transform(tr_imp))\n",
                "        val_parts.append(scaler.transform(v_imp))\n",
                "        \n",
                "        if strategy == 'MISSING_INDICATOR':\n",
                "            for c in AFFECTED_COLS:\n",
                "                if c in cols:\n",
                "                    train_parts.append(train_df[c].isna().astype(float).values.reshape(-1, 1))\n",
                "                    val_parts.append(val_df[c].isna().astype(float).values.reshape(-1, 1))\n",
                "                    \n",
                "    if cat_cols:\n",
                "        tr_cat = train_df[cat_cols].fillna('MISSING').astype(str)\n",
                "        v_cat = val_df[cat_cols].fillna('MISSING').astype(str)\n",
                "        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)\n",
                "        train_parts.append(encoder.fit_transform(tr_cat))\n",
                "        val_parts.append(encoder.transform(v_cat))\n",
                "        \n",
                "    return np.hstack(train_parts), np.hstack(val_parts)\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Controlled 5-Fold Cross-Validation Evaluation (Zero Data Leakage)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                f"cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED) if {is_classification} else KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)\n",
                "oof_preds = np.zeros(len(y))\n\n",
                "for fold, (train_idx, val_idx) in enumerate(cv.split(df_clean, y)):\n",
                "    train_fold = df_clean.iloc[train_idx]\n",
                "    val_fold = df_clean.iloc[val_idx]\n",
                f"    X_train, X_val = transform_fold(train_fold, val_fold, '{strategy_id}')\n",
                f"    clf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=RANDOM_SEED, n_jobs=-1) if {is_classification} else RandomForestRegressor(n_estimators=50, max_depth=6, random_state=RANDOM_SEED, n_jobs=-1)\n",
                "    clf.fit(X_train, y[train_idx])\n",
                "    oof_preds[val_idx] = clf.predict(X_val)\n\n",
                f"if {is_classification}:\n",
                "    acc = accuracy_score(y, oof_preds)\n",
                "    f1 = f1_score(y, oof_preds, average='weighted', zero_division=0)\n",
                "    print(f'Validation Accuracy: {acc:.4f}')\n",
                "    print(f'Validation Weighted F1: {f1:.4f}')\n",
                "else:\n",
                "    r2 = r2_score(y, oof_preds)\n",
                "    mae = mean_absolute_error(y, oof_preds)\n",
                "    print(f'Validation R2 Score: {r2:.4f}')\n",
                "    print(f'Validation MAE: {mae:.4f}')\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Summary & Audit Trail\n",
                f"- **Decision Candidate**: `{candidate.decision_id}`\n",
                f"- **Strategy Executed**: `{strategy_id}`\n",
                f"- **Status**: `{candidate.status.value}`\n",
                f"- **System Interpretation**: {candidate.system_interpretation or 'Evaluated via controlled A/B comparison.'}\n"
            ]
        }
    ]

    notebook_dict = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.11.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    return json.dumps(notebook_dict, indent=2)


def generate_pipeline_script(
    candidate: DecisionCandidate,
    report: AuditReport,
    selected_strategy_id: Optional[str] = None
) -> str:
    """Generate a clean standalone Python script defining the preprocessing pipeline."""
    strategy_id = selected_strategy_id or (
        candidate.user_decision.selected_strategy_id if candidate.user_decision else "MEDIAN_IMPUTATION"
    )
    target_col = report.dataset_profile.detected_target or "target"
    affected_cols = candidate.affected_columns

    script = f'''"""
Reproducible Preprocessing Pipeline for Decision {candidate.decision_id}
Dataset: {report.dataset_name}
Target: {target_col}
Selected Strategy: {strategy_id}
Generated: {report.created_at}
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

TARGET_COL = "{target_col}"
AFFECTED_COLS = {json.dumps(affected_cols)}
SELECTED_STRATEGY = "{strategy_id}"


class DecisionPreprocessor(BaseEstimator, TransformerMixin):
    """Applies the validated '{strategy_id}' strategy."""

    def __init__(self, strategy="{strategy_id}"):
        self.strategy = strategy
        self.num_cols = []
        self.cat_cols = []
        self.num_imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        cols = [c for c in df.columns if c != TARGET_COL]

        if self.strategy in ["DROP_FEATURE", "REMOVE_SUSPECT_FEATURE"]:
            cols = [c for c in cols if c not in AFFECTED_COLS]

        self.num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
        self.cat_cols = [c for c in cols if c not in self.num_cols]

        if self.num_cols:
            imputed = self.num_imputer.fit_transform(df[self.num_cols])
            self.scaler.fit(imputed)

        if self.cat_cols:
            cat_df = df[self.cat_cols].fillna("MISSING").astype(str)
            self.encoder.fit(cat_df)

        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        parts = []

        if self.num_cols:
            num_arr = self.num_imputer.transform(df[self.num_cols])
            num_scaled = self.scaler.transform(num_arr)
            parts.append(num_scaled)

            if self.strategy == "MISSING_INDICATOR":
                for col in AFFECTED_COLS:
                    if col in df.columns:
                        ind = df[col].isna().astype(float).values.reshape(-1, 1)
                        parts.append(ind)

        if self.cat_cols:
            cat_df = df[self.cat_cols].fillna("MISSING").astype(str)
            cat_enc = self.encoder.transform(cat_df)
            parts.append(cat_enc)

        return np.hstack(parts) if parts else np.empty((len(df), 0))


def build_pipeline():
    """Build and return reproducible scikit-learn pipeline."""
    return Pipeline([
        ("preprocessor", DecisionPreprocessor(strategy="{strategy_id}"))
    ])


if __name__ == "__main__":
    print(f"Pipeline ready for {report.dataset_name} with strategy '{strategy_id}'.")
'''
    return script


def generate_decision_html(
    candidate: DecisionCandidate,
    report: AuditReport
) -> str:
    """Generate a clean standalone HTML Decision Summary for stakeholders."""
    strategy_id = candidate.user_decision.selected_strategy_id if candidate.user_decision else "Pending Decision"
    rationale = candidate.user_decision.decision_rationale if candidate.user_decision else "Under Investigation"

    variants_html = ""
    if candidate.experiment_results:
        variants_html += """
        <table style="width:100%; border-collapse:collapse; margin-top:16px;">
            <thead>
                <tr style="background:#f1f5f9; text-align:left;">
                    <th style="padding:8px 12px; border:1px solid #e2e8f0;">Strategy</th>
                    <th style="padding:8px 12px; border:1px solid #e2e8f0;">Score</th>
                    <th style="padding:8px 12px; border:1px solid #e2e8f0;">Δ from Baseline</th>
                    <th style="padding:8px 12px; border:1px solid #e2e8f0;">Complexity</th>
                    <th style="padding:8px 12px; border:1px solid #e2e8f0;">Significance</th>
                </tr>
            </thead>
            <tbody>
        """
        for v in candidate.experiment_results:
            is_sel = v.strategy_id == strategy_id
            bg = "#ecfdf5" if is_sel else "#ffffff"
            variants_html += f"""
                <tr style="background:{bg};">
                    <td style="padding:8px 12px; border:1px solid #e2e8f0;"><strong>{v.strategy_name}</strong>{' (Selected)' if is_sel else ''}</td>
                    <td style="padding:8px 12px; border:1px solid #e2e8f0;">{v.primary_score:.4f} ({v.primary_metric_name})</td>
                    <td style="padding:8px 12px; border:1px solid #e2e8f0;">{v.delta_from_baseline:+.4f}</td>
                    <td style="padding:8px 12px; border:1px solid #e2e8f0;">{v.pipeline_complexity}</td>
                    <td style="padding:8px 12px; border:1px solid #e2e8f0;">{v.practical_importance}</td>
                </tr>
            """
        variants_html += "</tbody></table>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Decision Report: {candidate.decision_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background:#f8fafc; color:#0f172a; padding:32px; max-width:860px; margin:0 auto; line-height:1.6; }}
        .card {{ background:#ffffff; border:1px solid #e2e8f0; border-radius:8px; padding:24px; margin-bottom:20px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }}
        .badge {{ display:inline-block; padding:4px 8px; border-radius:4px; font-size:12px; font-weight:600; background:#e0f2fe; color:#0369a1; }}
        h1, h2, h3 {{ color:#0f172a; margin-top:0; }}
        .meta {{ color:#64748b; font-size:14px; margin-bottom:16px; }}
    </style>
</head>
<body>
    <div class="card">
        <span class="badge">{candidate.decision_id}</span>
        <h1>{candidate.question}</h1>
        <div class="meta">
            Dataset: <strong>{report.dataset_name}</strong> | Target: <strong>{report.dataset_profile.detected_target or 'None'}</strong> | Generated: <strong>{report.created_at}</strong>
        </div>
        <p><strong>Rationale:</strong> {candidate.rationale}</p>
        <p><strong>System Interpretation:</strong> {candidate.system_interpretation or 'Controlled evaluation completed.'}</p>
    </div>

    <div class="card">
        <h2>Controlled Strategy Comparison</h2>
        {variants_html}
    </div>

    <div class="card">
        <h2>Human Decision Record</h2>
        <p><strong>Selected Strategy:</strong> <code>{strategy_id}</code></p>
        <p><strong>Decision Rationale:</strong> {rationale}</p>
        {f'<p><strong>Custom Notes:</strong> {candidate.user_decision.custom_method_notes}</p>' if candidate.user_decision and candidate.user_decision.custom_method_notes else ''}
    </div>
</body>
</html>
"""
    return html

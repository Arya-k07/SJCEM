"""
Baseline Oracle Modeling Engine: builds reproducible, controlled classifiers and regressors
to establish benchmark metrics and out-of-fold validation predictions.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from dataset_autopilot.config import ColumnRole
from dataset_autopilot.schemas import DatasetProfile
from dataset_autopilot.utils import safe_float


def prepare_features_and_target(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_col: str,
    exclude_cols: Optional[List[str]] = None,
    impute_strategy: str = "median",
    winsorize: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str], bool]:
    """
    Clean and transform dataframe features into a numerical matrix.
    Returns (X_matrix, y_vector, feature_names, is_classification).
    """
    exclude_set = set(exclude_cols or [])
    exclude_set.add(target_col)

    # Exclude ID candidate columns and constant columns
    for col, p in profile.columns_profile.items():
        if p.is_id or p.is_constant:
            exclude_set.add(col)

    feature_cols = [c for c in df.columns if c not in exclude_set]
    if not feature_cols:
        raise ValueError("No valid predictive features remaining after exclusion.")

    # Target preparation
    target_series = df[target_col].copy()
    valid_mask = target_series.notna()

    df_clean = df.loc[valid_mask].copy()
    y_raw = target_series.loc[valid_mask]

    is_classification = profile.target_type in ["binary_classification", "multiclass_classification"]

    if is_classification:
        y, _ = pd.factorize(y_raw)
    else:
        y = pd.to_numeric(y_raw, errors="coerce").fillna(y_raw.median() if len(y_raw) > 0 else 0).values

    # Feature processing
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df_clean[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]

    processed_parts = []
    processed_names = []

    # 1. Numeric columns
    if num_cols:
        num_df = df_clean[num_cols].copy()
        
        if winsorize:
            # Clip between 1st and 99th percentiles
            for c in num_cols:
                low = num_df[c].quantile(0.01)
                high = num_df[c].quantile(0.99)
                num_df[c] = num_df[c].clip(lower=low, upper=high)

        # Imputation
        num_imputer = SimpleImputer(strategy=impute_strategy if impute_strategy in ["mean", "median"] else "median")
        num_arr = num_imputer.fit_transform(num_df)

        scaler = StandardScaler()
        num_arr_scaled = scaler.fit_transform(num_arr)

        processed_parts.append(num_arr_scaled)
        processed_names.extend(num_cols)

    # 2. Categorical columns
    if cat_cols:
        cat_df = df_clean[cat_cols].fillna("MISSING").astype(str)
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_arr = encoder.fit_transform(cat_df)

        processed_parts.append(cat_arr)
        processed_names.extend(cat_cols)

    if not processed_parts:
        raise ValueError("No features available for modeling.")

    X = np.hstack(processed_parts)
    return X, y, processed_names, is_classification


def train_and_evaluate_baseline(
    df: pd.DataFrame,
    profile: DatasetProfile,
    target_col: str,
    exclude_cols: Optional[List[str]] = None,
    impute_strategy: str = "median",
    winsorize: bool = False,
    n_folds: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Train a baseline Random Forest model via Cross-Validation and return comprehensive performance metrics.
    """
    X, y, feature_names, is_classification = prepare_features_and_target(
        df=df,
        profile=profile,
        target_col=target_col,
        exclude_cols=exclude_cols,
        impute_strategy=impute_strategy,
        winsorize=winsorize
    )

    n_samples = len(y)
    n_splits = min(n_folds, max(2, n_samples // 10))

    oof_preds = np.zeros(n_samples)
    oof_probs = np.zeros(n_samples) if is_classification else None

    if is_classification:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    else:
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_idx, val_idx in cv.split(X, y if is_classification else None):
        X_train, y_train = X[train_idx], y[train_idx]
        X_val = X[val_idx]

        if is_classification:
            clf = RandomForestClassifier(
                n_estimators=50,
                max_depth=6,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
            clf.fit(X_train, y_train)
            val_preds = clf.predict(X_val)
            oof_preds[val_idx] = val_preds

            if len(np.unique(y)) == 2:
                try:
                    oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]
                except Exception:
                    oof_probs[val_idx] = val_preds
        else:
            reg = RandomForestRegressor(
                n_estimators=50,
                max_depth=6,
                min_samples_leaf=2,
                random_state=random_state,
                n_jobs=-1
            )
            reg.fit(X_train, y_train)
            oof_preds[val_idx] = reg.predict(X_val)

    # Compute aggregate evaluation metrics
    metrics: Dict[str, Any] = {}

    if is_classification:
        metrics["accuracy"] = safe_float(accuracy_score(y, oof_preds))
        metrics["f1_weighted"] = safe_float(f1_score(y, oof_preds, average="weighted", zero_division=0))
        metrics["precision_weighted"] = safe_float(precision_score(y, oof_preds, average="weighted", zero_division=0))
        metrics["recall_weighted"] = safe_float(recall_score(y, oof_preds, average="weighted", zero_division=0))

        if len(np.unique(y)) == 2 and oof_probs is not None:
            try:
                metrics["roc_auc"] = safe_float(roc_auc_score(y, oof_probs))
            except Exception:
                metrics["roc_auc"] = None
        else:
            metrics["roc_auc"] = None

        primary_metric_name = "ROC-AUC" if metrics["roc_auc"] is not None else "F1-Score"
        primary_score = metrics["roc_auc"] if metrics["roc_auc"] is not None else metrics["f1_weighted"]
    else:
        rmse = float(np.sqrt(mean_squared_error(y, oof_preds)))
        mae = float(mean_absolute_error(y, oof_preds))
        r2 = float(r2_score(y, oof_preds))

        metrics["rmse"] = safe_float(rmse)
        metrics["mae"] = safe_float(mae)
        metrics["r2"] = safe_float(r2)

        primary_metric_name = "R² Score"
        primary_score = safe_float(r2)

    return {
        "model_name": "RandomForest (CV=5)",
        "is_classification": is_classification,
        "primary_metric_name": primary_metric_name,
        "primary_score": primary_score,
        "metrics": metrics,
        "y_true": y,
        "y_pred": oof_preds,
        "y_prob": oof_probs,
        "feature_count": len(feature_names),
        "sample_count": n_samples
    }

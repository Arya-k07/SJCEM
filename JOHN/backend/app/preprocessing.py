"""Schema-flexible preprocessing for feedback datasets."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
from pathlib import Path
import re
import unicodedata
from typing import Any

import numpy as np
import pandas as pd


TEXT_ALIASES = (
    "feedback_text",
    "feedbacktext",
    "review_text",
    "comment_text",
    "response_text",
    "feedback",
    "review",
    "comment",
    "response",
    "message",
    "description",
    "text",
)
DATE_ALIASES = (
    "created_at",
    "submitted_at",
    "timestamp",
    "date",
    "datetime",
    "time",
)
CATEGORY_ALIASES = (
    "category",
    "department",
    "type",
    "group",
    "class",
    "faculty",
    "course",
    "program",
)
NUMERIC_HINTS = (
    "rating",
    "score",
    "count",
    "number",
    "amount",
    "age",
    "rank",
)
PLACEHOLDER_TEXT = {"n/a", "na", "n.a.", "nan", "null", "none", "nil", "-", "--"}


class DatasetInputError(ValueError):
    """Raised when a dataset cannot be processed safely."""


def _normalized_name(value: Any) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _find_column(columns: list[Any], aliases: tuple[str, ...]) -> Any | None:
    normalized = {_normalized_name(column): column for column in columns}
    for alias in aliases:
        if alias in normalized:
            return normalized[alias]

    for column in columns:
        name = _normalized_name(column)
        if any(name.endswith(f"_{alias}") for alias in aliases):
            return column
    return None


def _find_candidates(columns: list[Any], aliases: tuple[str, ...]) -> list[Any]:
    alias_order = {alias: position for position, alias in enumerate(aliases)}
    candidates = [
        column
        for column in columns
        if _normalized_name(column) in alias_order
        or any(_normalized_name(column).endswith(f"_{alias}") for alias in aliases)
    ]
    return sorted(
        candidates,
        key=lambda column: (
            alias_order.get(_normalized_name(column), len(aliases)),
            list(columns).index(column),
        ),
    )


def detect_columns(columns: list[Any]) -> dict[str, Any | None]:
    """Detect optional semantic columns without requiring a fixed schema."""
    text_candidates = _find_candidates(columns, TEXT_ALIASES)
    return {
        "text": text_candidates[0] if text_candidates else None,
        "date": _find_column(columns, DATE_ALIASES),
        "category": _find_column(columns, CATEGORY_ALIASES),
        "text_candidates": text_candidates,
    }


def _repair_mojibake(value: str) -> str:
    """Repair common UTF-8-as-Latin-1 damage without changing normal text."""
    markers = ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00f0\u0178", "\u00ef\u00bf\u00bd")
    if not any(marker in value for marker in markers):
        return value
    candidates = []
    for encoding in ("latin1", "cp1252"):
        try:
            candidates.append(value.encode(encoding).decode("utf-8"))
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    if not candidates:
        return value
    repaired = min(candidates, key=lambda candidate: candidate.count("�"))
    return repaired if repaired.count("�") <= value.count("�") else value


def clean_text(value: Any) -> tuple[str, list[str]]:
    """Perform structural text cleanup while retaining NLP signals."""
    warnings: list[str] = []
    if pd.isna(value):
        return "", ["missing_text"]

    text = unicodedata.normalize("NFKC", str(value))
    repaired = _repair_mojibake(text)
    if repaired != text:
        warnings.append("encoding_repaired")
    if "\x00" in repaired:
        warnings.append("null_bytes_removed")
    text = repaired.replace("\x00", "")

    invisible_count = 0
    cleaned_characters: list[str] = []
    for character in text:
        category = unicodedata.category(character)
        if category == "Cc" and character not in {"\n", "\r", "\t"}:
            invisible_count += 1
            continue
        cleaned_characters.append(character)
    if invisible_count:
        warnings.append("invisible_control_characters_removed")

    text = "".join(cleaned_characters)
    normalized_text = re.sub(r"[ \t\r\f\v]+", " ", text)
    normalized_text = re.sub(r"\n[ ]+", "\n", normalized_text)
    normalized_text = normalized_text.strip()
    if normalized_text != text:
        warnings.append("whitespace_normalized")
    text = normalized_text
    if not text:
        warnings.append("empty_text")
    return text, warnings


def _is_missing_feedback(text: str) -> bool:
    return text.casefold().strip() in PLACEHOLDER_TEXT


def _normalize_category(value: Any) -> Any:
    if pd.isna(value):
        return pd.NA
    normalized = re.sub(r"\s+", " ", str(value)).strip()
    return normalized if normalized else pd.NA


def _normalize_numeric_series(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.replace(",", "", regex=False)
    values = values.str.replace(r"^\s*([$%])|([$%])\s*$", "", regex=True)
    return pd.to_numeric(values, errors="coerce")


def _normalize_date_series(series: pd.Series) -> pd.Series:
    """Parse common day-first and ISO dates without warning on valid values."""
    values = series.astype("string").str.strip()
    normalized = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns, UTC]")
    day_first = values.str.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{4}$", na=False)
    if day_first.any():
        normalized.loc[day_first] = pd.to_datetime(
            values.loc[day_first].str.replace("/", "-", regex=False),
            format="%d-%m-%Y",
            errors="coerce",
            utc=True,
        )
    remaining = ~day_first & values.notna() & values.ne("")
    if remaining.any():
        normalized.loc[remaining] = pd.to_datetime(
            values.loc[remaining], errors="coerce", utc=True, format="mixed"
        )
    return normalized


def _duplicate_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Flag exact and explainable punctuation-insensitive duplicate candidates."""
    texts = data["cleaned_text"].fillna("").astype("string")
    exact_keys = texts.str.casefold()
    near_keys = exact_keys.str.replace(r"[^\w]+", "", regex=True)
    duplicate = pd.DataFrame(False, index=data.index, columns=["is_duplicate"])
    duplicate["duplicate_group_id"] = pd.Series(pd.NA, index=data.index, dtype="string")
    duplicate["duplicate_type"] = pd.Series(pd.NA, index=data.index, dtype="string")
    duplicate["duplicate_similarity"] = pd.Series(pd.NA, index=data.index, dtype="Float64")

    group_number = 1
    for _, indexes in exact_keys[exact_keys.ne("")].groupby(exact_keys[exact_keys.ne("")]).groups.items():
        if len(indexes) < 2:
            continue
        group_id = f"duplicate-{group_number}"
        group_number += 1
        duplicate.loc[indexes, "is_duplicate"] = True
        duplicate.loc[indexes, "duplicate_group_id"] = group_id
        duplicate.loc[indexes, "duplicate_type"] = "exact"
        duplicate.loc[indexes, "duplicate_similarity"] = 1.0

    for _, indexes in near_keys[near_keys.ne("")].groupby(near_keys[near_keys.ne("")]).groups.items():
        available = [index for index in indexes if pd.isna(duplicate.loc[index, "duplicate_type"])]
        if len(available) < 2:
            continue
        group_id = f"duplicate-{group_number}"
        group_number += 1
        for index in available:
            similarity = max(
                SequenceMatcher(None, texts.loc[index].casefold(), texts.loc[other].casefold()).ratio()
                for other in available
                if other != index
            )
            duplicate.loc[index, "is_duplicate"] = True
            duplicate.loc[index, "duplicate_group_id"] = group_id
            duplicate.loc[index, "duplicate_type"] = "near_duplicate"
            duplicate.loc[index, "duplicate_similarity"] = round(similarity, 4)
    return duplicate


def _read_dataframe(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.is_file():
        raise DatasetInputError(f"Dataset file does not exist: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        try:
            return pd.read_csv(
                path,
                encoding="utf-8-sig",
                keep_default_na=False,
                engine="python",
                skip_blank_lines=False,
            )
        except UnicodeDecodeError:
            return pd.read_csv(
                path,
                encoding="cp1252",
                keep_default_na=False,
                engine="python",
                skip_blank_lines=False,
            )
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path)
    raise DatasetInputError("Supported dataset formats are CSV, XLSX, and XLSM.")


def clean_dataset(file_path: str | Path) -> dict[str, Any]:
    """Load and clean a dataset while retaining rejected rows and diagnostics."""
    raw = _read_dataframe(file_path)
    if raw.empty and len(raw.columns) == 0:
        raise DatasetInputError("The dataset has no columns.")

    mapping = detect_columns(list(raw.columns))
    text_column = mapping["text"]
    if text_column is None:
        raise DatasetInputError(
            "No feedback/text column was detected. Expected a column such as feedback, review, comment, response, or text."
        )

    data = raw.copy()
    data.insert(len(data.columns), "original_text", data[text_column])
    cleaned_values = data[text_column].map(clean_text)
    data["cleaned_text"] = cleaned_values.map(lambda result: result[0])
    row_warnings = cleaned_values.map(lambda result: result[1])
    row_warnings = pd.Series(
        [
            [*warnings, "missing_text"]
            if _is_missing_feedback(text) and "missing_text" not in warnings
            else warnings
            for warnings, text in zip(row_warnings, data["cleaned_text"], strict=True)
        ],
        index=data.index,
    )

    if mapping["date"] is not None:
        data["normalized_date"] = _normalize_date_series(data[mapping["date"]])
        raw_dates = data[mapping["date"]].astype("string").str.strip()
        missing_dates = data[mapping["date"]].isna() | raw_dates.isna() | raw_dates.eq("")
        invalid_dates = ~missing_dates & data["normalized_date"].isna()
        row_warnings = row_warnings.copy()
        row_warnings.loc[invalid_dates] = row_warnings.loc[invalid_dates].map(
            lambda warnings: [*warnings, "invalid_date"]
        )
        row_warnings.loc[missing_dates] = row_warnings.loc[missing_dates].map(
            lambda warnings: [*warnings, "missing_optional_date"]
        )

    if mapping["category"] is not None:
        data["normalized_category"] = data[mapping["category"]].map(_normalize_category)

    numeric_columns = [
        column
        for column in data.columns
        if any(hint in _normalized_name(column) for hint in NUMERIC_HINTS)
        and column not in {text_column, mapping["date"], mapping["category"]}
    ]
    for column in numeric_columns:
        normalized = _normalize_numeric_series(data[column])
        invalid_numbers = data[column].notna() & normalized.isna()
        row_warnings.loc[invalid_numbers] = row_warnings.loc[invalid_numbers].map(
            lambda warnings: [*warnings, f"invalid_numeric:{column}"]
        )
        data[f"normalized_{_normalized_name(column)}"] = normalized

    data["row_number"] = np.arange(2, len(data) + 2)
    duplicate_metadata = _duplicate_columns(data)
    data = pd.concat([data, duplicate_metadata], axis=1)
    row_warnings = row_warnings.copy()
    row_warnings.loc[data["duplicate_type"].eq("exact")] = row_warnings.loc[
        data["duplicate_type"].eq("exact")
    ].map(lambda warnings: [*warnings, "exact_duplicate"])
    row_warnings.loc[data["duplicate_type"].eq("near_duplicate")] = row_warnings.loc[
        data["duplicate_type"].eq("near_duplicate")
    ].map(lambda warnings: [*warnings, "near_duplicate"])

    data["quality_warnings"] = row_warnings.map(lambda warnings: ";".join(dict.fromkeys(warnings)))
    data["quality_status"] = np.where(
        data["cleaned_text"].eq("") | data["cleaned_text"].map(_is_missing_feedback),
        "rejected",
        np.where(data["quality_warnings"].ne(""), "accepted_with_warning", "accepted"),
    )

    rejected = data[data["quality_status"] == "rejected"].copy()
    clean = data[data["quality_status"] != "rejected"].copy()
    input_columns = list(raw.columns)
    if mapping["date"] is not None:
        clean[mapping["date"]] = clean["normalized_date"].dt.strftime("%Y-%m-%d")
    clean = clean[input_columns + ["cleaned_text"]]
    warning_text = data["quality_warnings"]
    report = {
        "rows_received": int(len(raw)),
        "rows_accepted": int((data["quality_status"] != "rejected").sum()),
        "rows_accepted_with_warning": int((data["quality_status"] == "accepted_with_warning").sum()),
        "rows_rejected": int(len(rejected)),
        "rows_with_warnings": int(warning_text.ne("").sum()),
        "duplicate_rows_flagged": int(data["is_duplicate"].sum()),
        "exact_duplicates": int(data["duplicate_type"].eq("exact").sum()),
        "near_duplicates": int(data["duplicate_type"].eq("near_duplicate").sum()),
        "invalid_date_rows": int(warning_text.str.contains("invalid_date", regex=False).sum()),
        "encoding_repairs": int(warning_text.str.contains("encoding_repaired", regex=False).sum()),
        "null_bytes_removed": int(warning_text.str.contains("null_bytes_removed", regex=False).sum()),
        "control_characters_removed": int(warning_text.str.contains("control_characters_removed", regex=False).sum()),
        "whitespace_normalizations": int(warning_text.str.contains("whitespace_normalized", regex=False).sum()),
        "missing_text_rows": int(warning_text.str.contains("missing_text|empty_text", regex=True).sum()),
        "duplicates_detected": int(data["is_duplicate"].sum()),
        "cleaning_operations": {
            "encoding_repairs": int(warning_text.str.contains("encoding_repaired", regex=False).sum()),
            "null_bytes_removed": int(warning_text.str.contains("null_bytes_removed", regex=False).sum()),
            "control_characters_removed": int(warning_text.str.contains("invisible_control_characters_removed", regex=False).sum()),
            "whitespace_repairs": int(warning_text.str.contains("whitespace_normalized", regex=False).sum()),
        },
        "detected_columns": {key: value for key, value in mapping.items() if key != "text_candidates" and value is not None},
        "selected_text_column": text_column,
        "text_candidates": mapping["text_candidates"],
        "text_signals_preserved": True,
    }
    return {
        "clean_data": clean,
        "rejected_data": rejected,
        "report": report,
        "column_mapping": mapping,
    }


def export_cleaned_dataset(cleaned_data: pd.DataFrame, output_path: str | Path) -> Path:
    """Write cleaned rows to a UTF-8 CSV without adding an index column."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(destination, index=False, encoding="utf-8")
    return destination


def export_rejected_dataset(rejected_data: pd.DataFrame, output_path: str | Path) -> Path:
    """Write rejected rows to a UTF-8 CSV, including an empty-file schema."""
    return export_cleaned_dataset(rejected_data, output_path)


def export_preprocessing_report(report: dict[str, Any], output_path: str | Path) -> Path:
    """Write preprocessing diagnostics as readable JSON."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return destination

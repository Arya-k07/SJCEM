from pathlib import Path
import shutil

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.preprocessing import (
    DatasetInputError,
    clean_dataset,
    clean_text,
    export_cleaned_dataset,
    export_rejected_dataset,
)


@pytest.fixture
def generated_csv(tmp_path: Path) -> Path:
    dataset = pd.DataFrame(
        {
            "Feedback Text": [
                "  Great service 😊!!!  ",
                "The Wi-Fi is soooo slow :/ ...",
                "The Wi-Fi is soooo slow :/ ...",
                "\x00",
                None,
                "Encoding issue \u00c3\u00a9 repaired?",
            ],
            "created_at": [
                "2026-09-01",
                "2026-09-02",
                "2026-09-02",
                "not-a-date",
                "2026-09-05",
                "2026-09-06",
            ],
            "Department": [" Student Services ", "IT", "IT", "Admin", "Admin", "IT"],
            "rating": [5, "4", "4", 3, 2, "bad"],
        }
    )
    path = tmp_path / "generated.csv"
    dataset.to_csv(path, index=False)
    return path


def test_clean_dataset_preserves_text_signals_and_original(generated_csv: Path) -> None:
    result = clean_dataset(generated_csv)
    clean = result["clean_data"]

    first = clean.loc[clean["cleaned_text"] == "Great service 😊!!!"].squeeze()
    assert first["cleaned_text"] == "Great service 😊!!!"
    assert (clean["cleaned_text"] == "The Wi-Fi is soooo slow :/ ...").any()
    assert "original_text" not in clean.columns
    assert "quality_warnings" not in clean.columns
    assert result["report"]["encoding_repairs"] >= 1
    assert result["report"]["text_signals_preserved"] is True


def test_detection_is_flexible_and_date_is_optional(tmp_path: Path) -> None:
    path = tmp_path / "feedback.csv"
    pd.DataFrame({"review": ["Good"], "group": ["A"]}).to_csv(path, index=False)

    result = clean_dataset(path)
    assert result["column_mapping"]["text"] == "review"
    assert result["column_mapping"]["date"] is None
    assert "normalized_date" not in result["clean_data"]


def test_invalid_rows_are_rejected_and_duplicates_are_flagged(generated_csv: Path) -> None:
    result = clean_dataset(generated_csv)

    assert result["report"]["rows_received"] == 6
    assert result["report"]["rows_rejected"] == 2
    assert result["report"]["duplicate_rows_flagged"] == 2
    assert len(result["clean_data"]) == 4
    assert "is_duplicate" not in result["clean_data"].columns
    assert result["report"]["exact_duplicates"] == 2
    assert result["report"]["invalid_date_rows"] == 1
    assert result["rejected_data"]["quality_warnings"].str.contains("invalid_date", regex=False).any()
    assert result["report"]["rows_accepted_with_warning"] > 0


def test_missing_text_column_fails_clearly(tmp_path: Path) -> None:
    path = tmp_path / "no_text.csv"
    pd.DataFrame({"department": ["IT"], "rating": [5]}).to_csv(path, index=False)

    with pytest.raises(DatasetInputError, match="No feedback/text column"):
        clean_dataset(path)


def test_preprocess_api_can_return_cleaned_data(generated_csv: Path) -> None:
    with TestClient(app) as client:
        with generated_csv.open("rb") as dataset:
            response = client.post(
                "/api/preprocess?include_cleaned_data=true",
                files={"file": ("feedback.csv", dataset, "text/csv")},
            )

    assert response.status_code == 200
    body = response.json()
    assert body["report"]["rows_accepted"] == 4
    assert len(body["cleaned_data"]) == 4
    assert body["cleaned_data"][0]["cleaned_text"] == "Great service 😊!!!"
    assert "quality_warnings" not in body["cleaned_data"][0]
    assert "rejected_data" not in body


def test_preprocess_api_rejects_unsupported_files() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/preprocess",
            files={"file": ("feedback.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 400


def test_root_route_is_available() -> None:
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_xlsx_and_xlsm_inputs_are_supported(tmp_path: Path) -> None:
    dataset = pd.DataFrame({"REVIEW": ["👍", "Good"], "DATE": ["2026-01-01", "2026-01-02"]})
    xlsx_path = tmp_path / "feedback.xlsx"
    xlsm_path = tmp_path / "feedback.xlsm"
    dataset.to_excel(xlsx_path, index=False)
    shutil.copyfile(xlsx_path, xlsm_path)

    assert len(clean_dataset(xlsx_path)["clean_data"]) == 2
    assert len(clean_dataset(xlsm_path)["clean_data"]) == 2


def test_text_candidates_are_deterministic_and_missing_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "candidates.csv"
    pd.DataFrame({"  COMMENT  ": ["first"], "feedback_text": ["second"]}).to_csv(path, index=False)

    result = clean_dataset(path)
    assert result["column_mapping"]["text"] == "feedback_text"
    assert result["report"]["text_candidates"] == ["feedback_text", "  COMMENT  "]
    with pytest.raises(DatasetInputError, match="does not exist"):
        clean_dataset(tmp_path / "missing.csv")


def test_placeholders_reject_but_short_meaningful_feedback_is_kept(tmp_path: Path) -> None:
    path = tmp_path / "missing_values.csv"
    pd.DataFrame({"response": ["N/A", "bad", "👍", "   ", ":("]}).to_csv(path, index=False)

    result = clean_dataset(path)
    assert len(result["clean_data"]) == 3
    assert len(result["rejected_data"]) == 2
    assert result["report"]["missing_text_rows"] == 2


def test_near_duplicates_are_flagged_without_deletion(tmp_path: Path) -> None:
    path = tmp_path / "near_duplicates.csv"
    pd.DataFrame({"feedback": ["WiFi is terrible", "WiFi is terrible!", "WiFi is amazing"]}).to_csv(
        path, index=False
    )

    result = clean_dataset(path)
    assert len(result["clean_data"]) == 3
    assert result["report"]["near_duplicates"] == 2
    assert result["report"]["duplicates_detected"] == 2


def test_report_tracks_cleaning_operations(tmp_path: Path) -> None:
    path = tmp_path / "warnings.csv"
    pd.DataFrame(
        {
            "feedback": ["  Great\x00 service  ", "fine"],
            "date": ["01/02/2026", "not-a-date"],
            "category": ["  Infrastructure ", None],
        }
    ).to_csv(path, index=False)

    result = clean_dataset(path)
    report = result["report"]
    assert report["rows_accepted_with_warning"] == 2
    assert report["encoding_repairs"] == 0
    assert report["null_bytes_removed"] == 1
    assert report["control_characters_removed"] == 0
    assert report["whitespace_normalizations"] == 1
    assert report["invalid_date_rows"] == 1
    assert "category" in result["clean_data"]
    assert result["clean_data"].loc[0, "category"] == "  Infrastructure "


def test_invisible_control_characters_are_removed_without_nlp_normalization() -> None:
    cleaned, warnings = clean_text("Hinglish अच्छा\x01 feedback!!! 😭 :/")

    assert cleaned == "Hinglish अच्छा feedback!!! 😭 :/"
    assert "invisible_control_characters_removed" in warnings


def test_exported_cleaned_csv_round_trips_unicode(tmp_path: Path, generated_csv: Path) -> None:
    result = clean_dataset(generated_csv)
    output_path = export_cleaned_dataset(result["clean_data"], tmp_path / "nested" / "cleaned_dataset.csv")

    exported = pd.read_csv(output_path, keep_default_na=False)
    assert output_path.exists()
    assert len(exported) == len(result["clean_data"])
    assert "😊" in exported.loc[0, "cleaned_text"]
    assert "original_text" not in exported.columns
    assert set(exported.columns) == {"Feedback Text", "created_at", "Department", "rating", "cleaned_text"}


def test_rejected_export_supports_rows_and_empty_data(tmp_path: Path, generated_csv: Path) -> None:
    result = clean_dataset(generated_csv)
    rejected_path = export_rejected_dataset(result["rejected_data"], tmp_path / "rejected_dataset.csv")

    assert rejected_path.exists()
    assert len(pd.read_csv(rejected_path, keep_default_na=False)) == len(result["rejected_data"])

    empty_path = export_rejected_dataset(result["clean_data"].iloc[0:0], tmp_path / "empty" / "rejected.csv")
    assert empty_path.exists()
    assert list(pd.read_csv(empty_path).columns) == list(result["clean_data"].columns)


def test_api_export_disabled_does_not_create_files(tmp_path: Path, generated_csv: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with TestClient(app) as client:
        with generated_csv.open("rb") as dataset:
            response = client.post(
                "/api/preprocess",
                files={"file": ("feedback.csv", dataset, "text/csv")},
            )

    assert response.status_code == 200
    assert "export" not in response.json()
    assert not (tmp_path / "outputs").exists()


def test_api_export_enabled_writes_both_files(tmp_path: Path, generated_csv: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with TestClient(app) as client:
        with generated_csv.open("rb") as dataset:
            response = client.post(
                "/api/preprocess?export_cleaned_data=true",
                files={"file": ("feedback.csv", dataset, "text/csv")},
            )

    assert response.status_code == 200
    export = response.json()["export"]
    cleaned_path = tmp_path / export["cleaned_file"]
    rejected_path = tmp_path / export["rejected_file"]
    assert cleaned_path.exists()
    assert rejected_path.exists()
    report_path = tmp_path / export["report_file"]
    assert report_path.exists()
    assert len(pd.read_csv(cleaned_path, keep_default_na=False)) == 4
    assert len(pd.read_csv(rejected_path, keep_default_na=False)) == 2

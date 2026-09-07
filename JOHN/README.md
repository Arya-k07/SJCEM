# FeedbackIQ

FeedbackIQ is a backend foundation for turning messy feedback datasets into clean, AI-ready data and structured NLP analysis.

Current scope:

```text
CSV / XLSX / XLSM
        |
        v
Preprocessing and validation
        |
        v
cleaned_dataset.csv
        |
        v
NLP analysis
        |
        v
analyzed_dataset.csv
```

The implementation is domain-agnostic. The current college feedback file is only a test dataset; no college-specific topic rules are required by the pipeline.

## Project Structure

```text
FeedbackIQ/
├── README.md
├── BUILD_PLAN (1).md
├── .gitignore
├── backend/
│   ├── requirements.txt
│   ├── app/
│   │   ├── main.py
│   │   ├── preprocessing.py
│   │   └── nlp.py
│   └── tests/
│       ├── test_preprocessing.py
│       ├── test_nlp.py
│       └── datasets/
│           └── college_feedback.csv
└── outputs/
    ├── cleaned_dataset.csv
    ├── analyzed_dataset.csv
    ├── rejected_dataset.csv
    └── preprocessing_report.json
```

Generated output files are ignored by Git.

## Technology Used

### Python

The implementation uses Python 3.x.

### Pandas

Pandas is used for:

- CSV loading
- XLSX/XLSM loading through OpenPyXL
- DataFrame transformation
- Date and numeric normalization
- Duplicate and validation processing
- CSV export

### NumPy

NumPy is used for row numbering and array-oriented data operations during preprocessing.

### OpenPyXL

OpenPyXL provides XLSX/XLSM workbook support for Pandas.

### FastAPI

FastAPI exposes the processing functionality through HTTP endpoints and automatically provides Swagger documentation.

### Uvicorn

Uvicorn runs the FastAPI application locally.

### Scikit-learn

Scikit-learn provides TF-IDF weighting for meaningful phrase and concept extraction in the NLP stage.

### Hugging Face Transformers and PyTorch

The primary sentiment provider uses a lightweight Hugging Face transformer, loaded lazily once per process. PyTorch selects CUDA when available and otherwise uses CPU inference. Unique feedback texts are processed in batches and cached so duplicate records do not trigger repeated inference.

If the transformer or model cannot load, the provider explicitly falls back to TextBlob rather than crashing the pipeline. The provider is abstracted behind `SentimentProvider`.

### TextBlob

TextBlob is the local fallback provider and supplies polarity/subjectivity signals used by contextual sentiment and emotion logic.

### Sentence Transformers

The lightweight `all-MiniLM-L6-v2` sentence-transformer is used optionally after TF-IDF candidate extraction to group semantically similar topic phrases. Embeddings are cached and are never written into the output CSV.

### Pytest

Pytest runs preprocessing, NLP, API, file-format, export, and quality tests.

## Preprocessing

Implementation: [backend/app/preprocessing.py](backend/app/preprocessing.py)

The public interface is:

```python
from app.preprocessing import clean_dataset

result = clean_dataset("backend/tests/datasets/college_feedback.csv")
```

The return value remains:

```python
{
    "clean_data": dataframe,
    "rejected_data": dataframe,
    "report": dictionary,
    "column_mapping": dictionary,
}
```

### Supported Inputs

- CSV
- XLSX
- XLSM

At minimum, a text/feedback column is required. Common recognized names include:

```text
feedback
feedback_text
review
review_text
comment
comment_text
response
response_text
message
description
text
```

Optional date columns include:

```text
date
created_at
submitted_at
timestamp
datetime
time
```

A `source` column is not required. If legitimate input metadata exists, it is preserved in the AI-ready data when appropriate.

### Structural Cleaning

The preprocessing layer conservatively fixes structural problems:

- Leading and trailing whitespace
- Excessive internal whitespace
- Null bytes
- Invisible control characters
- Confidently repairable mojibake
- Safe Unicode handling
- Optional date normalization
- Basic numeric normalization
- Basic category normalization
- Missing and placeholder feedback values
- Exact duplicate detection
- Near-duplicate detection

It does not perform traditional NLP normalization.

The following signals are preserved:

- Emojis
- Emoticons such as `:)`, `:(`, `:/`, and `:D`
- Expressive punctuation such as `?!`, `!!!`, and `...`
- Capitalization
- Repeated letters such as `soooo`
- Slang
- Hinglish and mixed-language feedback
- Short meaningful feedback such as `bad`, `okay`, `:(`, or `👍`

The system does not lowercase, stem, lemmatize, remove stopwords, translate, paraphrase, summarize, or rewrite feedback.

### Cleaned Data Contract

The main `clean_data` DataFrame is compact and intended for the NLP stage. It contains input data columns plus:

```text
cleaned_text
```

Internal diagnostic fields such as duplicate IDs, warning strings, row numbers, and quality status are not placed in the main AI-ready cleaned dataset.

### Rejected Data

Rows are rejected only when the feedback is genuinely unusable, such as:

- Null feedback
- Empty feedback
- Whitespace-only feedback
- Placeholder-only feedback such as `N/A`, `null`, or `-`

Invalid optional dates normally create diagnostics rather than rejecting an otherwise usable feedback row.

### Dates

Dates are normalized to `YYYY-MM-DD`. The current dataset uses day-first values such as `01-06-2026`, which become:

```text
2026-06-01
```

Invalid dates are reported without causing valid feedback to be discarded.

### Duplicates

Duplicates are detected but not deleted. This preserves repeated feedback because repeated reports can indicate issue frequency.

The diagnostics distinguish:

```text
exact
near_duplicate
```

## NLP Analysis

Implementation: [backend/app/nlp.py](backend/app/nlp.py)

The public interface is:

```python
from app.nlp import analyze_dataset

analyzed_data = analyze_dataset(result["clean_data"])
```

The NLP stage reads `cleaned_text` and enriches the data without changing the original feedback.

### NLP Output Fields

```text
sentiment
sentiment_score
sentiment_confidence
topics
keywords
emotion
aspect_sentiments
```

### Sentiment

The controlled sentiment vocabulary is:

```text
Positive
Negative
Neutral
Mixed
```

The primary local provider combines transformer predictions with the existing contextual structural layer for:

- Negation
- Failures and malfunctions
- Shortages
- Excessive values
- Delays
- Missing or outdated items
- Confusion
- Maintenance needs
- Contrast words such as `but`, `although`, and `however`
- Emojis and emoticons

Examples:

```text
network dies during online tests
-> Negative

not enough healthy options
-> Negative

teaching quality is actually great
-> Positive

the library is great but the Wi-Fi is terrible
-> Mixed
```

`sentiment_score` and `sentiment_confidence` are bounded confidence-style values between `0.0` and `1.0`. `sentiment_confidence` is the explicit downstream-facing field; `sentiment_score` is retained for backward compatibility.

Sentiment meanings:

- `Positive`: approval, satisfaction, or a favorable experience.
- `Negative`: dissatisfaction, malfunction, shortage, delay, poor quality, or an unmet need.
- `Neutral`: factual feedback without a strong positive or negative signal.
- `Mixed`: the same feedback contains materially positive and negative aspects.

The latest transformer run produced no `Neutral` rows in this dataset because every record received either a transformer polarity signal or a contextual complaint/contrast signal. `Neutral` remains a valid output for future datasets.

### Topics

Topics represent what the feedback is about, not generic grammar or sentiment words.

Examples:

```text
washrooms need better maintenance
-> ["Washrooms"]

network dies during online tests
-> ["Online Tests", "Network"]

the library is great but the Wi-Fi is terrible
-> ["Wi-Fi", "Library"]
```

Topic extraction is based on filtered semantic phrase candidates and TF-IDF weighting. It does not use college-specific `if "wifi"` rules.

### Keywords

Keywords are short meaningful concepts or issue terms. Generic filler and grammatical fragments are excluded where possible.

Examples:

```text
washrooms need better maintenance
-> ["washrooms", "maintenance"]

network dies during online tests
-> ["network dies", "online tests"]
```

### Aspect Sentiments

`aspect_sentiments` stores JSON records for entity-level sentiment:

```json
[
  {"aspect": "Library", "sentiment": "Positive"},
  {"aspect": "Wi-Fi", "sentiment": "Negative"}
]
```

This separates the entity from the problem being described and supports multi-aspect feedback.

### Emotion

The current output uses labels such as:

```text
Satisfaction
Joy
Frustration
Anger
Disappointment
Concern
Neutral
```

Emoji and emoticon signals can contribute to emotion detection, but surrounding text remains part of the decision.

### Duplicate Consistency

Analysis is deterministic. Identical cleaned feedback receives the same sentiment, topics, keywords, and aspect interpretation across duplicate rows.

## Export Functions

The preprocessing module provides:

```python
from app.preprocessing import (
    export_cleaned_dataset,
    export_rejected_dataset,
    export_preprocessing_report,
)
```

The NLP module provides:

```python
from app.nlp import export_analyzed_dataset
```

Example:

```python
from pathlib import Path

from app.preprocessing import clean_dataset, export_cleaned_dataset
from app.nlp import analyze_dataset, export_analyzed_dataset

result = clean_dataset("tests/datasets/college_feedback.csv")
export_cleaned_dataset(result["clean_data"], Path("outputs/cleaned_dataset.csv"))
analyzed = analyze_dataset(result["clean_data"])
export_analyzed_dataset(analyzed, "outputs/analyzed_dataset.csv")
```

Exports use UTF-8 and `index=False`, preserving Unicode and emojis.

## FastAPI API

Implementation: [backend/app/main.py](backend/app/main.py)

Start the API:

```powershell
cd D:\JOHN\backend
uvicorn app.main:app --reload
```

Available URLs:

```text
GET /
GET /health
POST /api/preprocess
POST /api/analyze
GET /docs
```

### Preprocessing API

Upload a dataset:

```text
POST /api/preprocess
```

Optional query parameters:

```text
include_cleaned_data=true
include_rejected_data=true
export_cleaned_data=true
```

When `export_cleaned_data=true`, the API writes collision-safe cleaned, rejected, and report files under `outputs/` and returns their paths.

### NLP API

Upload a dataset for preprocessing plus analysis:

```text
POST /api/analyze
```

The response includes:

- Number of analyzed rows
- Analyzed file path
- Analyzed records in the response body

## Testing

Run all tests:

```powershell
cd D:\JOHN\backend
python -m pytest -q
```

The test suite covers:

- CSV, XLSX, and XLSM loading
- Flexible column detection
- Missing files and unsupported formats
- Text preservation
- Unicode, emoji, emoticon, punctuation, slang, and Hinglish handling
- Whitespace and control-character cleanup
- Encoding repair
- Placeholder and short-feedback behavior
- Date normalization and invalid dates
- Exact and near-duplicate detection
- AI-ready output schema
- CSV exports and Unicode round trips
- API export behavior
- Sentiment labels and score bounds
- Mixed sentiment
- Topics and keywords
- Aspect sentiment
- NLP API output

## Current Demo Results

The latest full run on the 1,200-row college dataset produced:

```text
Rows analyzed: 1200
Rows accepted: 1200
Rows rejected: 0
Duplicates detected: 717
Positive: 336
Negative: 792
Neutral: 0
Mixed: 72
Average sentiment confidence: 0.6964
Unique feedback texts: 686
Duplicate groups: 172
Duplicate groups with consistent NLP output: 172
```

The latest analyzed artifact is:

[backend/outputs/analyzed_dataset_transformer.csv](backend/outputs/analyzed_dataset_transformer.csv)

Other generated artifacts may have run suffixes when an output file was open or locked by the editor.

## Current Scope Boundaries

Implemented:

- Data ingestion
- Structural preprocessing
- Validation and duplicate diagnostics
- AI-ready cleaned data
- Local NLP enrichment
- FastAPI preprocessing and analysis endpoints
- CSV and JSON exports

Not implemented yet:

- Supabase/PostgreSQL persistence
- Gemini integration
- LangGraph decision workflows
- Dashboard APIs
- PDF and Excel reporting
- Authentication
- Production deployment
- External LLM batching and provider retries

These belong to later phases of the larger build plan.

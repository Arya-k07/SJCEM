# FeedbackIQ — AI-Powered Feedback Intelligence Backend
## Master Backend Build Specification + API Checklist

> **Purpose:** This is the single source of truth for building the **backend/API-first version** of FeedbackIQ.
>
> **Current scope:** Build the complete backend, database, AI, LangGraph, chatbot, reporting, and real-time API layer. **Do NOT build the React/frontend now.**
>
> **Hackathon constraint:** 8-hour build. Finish the core backend first. Kafka, Redis, and Kubernetes are optional and must NEVER block the working core.

---

# 1. Project Overview

## Problem Statement

Organisations receive large volumes of feedback from reviews, surveys, support tickets, comments, forms, and other sources.

The backend must automatically:

- ingest feedback data
- clean and validate it
- analyse sentiment
- identify topics/tags/themes
- detect trends
- calculate useful analytics
- identify important/critical issues
- generate AI-powered insights
- make structured decisions using LangGraph
- generate recommendations
- provide an AI chatbot API
- generate downloadable PDF/Excel reports
- support real-time updates
- expose clean REST APIs for a frontend to consume later

## Product Story

**Feedback → Intelligence → Decision**

Suggested product name:

**FeedbackIQ**

Suggested tagline:

**Don't read 10,000 feedbacks. Ask them.**

---

# 2. Current Scope

## Build NOW

- [ ] FastAPI backend
- [ ] Supabase/PostgreSQL
- [ ] CSV ingestion
- [ ] XLSX ingestion
- [ ] Data cleaning/validation
- [ ] Sentiment analysis
- [ ] Topic extraction
- [ ] Trend analysis
- [ ] Aggregated analytics
- [ ] Significance/priority detection
- [ ] LangGraph decision engine
- [ ] Gemini integration
- [ ] Gemini API key round-robin manager
- [ ] AI executive insights
- [ ] AI decision recommendations
- [ ] Chatbot API
- [ ] PDF report generation
- [ ] Excel report generation
- [ ] Supabase Realtime integration
- [ ] API documentation through FastAPI Swagger
- [ ] Backend tests

## DO NOT BUILD NOW

- [ ] React
- [ ] Vite
- [ ] Tailwind
- [ ] Recharts
- [ ] Frontend pages
- [ ] Frontend chatbot UI
- [ ] Frontend dashboard

A frontend can be added later using these APIs.

---

# 3. Required Technology Stack

## Backend — MUST

- Python 3.x
- FastAPI
- Uvicorn
- Pydantic
- Pandas
- Async/background processing where useful

## Database — MUST

- Supabase
- PostgreSQL
- Supabase Realtime

> MongoDB is NOT used anywhere in this project.

## AI — MUST

- Gemini API
- Multiple authorized Gemini API keys
- Round-robin key manager
- Temporary unavailable/rate-limit handling

## Decision / Insight Layer — MUST

- LangGraph
- Gemini as the reasoning model inside selected LangGraph nodes

## Reporting — MUST

- ReportLab for PDF
- Pandas/OpenPyXL for Excel

## Optional — ONLY IF CORE IS STABLE

- Kafka
- Redis
- Kubernetes
- Docker improvements

---

# 4. Backend Architecture

```text
                    ┌──────────────────────────┐
                    │     External Sources     │
                    │ CSV / XLSX / REST API    │
                    │ Live Feedback            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │       FASTAPI API        │
                    │                          │
                    │ Upload / Ingestion       │
                    │ Analysis / Dashboard     │
                    │ Chat / Reports           │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   Processing Pipeline    │
                    │                          │
                    │ Validation               │
                    │ Cleaning                 │
                    │ Normalization            │
                    │ Deduplication            │
                    └────────────┬─────────────┘
                                 │
                                 ▼
              ┌─────────────────────────────────────┐
              │             SUPABASE                │
              │          PostgreSQL DB               │
              │                                     │
              │ datasets                             │
              │ feedback                             │
              │ analytics                            │
              │ decisions                            │
              │ reports                              │
              └───────────────┬─────────────────────┘
                              │
                              ▼
                 ┌────────────────────────────┐
                 │      AI ANALYSIS           │
                 │                            │
                 │ Sentiment                  │
                 │ Confidence                 │
                 │ Topics / Tags              │
                 │ Trends                     │
                 └────────────┬───────────────┘
                              │
                              ▼
                 ┌────────────────────────────┐
                 │  SIGNIFICANCE ENGINE       │
                 │                            │
                 │ Mention volume             │
                 │ Negative percentage        │
                 │ Trend growth               │
                 │ Priority score             │
                 └────────────┬───────────────┘
                              │
                    Significant issue?
                         /          \
                       NO            YES
                       │              │
                       │              ▼
                       │    ┌──────────────────────┐
                       │    │      LANGGRAPH       │
                       │    │  Decision Workflow   │
                       │    │                      │
                       │    │ Evidence             │
                       │    │ Significance         │
                       │    │ Severity              │
                       │    │ Impact                │
                       │    │ Recommendation       │
                       │    │ Validation            │
                       │    └──────────┬───────────┘
                       │               │
                       │               ▼
                       │    ┌──────────────────────┐
                       │    │      GEMINI API      │
                       │    │ Round-Robin Manager  │
                       │    └──────────┬───────────┘
                       │               │
                       └───────┬───────┘
                               ▼
                 ┌────────────────────────────┐
                 │   Results / Insights       │
                 │   Decisions / Reports      │
                 └────────────┬───────────────┘
                              │
                 ┌────────────┴──────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐       ┌──────────────────┐
       │ REST API Clients  │       │ Supabase Realtime│
       │ Frontend later    │       │ Live updates     │
       └──────────────────┘       └──────────────────┘
```

---

# 5. Architecture Principle

The core system MUST work without Kafka, Redis, or Kubernetes.

Use abstractions so optional infrastructure can be added later.

Recommended interfaces:

```text
FeedbackQueue
AnalyticsStore
Cache
AIProvider
DecisionEngine
```

Initial implementation:

```text
FeedbackQueue  → FastAPI/background processing
AnalyticsStore → Supabase
Cache          → direct DB / in-memory
AIProvider     → Gemini
DecisionEngine → LangGraph
```

Optional upgrade:

```text
FeedbackQueue  → Kafka
Cache          → Redis
Deployment     → Kubernetes
```

---

# 6. Repository Structure

```text
feedbackiq/
│
├── README.md
├── BUILD_PLAN.md
├── .gitignore
├── .env.example
│
├── backend/
│   ├── requirements.txt
│   ├── main.py
│   │
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes_health.py
│   │   │   ├── routes_datasets.py
│   │   │   ├── routes_upload.py
│   │   │   ├── routes_analysis.py
│   │   │   ├── routes_dashboard.py
│   │   │   ├── routes_insights.py
│   │   │   ├── routes_decisions.py
│   │   │   ├── routes_chat.py
│   │   │   └── routes_reports.py
│   │   │
│   │   ├── services/
│   │   │   ├── ingestion.py
│   │   │   ├── preprocessing.py
│   │   │   ├── sentiment.py
│   │   │   ├── topics.py
│   │   │   ├── analytics.py
│   │   │   ├── significance.py
│   │   │   ├── insights.py
│   │   │   ├── chatbot.py
│   │   │   └── report_generator.py
│   │   │
│   │   ├── ai/
│   │   │   ├── gemini_manager.py
│   │   │   ├── prompts.py
│   │   │   └── schemas.py
│   │   │
│   │   ├── langgraph/
│   │   │   ├── state.py
│   │   │   ├── graph.py
│   │   │   └── nodes.py
│   │   │
│   │   ├── db/
│   │   │   ├── supabase.py
│   │   │   └── queries.py
│   │   │
│   │   ├── models/
│   │   │   └── schemas.py
│   │   │
│   │   └── optional/
│   │       ├── kafka.py
│   │       └── redis.py
│   │
│   └── tests/
│       ├── test_health.py
│       ├── test_upload.py
│       ├── test_processing.py
│       ├── test_analytics.py
│       ├── test_langgraph.py
│       ├── test_gemini_manager.py
│       ├── test_chat.py
│       └── test_reports.py
│
├── supabase/
│   └── schema.sql
│
└── reports/
```

---

# 7. Environment Variables

Create `.env.example`.

Required:

```env
SUPABASE_URL=
SUPABASE_KEY=

GEMINI_KEY_1=
GEMINI_KEY_2=
GEMINI_KEY_3=

GEMINI_MODEL=
```

Optional:

```env
REDIS_URL=

KAFKA_BOOTSTRAP_SERVERS=
KAFKA_TOPIC=
```

Rules:

- [ ] Never hard-code API keys
- [ ] Never commit `.env`
- [ ] Use environment variables
- [ ] Keep Supabase service-role credentials server-side
- [ ] Only use authorized Gemini API keys
- [ ] Do not use key rotation to bypass provider limits

---

# 8. Supabase Database Schema

## 8.1 datasets

```text
id
name
source_type
uploaded_at
total_records
analysis_status
date_start
date_end
created_at
```

Status:

```text
uploaded
processing
completed
failed
```

## 8.2 feedback

```text
id
dataset_id
text
date
category
sentiment
sentiment_confidence
topics
tags
language
created_at
```

Sentiment:

```text
positive
neutral
negative
```

## 8.3 analytics

```text
id
dataset_id
total_count

positive_count
positive_percentage

neutral_count
neutral_percentage

negative_count
negative_percentage

top_topics
sentiment_trends
category_metrics

updated_at
```

## 8.4 decisions

```text
id
dataset_id
issue
topic
severity
confidence
evidence
impact
recommendation
status
needs_human_review
created_at
```

Severity:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

## 8.5 reports

```text
id
dataset_id
report_type
file_name
file_path
created_at
status
```

---

# 9. Supabase Requirements

- [ ] Create Supabase project
- [ ] Create PostgreSQL tables
- [ ] Add foreign keys
- [ ] Add indexes for common queries
- [ ] Configure appropriate RLS
- [ ] Connect backend
- [ ] Verify inserts
- [ ] Verify updates
- [ ] Verify queries
- [ ] Configure Realtime for relevant tables

Do not expose service-role keys through any future frontend.

---

# 10. API Contract

All APIs should:

- return JSON unless the endpoint explicitly returns a file
- use Pydantic request/response models
- return meaningful HTTP status codes
- validate inputs
- handle exceptions
- return structured error responses
- include dataset IDs where relevant
- be documented automatically in `/docs`

---

# 11. Required API Endpoints

## 11.1 Health

### GET `/health`

Purpose:

Verify that the backend is running.

Example:

```json
{
  "status": "ok",
  "service": "feedbackiq"
}
```

Checklist:

- [ ] Endpoint created
- [ ] Returns 200
- [ ] Swagger documentation visible

---

# 12. Dataset APIs

## POST `/api/datasets`

Create a dataset record before/alongside ingestion.

Request:

```json
{
  "name": "Customer Feedback September",
  "source_type": "csv"
}
```

Response:

```json
{
  "id": "dataset-id",
  "name": "Customer Feedback September",
  "status": "uploaded"
}
```

- [ ] Implement
- [ ] Validate input
- [ ] Store in Supabase

---

## GET `/api/datasets`

Return available datasets.

Response:

```json
{
  "datasets": [
    {
      "id": "123",
      "name": "Customer Feedback",
      "total_records": 10000,
      "analysis_status": "completed"
    }
  ]
}
```

- [ ] Implement
- [ ] Query Supabase

---

## GET `/api/datasets/{dataset_id}`

Return dataset metadata.

- [ ] Implement
- [ ] Handle unknown dataset
- [ ] Return analysis status

---

# 13. Upload / Ingestion APIs

## POST `/api/upload`

Accept:

- CSV
- XLSX

Use `multipart/form-data`.

Responsibilities:

```text
Upload file
 ↓
Validate extension
 ↓
Read dataframe
 ↓
Detect columns
 ↓
Clean/validate
 ↓
Create dataset
 ↓
Store feedback
 ↓
Return dataset ID
```

Response should include:

```json
{
  "dataset_id": "123",
  "file_name": "feedback.csv",
  "rows_received": 10000,
  "rows_accepted": 9850,
  "rows_rejected": 150,
  "detected_columns": {
    "text": "review",
    "date": "date",
    "category": "department"
  },
  "status": "uploaded"
}
```

- [ ] CSV
- [ ] XLSX
- [ ] File validation
- [ ] Column detection
- [ ] Dataset creation
- [ ] Supabase insertion
- [ ] Error handling

---

# 14. Optional Live Feedback API

## POST `/api/feedback/live`

Purpose:

Allow external systems to submit one new feedback item.

Request:

```json
{
  "dataset_id": "123",
  "text": "The Wi-Fi is very slow.",
  "date": "2026-09-07",
  "category": "Infrastructure"
}
```

Flow:

```text
POST
 ↓
Validate
 ↓
Store
 ↓
Analyse
 ↓
Update analytics
 ↓
Trigger significance check
 ↓
Potentially trigger LangGraph
```

- [ ] Endpoint
- [ ] Pydantic schema
- [ ] Supabase insert
- [ ] Analysis
- [ ] Realtime-compatible update

---

# 15. Data Processing APIs

## POST `/api/analyze/{dataset_id}`

Start analysis for a dataset.

Pipeline:

```text
Dataset
 ↓
Preprocessing
 ↓
Sentiment
 ↓
Topics
 ↓
Trends
 ↓
Aggregations
 ↓
Significance
 ↓
Insights
 ↓
Decision workflow where required
```

Response:

```json
{
  "dataset_id": "123",
  "status": "processing"
}
```

- [ ] Implement
- [ ] Background processing
- [ ] Status updates
- [ ] Error state

---

## GET `/api/analyze/{dataset_id}/status`

Return:

```json
{
  "dataset_id": "123",
  "status": "completed",
  "progress": 100
}
```

- [ ] Implement
- [ ] Processing percentage
- [ ] Error information

---

# 16. Sentiment Analysis

Required output per feedback:

```json
{
  "sentiment": "negative",
  "confidence": 0.94
}
```

Labels:

```text
positive
neutral
negative
```

Rules:

- [ ] Use an efficient classifier/model
- [ ] Avoid Gemini per row
- [ ] Store sentiment
- [ ] Store confidence
- [ ] Handle uncertain predictions

---

# 17. Topic Extraction

For each feedback:

```text
topics
tags
```

Example:

```json
{
  "topics": ["wifi", "connectivity"],
  "tags": ["infrastructure", "internet"]
}
```

Dataset-level output:

```text
Topic | Mentions | Negative %
```

- [ ] Topic extraction
- [ ] Topic aggregation
- [ ] Negative percentage per topic
- [ ] Store results

---

# 18. Trend Analysis

Calculate:

- daily trends
- weekly trends
- monthly trends where data supports them
- sentiment trends
- category trends
- topic trends

Example:

```text
Week 1 → 42% negative
Week 2 → 48% negative
Week 3 → 55% negative
Week 4 → 63% negative
```

- [ ] Date normalization
- [ ] Sentiment trend calculation
- [ ] Category trend calculation
- [ ] Topic trend calculation

---

# 19. Dashboard Data API

## GET `/api/dashboard/{dataset_id}`

This endpoint provides all aggregated data required by a future frontend.

Response:

```json
{
  "dataset": {
    "id": "123",
    "name": "Customer Feedback",
    "total_records": 10000
  },
  "kpis": {
    "total_feedback": 10000,
    "positive_percentage": 32.0,
    "neutral_percentage": 18.0,
    "negative_percentage": 50.0
  },
  "sentiment_distribution": [],
  "sentiment_trend": [],
  "top_topics": [],
  "category_sentiment": [],
  "critical_issues": [],
  "insights": [],
  "decisions": []
}
```

- [ ] Implement
- [ ] Verify calculations
- [ ] Query Supabase efficiently

---

# 20. Filtering APIs

The future frontend should not need to implement complex database logic itself.

Support query parameters:

```text
GET /api/feedback/{dataset_id}
    ?sentiment=negative
    &category=Infrastructure
    &topic=wifi
    &start_date=2026-09-01
    &end_date=2026-09-07
    &page=1
    &limit=50
```

Response:

```json
{
  "data": [],
  "page": 1,
  "limit": 50,
  "total": 1234
}
```

- [ ] Sentiment filter
- [ ] Category filter
- [ ] Topic filter
- [ ] Date filter
- [ ] Pagination
- [ ] Sorting if useful

---

# 21. Significance / Priority Engine

Basic analytics must determine whether an issue deserves deeper AI reasoning.

Example:

```text
Priority Score =
Mention Frequency × Negative Percentage
```

Potential trigger:

```text
IF negative_percentage > threshold
AND mention_count > threshold
AND trend_growth > threshold
THEN trigger LangGraph
```

Thresholds must be configurable.

- [ ] Priority score
- [ ] Threshold configuration
- [ ] Significant issue detection
- [ ] Critical issue detection
- [ ] Unit tests

---

# 22. LangGraph Decision Engine

## Purpose

LangGraph is used for **higher-level decision and insight generation**.

It should NOT process every feedback row.

Flow:

```text
Analytics
 ↓
Significance Engine
 ↓
Significant issue
 ↓
LangGraph
 ↓
Gemini reasoning
 ↓
Validated decision
```

---

# 23. LangGraph State

Suggested state:

```python
{
    "dataset_id": "...",
    "issue": "...",
    "topic": "...",
    "metrics": {},
    "evidence": [],
    "severity": "",
    "confidence": 0.0,
    "impact": [],
    "recommendations": [],
    "validation": {},
    "final_decision": ""
}
```

- [ ] Define TypedDict/Pydantic-compatible state
- [ ] Add required fields
- [ ] Validate state

---

# 24. LangGraph Nodes

## Node 1 — collect_evidence

Collect:

- mention count
- sentiment distribution
- negative percentage
- trend
- category
- representative feedback

- [ ] Implement

## Node 2 — evaluate_significance

Determine whether evidence justifies action.

- [ ] Implement

## Node 3 — determine_severity

Return:

```text
LOW
MEDIUM
HIGH
CRITICAL
```

- [ ] Implement

## Node 4 — assess_impact

Possible impacts:

```text
customer satisfaction
student satisfaction
retention
revenue
operations
service quality
```

- [ ] Implement

## Node 5 — generate_recommendation

Use Gemini.

Recommendations must be actionable and evidence-based.

- [ ] Implement

## Node 6 — validate_decision

Check:

- evidence support
- unsupported claims
- confidence
- actionability

- [ ] Implement

## Graph Completion

- [ ] START node
- [ ] Evidence node
- [ ] Significance node
- [ ] Severity node
- [ ] Impact node
- [ ] Recommendation node
- [ ] Validation node
- [ ] END node
- [ ] Persist final decision

---

# 25. Gemini API Manager

Create:

```text
backend/app/ai/gemini_manager.py
```

Basic rotation:

```text
Request 1 → Key 1
Request 2 → Key 2
Request 3 → Key 3
Request 4 → Key 1
```

Better behavior:

```text
Get next key
 ↓
Skip temporarily unavailable keys
 ↓
Call Gemini
 ↓
If rate limited:
    mark key temporarily unavailable
    try next authorized key
 ↓
If all unavailable:
    controlled failure/retry
```

Use Gemini for:

- executive summary
- high-level insights
- LangGraph reasoning
- recommendations
- chatbot
- report narrative

Do NOT send the complete raw dataset to Gemini unnecessarily.

- [ ] Load keys from environment
- [ ] Round-robin pointer
- [ ] Concurrency-safe key selection if needed
- [ ] Rate-limit handling
- [ ] Temporary key cooldown
- [ ] Retry logic
- [ ] Structured response handling
- [ ] Tests

---

# 26. AI Insights API

## GET `/api/insights/{dataset_id}`

Return:

```json
{
  "executive_summary": "...",
  "key_strengths": [],
  "key_issues": [],
  "recommended_actions": []
}
```

Insights must be grounded in stored analytics.

- [ ] Executive summary
- [ ] Strengths
- [ ] Issues
- [ ] Recommendations
- [ ] Supabase persistence

---

# 27. Decisions API

## GET `/api/decisions/{dataset_id}`

Return all AI decisions.

Example:

```json
{
  "decisions": [
    {
      "issue": "Wi-Fi reliability",
      "severity": "HIGH",
      "confidence": 0.93,
      "impact": [
        "customer satisfaction",
        "service quality"
      ],
      "recommendation": [
        "Investigate network infrastructure",
        "Identify affected locations"
      ]
    }
  ]
}
```

- [ ] Implement
- [ ] Filtering by severity
- [ ] Supabase query
- [ ] Pagination if necessary

---

# 28. AI Chatbot API

## POST `/api/chat`

Request:

```json
{
  "dataset_id": "123",
  "question": "What are the biggest problems?"
}
```

Flow:

```text
Question
 ↓
Identify relevant information
 ↓
Retrieve dataset evidence
 ↓
Build compact context
 ↓
Gemini
 ↓
Structured answer
```

Response:

```json
{
  "answer": "The biggest issue is Wi-Fi reliability...",
  "evidence": [
    {
      "topic": "Wi-Fi",
      "mentions": 741,
      "negative_percentage": 72
    }
  ],
  "confidence": 0.91
}
```

Suggested questions:

```text
What are the biggest problems?

Why are users unhappy?

Which category has the most negative feedback?

Summarize negative feedback.

What should management fix first?

Compare two categories.

Why did negative sentiment increase?

What are the strongest positive areas?
```

Rules:

- [ ] Ground responses in dataset evidence
- [ ] Retrieve relevant statistics
- [ ] Do not invent metrics
- [ ] Return insufficient-evidence response when necessary
- [ ] Use Gemini manager

---

# 29. PDF Report API

## POST `/api/reports/pdf/{dataset_id}`

Generate:

1. Executive Summary
2. Dataset Overview
3. Overall Sentiment
4. Sentiment Trends
5. Topic Analysis
6. Category Analysis
7. Critical Issues
8. Positive Insights
9. AI Decisions
10. Recommendations
11. Supporting Evidence
12. Methodology

Response can return:

```json
{
  "report_id": "123",
  "file_name": "feedbackiq_report.pdf",
  "status": "completed"
}
```

- [ ] ReportLab implementation
- [ ] Existing analytics reused
- [ ] AI insights included
- [ ] Decisions included
- [ ] File stored/served
- [ ] Error handling

---

# 30. Excel Report API

## POST `/api/reports/excel/{dataset_id}`

Suggested sheets:

```text
Feedback
Sentiment Summary
Topic Analysis
Category Analysis
Trends
Decisions
```

- [ ] Pandas/OpenPyXL
- [ ] Multiple sheets
- [ ] Correct data
- [ ] File generated
- [ ] File served/downloadable

---

# 31. Report History API

## GET `/api/reports/{dataset_id}`

Return previous reports.

Example:

```json
{
  "reports": [
    {
      "id": "1",
      "type": "pdf",
      "file_name": "feedback_report.pdf",
      "created_at": "2026-09-07T10:00:00"
    }
  ]
}
```

- [ ] Implement
- [ ] Supabase query

---

# 32. Real-Time Capability

Use:

**Supabase Realtime**

Important events:

```text
feedback INSERT
analytics UPDATE
decisions INSERT
```

Backend responsibility:

- update Supabase correctly
- expose any needed status APIs
- keep database state consistent

Future frontend can subscribe directly to Supabase Realtime.

Flow:

```text
New feedback
 ↓
FastAPI
 ↓
Supabase
 ↓
AI analysis
 ↓
Analytics/decision rows updated
 ↓
Supabase Realtime
 ↓
Future frontend receives update
```

- [ ] Enable Realtime
- [ ] Test database event
- [ ] Verify frontend-independent realtime behavior
- [ ] Document subscription tables

---

# 33. Optional Kafka

Only start after core backend works.

Purpose:

- asynchronous ingestion
- event streaming
- decouple API from workers
- scalable processing

Potential:

```text
POST /api/feedback/live
        ↓
FastAPI
        ↓
Kafka topic: feedback-events
        ↓
AI Worker
        ↓
Supabase
```

Checklist:

- [ ] Kafka connection
- [ ] Topic
- [ ] Producer
- [ ] Consumer
- [ ] AI worker
- [ ] Error handling
- [ ] Retry strategy
- [ ] Verify no regression when disabled

If time is short, skip Kafka.

---

# 34. Optional Redis

Only start after core backend works.

Purpose:

- cache dashboard metrics
- cache recent insights
- cache repeated chatbot context
- reduce repeated Supabase queries

Potential:

```text
API
 ↓
Redis
 ↓ cache hit
Response

cache miss
 ↓
Supabase
 ↓
Redis
 ↓
Response
```

Checklist:

- [ ] Redis connection
- [ ] Cache wrapper
- [ ] Dashboard caching
- [ ] TTL
- [ ] Cache invalidation
- [ ] Fallback if Redis unavailable

Redis must NOT become the primary persistent database.

If time is short, skip Redis.

---

# 35. Optional Kubernetes

Last priority.

Only implement after all core APIs work.

Possible services:

```text
FastAPI
AI Worker
Kafka
Redis
```

Checklist:

- [ ] Docker image
- [ ] Deployment
- [ ] Service
- [ ] Environment/secrets
- [ ] Health checks
- [ ] Basic scaling
- [ ] Verify deployment

If time is short, skip Kubernetes.

---

# 36. Error Handling

All APIs should handle:

- invalid dataset ID
- invalid file
- missing file
- empty dataset
- missing feedback column
- malformed rows
- invalid dates
- Supabase failure
- Gemini failure
- rate limits
- LangGraph failure
- report generation failure
- timeout

Use consistent format:

```json
{
  "error": {
    "code": "DATASET_NOT_FOUND",
    "message": "The requested dataset does not exist."
  }
}
```

Checklist:

- [ ] Global exception handler
- [ ] Validation errors
- [ ] Database errors
- [ ] AI errors
- [ ] File errors

---

# 37. AI Hallucination Controls

Rules:

1. Calculate analytics programmatically.
2. Give Gemini evidence rather than raw assumptions.
3. Never allow Gemini to invent statistics.
4. Validate structured output.
5. Return uncertainty when evidence is insufficient.
6. Use LangGraph validation node.
7. Keep recommendations tied to observed issues.

Checklist:

- [ ] Evidence context builder
- [ ] Structured Gemini output
- [ ] Validation
- [ ] Confidence handling
- [ ] Insufficient-evidence response

---

# 38. Performance Strategy

DO NOT:

```text
10,000 feedback rows
 ↓
10,000 Gemini requests
```

Prefer:

```text
10,000 rows
 ↓
Batch/local sentiment
 ↓
Topic aggregation
 ↓
Trend aggregation
 ↓
Significant issues only
 ↓
Gemini/LangGraph
```

Goals:

- low latency
- lower API usage
- fewer failures
- better scalability

Checklist:

- [ ] Batch processing
- [ ] Avoid unnecessary LLM calls
- [ ] Efficient Supabase queries
- [ ] Pagination
- [ ] Background analysis

---

# 39. Testing

## Health

- [ ] `/health`

## Dataset

- [ ] Create dataset
- [ ] List datasets
- [ ] Get dataset
- [ ] Unknown dataset handling

## Upload

- [ ] CSV
- [ ] XLSX
- [ ] Invalid extension
- [ ] Empty file
- [ ] Missing text column
- [ ] Duplicate data
- [ ] Missing dates
- [ ] Missing categories

## Analysis

- [ ] Sentiment
- [ ] Confidence
- [ ] Topics
- [ ] Trends
- [ ] Aggregations
- [ ] Priority score

## LangGraph

- [ ] State creation
- [ ] Evidence node
- [ ] Significance node
- [ ] Severity node
- [ ] Impact node
- [ ] Recommendation node
- [ ] Validation node
- [ ] Final decision

## Gemini

- [ ] Single key
- [ ] Round robin
- [ ] Key rotation
- [ ] Rate-limit handling
- [ ] All keys unavailable
- [ ] Structured output

## Chat

- [ ] Basic question
- [ ] Dataset-grounded answer
- [ ] Evidence returned
- [ ] Unknown information
- [ ] Gemini failure

## Reports

- [ ] PDF
- [ ] Excel
- [ ] Correct data
- [ ] Report history

## Database

- [ ] Supabase connection
- [ ] Inserts
- [ ] Updates
- [ ] Queries
- [ ] Realtime event

---

# 40. 8-Hour Backend Execution Plan

## 0:00–0:45 — Setup

- [ ] Repository
- [ ] FastAPI
- [ ] Requirements
- [ ] Environment variables
- [ ] Supabase connection
- [ ] Gemini connection
- [ ] `/health`

**STOP CONDITION:** FastAPI → Supabase → Gemini basic connectivity works.

---

## 0:45–2:00 — Ingestion

- [ ] Supabase schema
- [ ] Dataset model
- [ ] CSV upload
- [ ] XLSX upload
- [ ] Column detection
- [ ] Cleaning
- [ ] Deduplication
- [ ] Store feedback

**STOP CONDITION:** Real dataset is stored in Supabase.

---

## 2:00–3:15 — Analysis

- [ ] Sentiment
- [ ] Confidence
- [ ] Topics
- [ ] Trends
- [ ] Aggregations
- [ ] Dashboard API
- [ ] Filtering API

**STOP CONDITION:** Complete analytics are available through REST APIs.

---

## 3:15–4:30 — Decision Engine

- [ ] Significance rules
- [ ] LangGraph state
- [ ] LangGraph nodes
- [ ] Gemini manager
- [ ] Decision persistence
- [ ] Decision API

**STOP CONDITION:** A real significant issue produces a validated AI decision.

---

## 4:30–5:15 — AI Insights

- [ ] Executive summary
- [ ] Key strengths
- [ ] Key issues
- [ ] Recommended actions
- [ ] Insights API

**STOP CONDITION:** Dataset can be converted into useful management-level insights.

---

## 5:15–6:15 — Chatbot

- [ ] Chat endpoint
- [ ] Evidence retrieval
- [ ] Gemini context
- [ ] Round-robin keys
- [ ] Grounded answers

**STOP CONDITION:** API can answer questions about actual feedback.

---

## 6:15–7:00 — Reports + Realtime

- [ ] PDF
- [ ] Excel
- [ ] Report history
- [ ] Supabase Realtime
- [ ] Live feedback endpoint

**STOP CONDITION:** Backend can generate reports and process new feedback.

---

## 7:00–8:00 — Testing + Optional Infrastructure

Priority:

```text
1. Fix core bugs
2. Test end-to-end
3. Verify Swagger
4. Verify Supabase
5. Verify Gemini
6. Verify LangGraph
7. Improve error handling
8. OPTIONAL Redis
9. OPTIONAL Kafka
10. OPTIONAL Kubernetes
```

Do not sacrifice a working core API for optional infrastructure.

---

# 41. Definition of Done — Backend MVP

The backend MVP is COMPLETE when this works:

```text
CSV/XLSX
   ↓
POST /api/upload
   ↓
Supabase
   ↓
POST /api/analyze/{dataset_id}
   ↓
Sentiment + Topics + Trends
   ↓
GET /api/dashboard/{dataset_id}
   ↓
Significance Detection
   ↓
LangGraph
   ↓
Gemini
   ↓
Decision
   ↓
GET /api/decisions/{dataset_id}
   ↓
POST /api/chat
   ↓
POST /api/reports/pdf/{dataset_id}
   ↓
POST /api/reports/excel/{dataset_id}
```

If this works end-to-end:

**MVP is DONE.**

---

# 42. API Endpoint Master Checklist

## Health

- [ ] `GET /health`

## Dataset

- [ ] `POST /api/datasets`
- [ ] `GET /api/datasets`
- [ ] `GET /api/datasets/{dataset_id}`

## Upload

- [ ] `POST /api/upload`
- [ ] `POST /api/feedback/live`

## Analysis

- [ ] `POST /api/analyze/{dataset_id}`
- [ ] `GET /api/analyze/{dataset_id}/status`

## Data

- [ ] `GET /api/feedback/{dataset_id}`

## Dashboard

- [ ] `GET /api/dashboard/{dataset_id}`

## Insights

- [ ] `GET /api/insights/{dataset_id}`

## Decisions

- [ ] `GET /api/decisions/{dataset_id}`

## Chatbot

- [ ] `POST /api/chat`

## Reports

- [ ] `POST /api/reports/pdf/{dataset_id}`
- [ ] `POST /api/reports/excel/{dataset_id}`
- [ ] `GET /api/reports/{dataset_id}`

---

# 43. Swagger / API Documentation

FastAPI should expose:

```text
/docs
/redoc
/openapi.json
```

Every endpoint should have:

- summary
- description
- request schema
- response schema
- possible error responses

Checklist:

- [ ] Swagger loads
- [ ] All endpoints visible
- [ ] Request models documented
- [ ] Response models documented
- [ ] File upload documented
- [ ] Example payloads added where useful

---

# 44. GitHub Copilot Instructions

When working on this repository:

## MUST

1. Read `BUILD_PLAN.md` before making changes.
2. Inspect existing files before creating new files.
3. Preserve working functionality.
4. Implement one feature at a time.
5. Update this checklist after completing work.
6. Use Supabase, NOT MongoDB.
7. Use LangGraph for decision/insight workflows.
8. Use Gemini for higher-level reasoning.
9. Keep Kafka/Redis/Kubernetes optional.
10. Never hard-code secrets.
11. Use Pydantic schemas.
12. Add error handling.
13. Add tests for important functionality.
14. Use real Supabase data.
15. Do not create fake analytics for the final backend.
16. Avoid Gemini per-row processing.
17. Keep APIs frontend-independent.
18. Make all important functionality accessible through REST endpoints.

## MUST NOT

- [ ] Add MongoDB
- [ ] Build React now
- [ ] Make Kafka mandatory
- [ ] Make Redis mandatory
- [ ] Make Kubernetes mandatory
- [ ] Hard-code Gemini keys
- [ ] Hard-code Supabase secrets
- [ ] Rewrite working modules unnecessarily
- [ ] Generate fake dashboard data
- [ ] Use Gemini for every row by default

---

# 45. Copilot Working Protocol

Before every task:

```text
1. Read BUILD_PLAN.md.
2. Inspect repository.
3. Check completed items.
4. Identify dependencies.
5. Implement only requested functionality.
6. Run tests.
7. Fix errors.
8. Update BUILD_PLAN.md.
9. Report changed files.
10. Recommend next unchecked task.
```

When chat context is getting large:

```text
Update BUILD_PLAN.md with the exact implementation status,
errors, decisions, and remaining work before continuing.
```

---

# 46. Recommended Copilot Prompt

Use this at the beginning of a new Copilot session:

```text
You are working on FeedbackIQ, an AI-Powered Feedback Intelligence backend.

IMPORTANT:
Read BUILD_PLAN.md completely before doing anything.

CURRENT SCOPE:
We are building ONLY the backend/API layer right now.
Do NOT create React, Vite, Tailwind, charts, or frontend pages.

STACK:
- Python
- FastAPI
- Pandas
- Supabase/PostgreSQL
- Supabase Realtime
- Gemini API
- LangGraph
- ReportLab
- OpenPyXL/Pandas

DATABASE:
Supabase is the only database.
MongoDB must NOT be introduced.

ARCHITECTURE:
Feedback sources
→ FastAPI
→ preprocessing
→ Supabase
→ sentiment/topics/trends
→ significance engine
→ LangGraph for significant issues
→ Gemini reasoning
→ decisions/insights
→ REST APIs
→ reports/chatbot

IMPORTANT AI RULE:
Do not call Gemini once per feedback row by default.
Use efficient sentiment/analysis processing first.
Use Gemini for higher-level summaries, recommendations, chatbot answers,
report narratives, and selected LangGraph reasoning nodes.

LANGGRAPH:
LangGraph is specifically responsible for controlled multi-step
decision and insight workflows:
1. collect evidence
2. evaluate significance
3. determine severity
4. assess impact
5. generate recommendation
6. validate decision

GEMINI:
Implement a reusable Gemini API key manager with round-robin selection.
Skip temporarily unavailable/rate-limited authorized keys.
Never hard-code keys.

OPTIONAL:
Kafka, Redis, Kubernetes are optional.
Do not implement them until the core backend is stable.

API:
All important functionality must be exposed through FastAPI endpoints.
Swagger `/docs` must be complete.

WORKING RULES:
1. Read BUILD_PLAN.md.
2. Inspect existing code.
3. Do not duplicate existing functionality.
4. Implement the requested task.
5. Run relevant tests.
6. Fix errors.
7. Update BUILD_PLAN.md checklist.
8. Report:
   - files changed
   - functionality implemented
   - tests run
   - remaining issues
   - next recommended task

Do not move to frontend development.
Focus on making the backend complete and independently testable.
```

---

# 47. Jury Explanation — Backend

## What does the backend do?

```text
The backend accepts feedback from files or live sources,
processes and stores the data in Supabase,
performs sentiment/topic/trend analysis,
detects significant issues,
and uses LangGraph with Gemini to turn those issues into
structured decisions and recommendations.
```

## Why LangGraph?

```text
LangGraph provides a controlled multi-step decision workflow.

Instead of giving one large prompt to an LLM, the system separates:
evidence collection, significance evaluation, severity,
impact analysis, recommendation generation, and validation.

This makes the AI decision process more structured and controllable.
```

## Why not Gemini for everything?

```text
Basic statistics and row-level analysis do not require an LLM.

Programmatic analysis is faster and cheaper.
Gemini is used where reasoning adds value:
insights, decisions, recommendations, chatbot answers, and reports.
```

## Why Supabase?

```text
Supabase provides PostgreSQL persistence and Realtime capabilities.
It allows us to store feedback, analytics, decisions, and reports
while also supporting real-time data updates.
```

## Why Kafka?

```text
Kafka is an optional scalability layer that can decouple ingestion
from asynchronous AI processing when feedback volume becomes large.
```

## Why Redis?

```text
Redis is an optional caching layer for frequently requested analytics,
recent insights, and temporary AI context.
```

---

# 48. Final Status Tracker

## Architecture

- [x] Problem Statement 1 selected
- [x] MongoDB removed
- [x] Supabase selected
- [x] LangGraph selected for decision/insight workflow
- [x] Gemini selected for AI reasoning
- [x] Gemini round-robin planned
- [x] Kafka optional
- [x] Redis optional
- [x] Kubernetes optional
- [x] Frontend deferred
- [x] Backend/API-first approach selected

## Core Backend

- [ ] Repository initialized
- [ ] FastAPI initialized
- [ ] Environment configuration
- [ ] Supabase schema
- [ ] Supabase connection
- [ ] Dataset APIs
- [ ] CSV upload
- [ ] XLSX upload
- [ ] Preprocessing
- [ ] Sentiment
- [ ] Topics
- [ ] Trends
- [ ] Analytics
- [ ] Filtering
- [ ] Dashboard API
- [ ] Significance engine
- [ ] Gemini manager
- [ ] LangGraph
- [ ] AI insights
- [ ] Decisions API
- [ ] Chatbot API
- [ ] PDF reports
- [ ] Excel reports
- [ ] Report history
- [ ] Realtime
- [ ] Tests
- [ ] Swagger complete
- [ ] End-to-end verification

## Optional

- [ ] Redis
- [ ] Kafka
- [ ] Docker improvements
- [ ] Kubernetes

---

# 49. Current Project Status

**Status: Backend architecture/specification complete. Implementation pending.**

### Already decided

- [x] Problem Statement 1
- [x] Supabase instead of MongoDB
- [x] Backend/API-first development
- [x] LangGraph for decision and insight workflows
- [x] Gemini for higher-level reasoning
- [x] Gemini round-robin manager
- [x] Supabase Realtime
- [x] AI chatbot API
- [x] PDF report API
- [x] Excel report API
- [x] Kafka optional
- [x] Redis optional
- [x] Kubernetes optional

### Not yet implemented

- [ ] FastAPI project
- [ ] Supabase schema
- [ ] Ingestion
- [ ] Processing
- [ ] Sentiment
- [ ] Topics
- [ ] Trends
- [ ] Analytics APIs
- [ ] Significance engine
- [ ] Gemini manager
- [ ] LangGraph
- [ ] Insights
- [ ] Chatbot
- [ ] Reports
- [ ] Realtime
- [ ] Tests

---

# 50. Session Handoff Template

When moving to another AI/Copilot session:

```text
This is an existing backend project.

Read BUILD_PLAN.md completely before making changes.

Do not assume previous chat context.

Current status:
[Paste Current Project Status.]

Last completed task:
[Describe task.]

Current error/problem:
[Paste error.]

Requested next task:
[Describe task.]

Rules:
- Supabase, NOT MongoDB
- Backend/API only
- LangGraph for decision/insight workflow
- Gemini for reasoning
- Gemini key round-robin
- Kafka/Redis/Kubernetes optional
- No frontend yet
- Do not rewrite working code unnecessarily
- Update BUILD_PLAN.md after completion
- Run tests after implementation
```

---

# 51. Critical Rule

## BUILD THE BACKEND PRODUCT BEFORE THE INFRASTRUCTURE.

Priority:

```text
1. FastAPI
2. Supabase
3. Ingestion
4. Processing
5. Sentiment
6. Topics
7. Trends
8. Analytics APIs
9. Significance engine
10. LangGraph
11. Gemini manager
12. AI insights
13. Chatbot
14. Reports
15. Realtime
16. Redis
17. Kafka
18. Kubernetes
```

If time becomes limited:

```text
KEEP:
FastAPI
Supabase
Sentiment
Topics
Analytics
LangGraph
Gemini
Chatbot
Reports
Realtime

DROP/DEFER:
Kubernetes
Kafka
Redis
```

The backend is successful when another application can consume the APIs and get the complete:

**Feedback → Intelligence → Decision → Report**

pipeline without requiring any frontend implementation.

# Steam Player Experience Intelligence (PEI)

### Review-to-Insight System for Game Health, Churn Risk, and Feature Opportunity Detection

This repository implements a production-grade player analytics pipeline and dashboard designed to mirror live-ops game intelligence and product health monitoring systems. This project processes data ingestion (ETL), SQL-based data transformations, Machine Learning classification, and business-focused dashboard reporting.

---

## System Architecture and Data Flow

The system coordinates raw player reviews and metadata through a modular four-stage analytics pipeline:

```
[ Steam Web API / Live Ingestion ]
              │
              ▼ (ingest.py)
┌─────────────────────────────────────────┐
│     DuckDB: Raw Ingestion Layer         │
│  - raw_games (App details metadata)     │
│  - raw_reviews (Raw review strings)     │
└─────────────────────────────────────────┘
              │
              ▼ (models.py)
┌─────────────────────────────────────────┐
│     NLP Theme Classification (ML)       │
│  - Weak supervision labelling           │
│  - scikit-learn Multi-label Classifier  │
│  - reviews_cleaned (Annotated reviews)  │
└─────────────────────────────────────────┘
              │
              ▼ (transform.py)
┌─────────────────────────────────────────┐
│   DuckDB: SQL Transformation Layer      │
│  - Window functions & rolling averages  │
│  - game_daily_kpis                      │
│  - game_kpi_summary                     │
└─────────────────────────────────────────┘
              │
              ▼ (app.py)
┌─────────────────────────────────────────┐
│           Streamlit Dashboard           │
│  - Executive metrics & PERI indicator   │
│  - Daily KPI trends & review velocity   │
│  - NLP theme footprint & explorer       │
│  - Actionable product recommendations   │
└─────────────────────────────────────────┘
```

---

## Relational Schema (DuckDB)

The project leverages DuckDB to model a clean, performant dimensional schema:

| Table Name | Description | Key Fields |
| :--- | :--- | :--- |
| `raw_games` | Steam storefront metadata for ingested titles. | `appid` (PK), `title`, `developer`, `publisher`, `genres`, `price`, `release_date`, `header_image` |
| `raw_reviews` | Raw player reviews from the Steam Web API. | `review_id` (PK), `appid` (FK), `review_text`, `recommended` (BOOL), `playtime_forever`, `votes_up`, `timestamp_created` |
| `reviews_cleaned` | Processed reviews with sentiment scores and NLP labels. | `review_id` (PK), `sentiment_score`, `is_performance` (BOOL), `is_monetization` (BOOL), `is_bugs` (BOOL), `is_grind` (BOOL), `is_balance` (BOOL), `is_matchmaking` (BOOL) |
| `game_daily_kpis` | SQL-aggregated daily metrics for time-series charts. | `appid` (FK), `review_date` (PK), `review_count`, `positive_count`, `negative_count`, `rolling_7d_pos_rate`, `rolling_7d_review_volume` |
| `game_kpi_summary` | Summary table containing overall KPIs and default statistics. | `appid` (PK), `title`, `total_reviews`, `positive_reviews`, `overall_pos_rate`, `avg_playtime`, `total_veterans`, `veteran_churn_rate` |

---

## Player Experience Risk Index (PERI)

The Player Experience Risk Index (PERI) is a compound risk score (0 to 100) indicating the likelihood of player churn and sentiment deterioration. Rather than relying on simple aggregate review percentages, the PERI weights structural indicators:

\[PERI = 0.35 \cdot R_{\text{neg}} + 0.25 \cdot V_{\text{churn}} + 0.15 \cdot M_{\text{recent}} + 0.25 \cdot S_{\text{complaints}}\]

### Components:

1. **Sentiment Risk ($R_{\text{neg}}$)**: Overall ratio of negative reviews, scaled to 0-100. (Weight: **35%**)
2. **Veteran Churn Risk ($V_{\text{churn}}$)**: Ratio of negative reviews submitted by long-term players (playtime $>50$ hours). Churning veterans indicate deeper engagement issues. (Weight: **25%**)
3. **Recent Sentiment Momentum ($M_{\text{recent}}$)**: Compares the positive review rate of the last 7 days against the last 30 days. If the positive rate drops rapidly, risk accelerates. (Weight: **15%**)
4. **Complaint Severity Index ($S_{\text{complaints}}$)**: A weighted index based on active complaint categories in negative reviews. Issues that prevent gameplay (performance/matchmaking) are penalized heavier:
   \[Severity = 0.35 \cdot c_{\text{perf}} + 0.25 \cdot c_{\text{monetization}} + 0.20 \cdot c_{\text{matchmaking}} + 0.15 \cdot c_{\text{bugs}} + 0.10 \cdot c_{\text{balance}} + 0.10 \cdot c_{\text{grind}}\]
   (Weight: **25%**)

---

## Machine Learning & Weak Supervision NLP

To classify unstructured reviews into the 6 complaint categories without requiring expensive manual annotations, this project uses Weak Supervision:

- **Regex Bootstrapping**: Review text is scanned using domain-specific regex lexicons (optimised for gaming terminology like *fps drop*, *battle pass*, *server queue*, *nerfed*) to create soft labels.
- **Supervised Generalization**: We train a `scikit-learn` `MultiOutputClassifier` wrapping a `LogisticRegression` pipeline on TF-IDF word vectors. 
- **Semantic Generalization**: The trained model generalizes labels across reviews that don't match the exact dictionary keywords by learning semantic associations of co-occurring words.

---

## Setup & Execution

### 1. Requirements & Dependencies
Ensure Python 3.10+ is installed. Create a virtual environment and install the required libraries:

```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Data Pipeline
Clear old datasets and fetch real player reviews from the last 90 days:

```bash
# Fetch real reviews from the last 90 days for Helldivers 2 (default AppID: 553850)
# This initializes the database, executes the ETL, and runs the SQL/ML pipelines
python run_pipeline.py --clean

# Ingest and analyze real reviews of the last 90 days for any other Steam AppID (e.g. 730 for Counter-Strike 2)
python run_pipeline.py --appid 730

# Verify the database schema and table counts
python run_pipeline.py --test
```

### 3. Start the Interactive Dashboard
Launch the Streamlit dashboard to interact with charts, filter reviews, and view action recommendations:

```bash
streamlit run frontend/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Key Features

- **Steam Store Search Integration**: Users search for any game by name directly in the UI, fetching its live Steam storefront metadata and initiating the review ingestion.
- **Modular Timeframe Selection**: Select dynamic timeframe windows (`Last 30 Days`, `Last 60 Days`, `Last 90 Days`) to filter raw data, and calculate metrics, trend charts, and PERI scores instantly.
- **Executive Metrics Panel**: Track overall game risk index (PERI) and health status (Critical, High, Moderate, Stable, Healthy).
- **SQL Analytics Area**: Explore rolling sentiment timelines and review velocity bars using interactive Plotly figures.
- **NLP Theme Miner**: Inspect the 6 complaint themes and check their prevalence across negative player feedback.
- **Action Plan & Recommendations**: Review prioritized recommendations describing what the engineering, design, and live-ops teams should focus on first.
- **Semantic Review Search**: Filter and search through raw reviews by playtime, recommendation status, and specific detected complaint categories.

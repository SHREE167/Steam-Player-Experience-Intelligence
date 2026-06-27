# Steam Player Experience Intelligence (PEI)

### *Review-to-Insight System for Game Health, Churn Risk, and Feature Opportunity Detection*

This repository implements a production-grade player analytics pipeline and dashboard. It is designed to mirror professional industry practices for **Live-Ops Game Intelligence** and **Product Health Monitoring**. Rather than a basic notebook or modeling script, this project separates data ingestion (ETL), SQL-based data transformations, Machine Learning classification, and business-focused dashboard reporting.

---

## 🏗️ System Architecture & Data Flow

The system coordinates raw player reviews and metadata through a clean four-stage analytics pipeline:

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
│  - Window functions & rolling averages   │
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

## 📊 Relational Schema (DuckDB)

The project leverages **DuckDB** to model a clean, highly performant dimensional schema:

*   `raw_games`: Steam metadata including AppID, Title, Developer, Publisher, Genre tags, Price, Release date, and Header image URL.
*   `raw_reviews`: Raw review documents containing Review ID, AppID, recommendation vote, playtime hours, helpful votes, and creation timestamp.
*   `reviews_cleaned`: NLP-processed table containing predicted sentiment scores and binary theme flags (`is_performance`, `is_monetization`, `is_bugs`, `is_grind`, `is_balance`, `is_matchmaking`).
*   `game_daily_kpis`: SQL-aggregated daily table tracking review counts, average playtime, rolling 7-day moving averages of sentiment, and rolling complaint volumes.
*   `game_kpi_summary`: Combined metrics containing overall counts, veteran churn ratios, recent momentum shifts, and calculated risk scores.

---

## 🧮 Player Experience Risk Index (PERI)

The **Player Experience Risk Index (PERI)** is a compound risk score (0 to 100) indicating the likelihood of player churn and sentiment deterioration. Rather than simple aggregate review percentages, the PERI weights structural indicators:

$$PERI = 0.35 \cdot R_{\text{neg}} + 0.25 \cdot V_{\text{churn}} + 0.15 \cdot M_{\text{recent}} + 0.25 \cdot S_{\text{complaints}}$$

### Components:
1.  **Sentiment Risk ($R_{\text{neg}}$)**: Overall ratio of negative reviews, scaled to $0-100$. (Weight: **35%**)
2.  **Veteran Churn Risk ($V_{\text{churn}}$)**: Ratio of negative reviews submitted by long-term players (playtime $>50$ hours). Churning veterans indicate deeper engagement issues. (Weight: **25%**)
3.  **Recent Sentiment Momentum ($M_{\text{recent}}$)**: Compares the positive review rate of the last 7 days against the last 30 days. If the positive rate drops rapidly, risk accelerates. (Weight: **15%**)
4.  **Complaint Severity Index ($S_{\text{complaints}}$)**: A weighted index based on active complaint categories in negative reviews. Issues that prevent gameplay (performance/matchmaking) are penalized heavier:
    $$Severity = 0.35 \cdot c_{\text{perf}} + 0.25 \cdot c_{\text{monetization}} + 0.20 \cdot c_{\text{matchmaking}} + 0.15 \cdot c_{\text{bugs}} + 0.10 \cdot c_{\text{balance}} + 0.10 \cdot c_{\text{grind}}$$
    (Weight: **25%**)

---

## 🧠 Machine Learning & Weak Supervision NLP

To classify unstructured reviews into the 6 complaint categories without requiring expensive manual annotations, this project uses **Weak Supervision**:
1.  **Regex Bootstrapping**: Review text is scanned using domain-specific regex lexicons (optimized for gaming terminology like *fps drop*, *battle pass*, *server queue*, *nerfed*) to create soft labels.
2.  **Supervised Generalization**: We train a `scikit-learn` `MultiOutputClassifier` wrapping a `LogisticRegression` pipeline on TF-IDF word vectors. 
3.  **Generalization**: The trained model generalizes labels across reviews that don't match the exact dictionary keywords by learning semantic associations of co-occurring words.

---

## 🚀 Setup & Execution

### 1. Requirements & Dependencies
Ensure you have Python 3.10+ installed. Create a virtual environment and install the required libraries:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Run the Data Pipeline
Clear old datasets and fetch real player reviews from the last 90 days:
```bash
# Fetch real reviews from the last 90 days for Helldivers 2 (default AppID: 553850)
# This creates a clean database, runs ETL, and executes the SQL/ML pipelines
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

## 📌 Deliverable Features

*   **Steam Store Search Integration**: Search for any game on Steam by typing its name, which queries Steam store search suggestions, fetches metadata, and runs the 30-day review scraping pipeline.
*   **Executive Metrics Panel**: Track overall game risk index (PERI) and game health tier (Critical, High, Moderate, Stable, Healthy).
*   **SQL Analytics Area**: Visualize rolling sentiment timelines and review velocity bars using interactive Plotly figures.
*   **NLP Theme Miner**: Deep dive into the 6 complaint themes and view their prevalence across player feedback.
*   **Action Plan & Recommendations**: View autogenerated, prioritized product fixes for the live-ops and design teams (e.g., targetting matchmaking queues or optimizing frame rate stutters).
*   **Semantic Review Search**: Filter and search through raw reviews by playtime, recommendation status, and specific detected complaint categories.
"# Steam-Player-Experience-Intelligence" 

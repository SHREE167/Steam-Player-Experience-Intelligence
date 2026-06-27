import os
import sys
import time
import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import duckdb

# Add root folder to python path for pipeline access
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from run_pipeline import run_pipeline
from backend.pipeline.ingest import search_steam_games

# Database path
DB_PATH = os.path.join(ROOT_DIR, 'data', 'steam_analytics.db')

# Streamlit Page Setup
st.set_page_config(
    page_title="Steam Player Experience Intelligence (PEI)",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dark Theme & Layout
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background-color: #050505;
        color: #e5e5e5;
    }
    
    /* Typography */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
        background: linear-gradient(135deg, #d4af37 0%, #f3e5ab 50%, #aa771c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #0c0c0c;
        border-right: 1px solid #1c1917;
    }
    
    /* Metrics glassmorphism styling */
    .metric-card {
        background: rgba(18, 18, 18, 0.8);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(212, 175, 55, 0.15);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);
        transition: transform 0.2s ease-in-out, border-color 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(212, 175, 55, 0.5);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 5px;
        color: #d4af37;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a3a3a3;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Glowing status badges */
    .badge-critical {
        background: linear-gradient(135deg, #4c0519 0%, #dc2626 100%);
        color: white; padding: 4px 12px; border-radius: 20px; font-weight: 700;
        box-shadow: 0 0 15px rgba(220, 38, 38, 0.3);
        border: 1px solid rgba(220, 38, 38, 0.4);
    }
    .badge-high {
        background: linear-gradient(135deg, #431407 0%, #ea580c 100%);
        color: white; padding: 4px 12px; border-radius: 20px; font-weight: 700;
        box-shadow: 0 0 15px rgba(234, 88, 12, 0.3);
        border: 1px solid rgba(234, 88, 12, 0.4);
    }
    .badge-moderate {
        background: linear-gradient(135deg, #713f12 0%, #d4af37 100%);
        color: white; padding: 4px 12px; border-radius: 20px; font-weight: 700;
        box-shadow: 0 0 15px rgba(212, 175, 55, 0.3);
        border: 1px solid rgba(212, 175, 55, 0.4);
    }
    .badge-stable {
        background: linear-gradient(135deg, #14532d 0%, #16a34a 100%);
        color: white; padding: 4px 12px; border-radius: 20px; font-weight: 700;
        box-shadow: 0 0 15px rgba(22, 163, 74, 0.2);
    }
    .badge-healthy {
        background: linear-gradient(135deg, #064e3b 0%, #059669 100%);
        color: white; padding: 4px 12px; border-radius: 20px; font-weight: 700;
        box-shadow: 0 0 15px rgba(5, 150, 105, 0.2);
    }
    
    /* Recommendations Box */
    .rec-box {
        background: rgba(18, 18, 18, 0.85);
        border-left: 5px solid #d4af37;
        padding: 15px;
        margin-bottom: 12px;
        border-radius: 0 8px 8px 0;
        border-top: 1px solid rgba(212, 175, 55, 0.05);
        border-right: 1px solid rgba(212, 175, 55, 0.05);
        border-bottom: 1px solid rgba(212, 175, 55, 0.05);
    }
    .rec-urgent {
        border-left-color: #ef4444;
        background: rgba(239, 68, 68, 0.03);
    }
    .rec-high {
        border-left-color: #f97316;
        background: rgba(249, 115, 22, 0.03);
    }
    .rec-medium {
        border-left-color: #d4af37;
        background: rgba(212, 175, 55, 0.03);
    }
    .rec-low {
        border-left-color: #10b981;
        background: rgba(16, 185, 129, 0.03);
    }
    
    /* Review Explorer Rows */
    .review-row {
        background: rgba(15, 15, 15, 0.85);
        border: 1px solid rgba(212, 175, 55, 0.05);
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .tag-badge {
        display: inline-block;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        margin-right: 5px;
        margin-top: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to query DB safely
def query_db(query, params=None):
    if not os.path.exists(DB_PATH):
        return None
    try:
        conn = duckdb.connect(DB_PATH)
        if params:
            df = conn.execute(query, params).fetchdf()
        else:
            df = conn.execute(query).fetchdf()
        conn.close()
        return df
    except Exception as e:
        st.error(f"Database query failed: {e}")
        return None

def get_dynamic_metrics(appid, timeframe_days):
    cutoff_ts = int(time.time()) - (timeframe_days * 86400)
    cutoff_7d_ts = int(time.time()) - (7 * 86400)
    cutoff_30d_ts = int(time.time()) - (30 * 86400)
    
    query = """
        WITH overall_stats AS (
            SELECT 
                r.appid,
                COUNT(*) as total_reviews,
                SUM(CASE WHEN r.recommended = TRUE THEN 1 ELSE 0 END) as positive_reviews,
                SUM(CASE WHEN r.recommended = FALSE THEN 1 ELSE 0 END) as negative_reviews,
                AVG(r.playtime_forever) as avg_playtime,
                
                -- Veteran stats (playtime > 50 hours)
                SUM(CASE WHEN r.playtime_forever > 50.0 THEN 1 ELSE 0 END) as total_veterans,
                SUM(CASE WHEN r.playtime_forever > 50.0 AND r.recommended = FALSE THEN 1 ELSE 0 END) as veteran_churners,
                
                -- Theme totals (only among negative reviews to get mathematically correct ratios)
                COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_performance ELSE 0 END), 0) as total_perf_complaints,
                COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_monetization ELSE 0 END), 0) as total_money_complaints,
                COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_bugs ELSE 0 END), 0) as total_bug_complaints,
                COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_grind ELSE 0 END), 0) as total_grind_complaints,
                COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_balance ELSE 0 END), 0) as total_balance_complaints,
                COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_matchmaking ELSE 0 END), 0) as total_mm_complaints
            FROM raw_reviews r
            LEFT JOIN reviews_cleaned c ON r.review_id = c.review_id
            WHERE r.appid = ? AND r.timestamp_created >= ?
            GROUP BY r.appid
        ),
        recency_stats AS (
            SELECT 
                r.appid,
                SUM(CASE WHEN r.timestamp_created >= ? AND r.recommended = TRUE THEN 1 ELSE 0 END) as pos_reviews_7d,
                COUNT(CASE WHEN r.timestamp_created >= ? THEN 1 ELSE NULL END) as total_reviews_7d,
                SUM(CASE WHEN r.timestamp_created >= ? AND r.recommended = TRUE THEN 1 ELSE 0 END) as pos_reviews_30d,
                COUNT(CASE WHEN r.timestamp_created >= ? THEN 1 ELSE NULL END) as total_reviews_30d
            FROM raw_reviews r
            WHERE r.appid = ? AND r.timestamp_created >= ?
            GROUP BY r.appid
        )
        SELECT 
            g.appid,
            g.title,
            g.developer,
            g.publisher,
            g.genres,
            g.price,
            g.header_image,
            o.total_reviews,
            o.positive_reviews,
            o.negative_reviews,
            CAST(o.positive_reviews AS DOUBLE) / NULLIF(o.total_reviews, 0) as overall_pos_rate,
            o.avg_playtime,
            o.total_veterans,
            o.veteran_churners,
            CAST(o.veteran_churners AS DOUBLE) / NULLIF(o.total_veterans, 0) as veteran_churn_rate,
            
            -- Complaint distribution (relative to negative reviews)
            CAST(o.total_perf_complaints AS DOUBLE) / NULLIF(o.negative_reviews, 0) as perf_complaint_ratio,
            CAST(o.total_money_complaints AS DOUBLE) / NULLIF(o.negative_reviews, 0) as money_complaint_ratio,
            CAST(o.total_bug_complaints AS DOUBLE) / NULLIF(o.negative_reviews, 0) as bug_complaint_ratio,
            CAST(o.total_grind_complaints AS DOUBLE) / NULLIF(o.negative_reviews, 0) as grind_complaint_ratio,
            CAST(o.total_balance_complaints AS DOUBLE) / NULLIF(o.negative_reviews, 0) as balance_complaint_ratio,
            CAST(o.total_mm_complaints AS DOUBLE) / NULLIF(o.negative_reviews, 0) as mm_complaint_ratio,
            
            -- Recency metrics
            CAST(r.pos_reviews_7d AS DOUBLE) / NULLIF(r.total_reviews_7d, 0) as pos_rate_7d,
            CAST(r.pos_reviews_30d AS DOUBLE) / NULLIF(r.total_reviews_30d, 0) as pos_rate_30d
        FROM raw_games g
        LEFT JOIN overall_stats o ON g.appid = o.appid
        LEFT JOIN recency_stats r ON g.appid = r.appid
        WHERE g.appid = ?
    """
    params = [
        appid, cutoff_ts,
        cutoff_7d_ts, cutoff_7d_ts, cutoff_30d_ts, cutoff_30d_ts, appid, cutoff_ts,
        appid
    ]
    df = query_db(query, params)
    return df

def calculate_peri_score(row):
    # Component 1: Sentiment Risk (overall_pos_rate)
    neg_rate = 1.0 - row['overall_pos_rate'] if not pd.isna(row['overall_pos_rate']) else 0.0
    sentiment_risk = neg_rate * 100
    
    # Component 2: Veteran Churn Risk
    veteran_churn_rate = row['veteran_churn_rate']
    if pd.isna(veteran_churn_rate) or row['total_veterans'] == 0:
        veteran_churn_risk = sentiment_risk
    else:
        veteran_churn_risk = veteran_churn_rate * 100
        
    # Component 3: Recent Sentiment Decline (7d vs 30d)
    pos_7d = row['pos_rate_7d']
    pos_30d = row['pos_rate_30d']
    if pd.isna(pos_7d) or pd.isna(pos_30d):
        momentum_risk = 0.0
    else:
        decline = pos_30d - pos_7d
        if decline > 0:
            momentum_risk = min(decline * 200, 100.0)
        else:
            momentum_risk = max(decline * 50, -15.0)
            
    # Component 4: Severity Index
    perf_ratio = row['perf_complaint_ratio'] if not pd.isna(row['perf_complaint_ratio']) else 0.0
    money_ratio = row['money_complaint_ratio'] if not pd.isna(row['money_complaint_ratio']) else 0.0
    mm_ratio = row['mm_complaint_ratio'] if not pd.isna(row['mm_complaint_ratio']) else 0.0
    bug_ratio = row['bug_complaint_ratio'] if not pd.isna(row['bug_complaint_ratio']) else 0.0
    balance_ratio = row['balance_complaint_ratio'] if not pd.isna(row['balance_complaint_ratio']) else 0.0
    grind_ratio = row['grind_complaint_ratio'] if not pd.isna(row['grind_complaint_ratio']) else 0.0
    
    severity_score = (
        perf_ratio * 0.35 +
        money_ratio * 0.25 +
        mm_ratio * 0.20 +
        bug_ratio * 0.15 +
        balance_ratio * 0.10 +
        grind_ratio * 0.10
    )
    severity_risk = min(severity_score * 100 * 2.0, 100.0)
    
    # Combined PERI
    peri = (
        sentiment_risk * 0.35 +
        veteran_churn_risk * 0.25 +
        momentum_risk * 0.15 +
        severity_risk * 0.25
    )
    peri = max(0.0, min(100.0, peri))
    
    # Assign Risk Tier
    if peri >= 75.0:
        tier = "CRITICAL"
    elif peri >= 55.0:
        tier = "HIGH"
    elif peri >= 35.0:
        tier = "MODERATE"
    elif peri >= 15.0:
        tier = "STABLE"
    else:
        tier = "HEALTHY"
        
    return round(peri, 1), tier

# Sidebar Ingestion Control
st.sidebar.title("🎮 PEI Control Panel")

# Check if database has any games loaded
games_df = query_db("SELECT appid, title FROM game_kpi_summary")
preset_list = []
if games_df is not None and not games_df.empty:
    preset_list = list(games_df['title'].unique())

# Sidebar Dropdown
st.sidebar.subheader("Select Game")
selected_game_name = st.sidebar.selectbox("Choose analyzed game", preset_list if preset_list else ["Database empty"])

# Timeframe Window Filter
st.sidebar.subheader("Timeframe Window")
selected_timeframe = st.sidebar.selectbox("Select Analysis Timeframe", ["Last 30 Days", "Last 60 Days", "Last 90 Days"], index=0)
timeframe_days = 30
if selected_timeframe == "Last 60 Days":
    timeframe_days = 60
elif selected_timeframe == "Last 90 Days":
    timeframe_days = 90

# Option to search and fetch new game reviews
st.sidebar.subheader("Analyze New Game")
search_query = st.sidebar.text_input("🔍 Search Steam Store", "")

if search_query:
    with st.spinner("Searching Steam store..."):
        search_results, err = search_steam_games(search_query)
        
    if err:
        st.sidebar.error(f"Search failed: {err}")
    elif search_results:
        options = {item['title']: item['appid'] for item in search_results}
        selected_search = st.sidebar.selectbox("Matching Games", list(options.keys()))
        selected_appid = options[selected_search]
        
        btn_analyze = st.sidebar.button("🚀 Ingest & Analyze Reviews")
        if btn_analyze:
            with st.spinner(f"Ingesting & analyzing 90-day reviews for {selected_search}..."):
                try:
                    run_pipeline(appid=selected_appid)
                    st.sidebar.success("Analysis complete!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Error during pipeline run: {e}")
    else:
        st.sidebar.warning("No matching games found on Steam.")

# Option to reset database
if st.sidebar.button("Clear & Reset Database"):
    with st.spinner("Clearing database and reloading default game (Helldivers 2)..."):
        run_pipeline(appid=553850, clean_db=True)
        st.sidebar.success("Database reset complete!")
        st.rerun()

# ----------------- MAIN INTERFACE -----------------

# Header Section
st.title("Steam Player Experience Intelligence (PEI)")
st.caption("🎮 Portfolio-Grade Review-to-Insight System | Ingestion -> SQL KPIs -> NLP Multi-Labeling -> Churn Risk Scoring")

# Database empty fallback
if not preset_list:
    st.warning("⚠️ No data in database. Click **Clear & Reset Database** in the sidebar to populate the DuckDB database with Helldivers 2 reviews from the last 90 days, or enter a Steam AppID to query real-time reviews.")
    st.stop()

# Fetch metadata appid for selected game name
game_base = query_db("SELECT appid FROM raw_games WHERE title = ?", [selected_game_name])
if game_base is None or game_base.empty:
    st.error(f"Could not load metadata for {selected_game_name}")
    st.stop()
appid = int(game_base.iloc[0]['appid'])

# Fetch dynamic metrics based on selected timeframe window
game_info = get_dynamic_metrics(appid, timeframe_days)
if game_info is None or game_info.empty or pd.isna(game_info.iloc[0]['total_reviews']):
    st.error(f"No player reviews found for {selected_game_name} within the {selected_timeframe} window. Try expanding the timeframe or scraping fresh reviews.")
    st.stop()
    
game = game_info.iloc[0]

# Banner & Game Metadata Display
col_img, col_desc = st.columns([1, 4])
with col_img:
    if game['header_image']:
        st.image(game['header_image'], use_container_width=True)
    else:
        st.write("No Image")
with col_desc:
    st.subheader(game['title'])
    st.write(f"**Developer**: `{game['developer']}` | **Publisher**: `{game['publisher']}` | **Genres**: `{game['genres']}`")
    st.write(f"**Price**: `${game['price']:.2f}`" if game['price'] > 0 else "**Price**: `Free-to-Play`")

# 1. Dynamic KPI Metrics Grid
st.markdown(f"### 📊 Executive Player Experience Overview ({selected_timeframe})")
col_peri, col_total, col_pos, col_vet = st.columns(4)

# Color match risk tier dynamically
peri_score, risk_tier = calculate_peri_score(game)
badge_class = f"badge-{risk_tier.lower()}"

with col_peri:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="color: {'#ef4444' if risk_tier in ['CRITICAL', 'HIGH'] else '#f59e0b' if risk_tier == 'MODERATE' else '#10b981'};">
            {peri_score:.1f}
        </div>
        <div class="metric-label">Experience Risk (PERI)</div>
        <div style="margin-top:8px;"><span class="{badge_class}">{risk_tier}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_total:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{int(game['total_reviews']):,}</div>
        <div class="metric-label">Reviews Analyzed</div>
        <div style="margin-top:8px; font-size: 0.8rem; color:#64748b;">DuckDB raw_reviews count</div>
    </div>
    """, unsafe_allow_html=True)

with col_pos:
    pos_rate = game['overall_pos_rate'] * 100
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{pos_rate:.1f}%</div>
        <div class="metric-label">Positive Sentiment</div>
        <div style="margin-top:8px; font-size: 0.8rem; color:#64748b;">Overall recommend ratio</div>
    </div>
    """, unsafe_allow_html=True)

with col_vet:
    vet_churn = game['veteran_churn_rate'] * 100 if not pd.isna(game['veteran_churn_rate']) else 0.0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{vet_churn:.1f}%</div>
        <div class="metric-label">Veteran Churn Risk</div>
        <div style="margin-top:8px; font-size: 0.8rem; color:#64748b;">Negative reviews (Playtime > 50h)</div>
    </div>
    """, unsafe_allow_html=True)

st.write("---")

# 2. Daily Trends Section (SQL aggregated data)
st.markdown("### 📈 SQL Analytical Intelligence: Game Health & Velocity")
daily_kpis = query_db("SELECT * FROM game_daily_kpis WHERE appid = ? AND review_date >= ? ORDER BY review_date", [appid, datetime.date.today() - datetime.timedelta(days=timeframe_days)])

if daily_kpis is not None and not daily_kpis.empty:
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # Plotly Sentiment Trend Line Chart (Themed Gold)
        fig_sent = go.Figure()
        fig_sent.add_trace(go.Scatter(
            x=daily_kpis['review_date'],
            y=daily_kpis['rolling_7d_pos_rate'] * 100,
            mode='lines',
            name='7d Rolling Positive Rate',
            line=dict(color='#d4af37', width=3),
            fill='tozeroy',
            fillcolor='rgba(212, 175, 55, 0.05)'
        ))
        fig_sent.update_layout(
            title="Rolling 7-Day Positive Sentiment Trend (%)",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#d4af37'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#262626', range=[0, 100])
        )
        st.plotly_chart(fig_sent, use_container_width=True)
        
    with col_chart2:
        # Review Velocity Bar Chart (Themed Gold)
        fig_vol = px.bar(
            daily_kpis,
            x='review_date',
            y='review_count',
            labels={'review_count': 'Daily Reviews', 'review_date': 'Date'},
            title="Review Velocity (Ingestion Volume Tracker)",
            color_discrete_sequence=['#aa771c']
        )
        fig_vol.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#d4af37'),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#262626')
        )
        st.plotly_chart(fig_vol, use_container_width=True)

st.write("---")

# 3. NLP Theme Analysis Section (ML model predictions)
st.markdown("### 🧠 NLP Theme Mining: Complaint Footprint")

# Build a dataframe of the 6 complaint ratios
complaint_data = {
    'Theme': ['Performance', 'Monetization', 'Bugs', 'Grind', 'Balance', 'Matchmaking'],
    'Ratio': [
        game['perf_complaint_ratio'],
        game['money_complaint_ratio'],
        game['bug_complaint_ratio'],
        game['grind_complaint_ratio'],
        game['balance_complaint_ratio'],
        game['mm_complaint_ratio']
    ],
    'Color': ['#d4af37', '#b89730', '#c5a059', '#aa771c', '#f3e5ab', '#e6c35c']
}
df_comp = pd.DataFrame(complaint_data)
df_comp['Percentage'] = df_comp['Ratio'] * 100

col_comp_chart, col_comp_desc = st.columns([3, 2])

with col_comp_chart:
    fig_comp = px.bar(
        df_comp,
        x='Percentage',
        y='Theme',
        orientation='h',
        title="Detected Complaint Prevalence (in Negative Reviews)",
        color='Theme',
        color_discrete_sequence=df_comp['Color'].tolist()
    )
    fig_comp.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#d4af37'),
        xaxis=dict(showgrid=True, gridcolor='#262626', range=[0, 100]),
        yaxis=dict(showgrid=False),
        showlegend=False
    )
    st.plotly_chart(fig_comp, use_container_width=True)

with col_comp_desc:
    st.markdown("#### Theme Severity Analysis")
    st.write(
        "Our multi-output TF-IDF classifier scans review text for semantic patterns "
        "and categorizes negative reports into these 6 themes. The chart reflects "
        "the percentage of negative reviewers who actively complained about each topic."
    )
    # Highlight top problem
    top_complaint = df_comp.loc[df_comp['Percentage'].idxmax()]
    st.info(f"🚨 **Primary Issue**: **{top_complaint['Theme']}** is the highest volume complaint category, appearing in **{top_complaint['Percentage']:.1f}%** of negative player reviews.")

st.write("---")

# 4. Product Recommendations
st.markdown("### 📋 Actionable Recommendations for Product & Live-Ops Teams")

# Rule-based recommendation engine ranking priorities
recommendations = []
# 1. Performance
if game['perf_complaint_ratio'] >= 0.20:
    prio = "URGENT" if game['perf_complaint_ratio'] >= 0.35 else "HIGH"
    recommendations.append({
        'theme': 'Performance/Optimization', 'prio': prio, 'ratio': game['perf_complaint_ratio'],
        'css': 'rec-urgent' if prio == "URGENT" else 'rec-high',
        'desc': f"**Core Stability Issues**: Performance and crashes are reported in **{game['perf_complaint_ratio']*100:.1f}%** of negative reviews. Players are experiencing frame drops, stuttering, or full crashes.",
        'action': "Deploy hotfix targetting hardware optimization and memory leak issues. Core engine stability is a major driver of early-stage churn."
    })
# 2. Monetization
if game['money_complaint_ratio'] >= 0.15:
    prio = "HIGH" if game['money_complaint_ratio'] >= 0.30 or game['price'] == 0.0 else "MEDIUM"
    recommendations.append({
        'theme': 'Monetization/Microtransactions', 'prio': prio, 'ratio': game['money_complaint_ratio'],
        'css': 'rec-high' if prio == "HIGH" else 'rec-medium',
        'desc': f"**Commercial Frustration**: Player reaction to microtransactions/pricing is negative (**{game['money_complaint_ratio']*100:.1f}%** of negative reviews). This creates friction in progression.",
        'action': "Re-evaluate battle pass value propositions or cosmetics store pricing. Address player sentiment about 'paywalls' or 'cash grab' elements in the next community developer update."
    })
# 3. Matchmaking
if game['mm_complaint_ratio'] >= 0.15:
    prio = "HIGH" if game['mm_complaint_ratio'] >= 0.25 else "MEDIUM"
    recommendations.append({
        'theme': 'Matchmaking/Netcode', 'prio': prio, 'ratio': game['mm_complaint_ratio'],
        'css': 'rec-high' if prio == "HIGH" else 'rec-medium',
        'desc': f"**Connectivity & Queue Times**: Matchmaking or server disconnection issues represent **{game['mm_complaint_ratio']*100:.1f}%** of negative reviews.",
        'action': "Upgrade regional server capacities, optimize matchmaking logic (smurf detection / queue wait thresholds), and resolve netcode tick-rate issues."
    })
# 4. Bugs
if game['bug_complaint_ratio'] >= 0.20:
    prio = "MEDIUM" if game['bug_complaint_ratio'] < 0.35 else "HIGH"
    recommendations.append({
        'theme': 'Bugs/Glitches', 'prio': prio, 'ratio': game['bug_complaint_ratio'],
        'css': 'rec-medium' if prio == "MEDIUM" else 'rec-high',
        'desc': f"**Software Quality**: General bugs/glitches are prevalent (**{game['bug_complaint_ratio']*100:.1f}%** of negative reviews).",
        'action': "Identify and log critical quest-blocking bugs and visual glitches reported by users. Increase Q&A sweep frequencies before patch rollouts."
    })
# 5. Balance
if game['balance_complaint_ratio'] >= 0.15:
    recommendations.append({
        'theme': 'Balance/Gameplay Meta', 'prio': 'MEDIUM', 'ratio': game['balance_complaint_ratio'],
        'css': 'rec-medium',
        'desc': f"**Meta Unfairness**: Balancing issues are cited in **{game['balance_complaint_ratio']*100:.1f}%** of negative reviews. Players complain about specific weapons/classes being over or under-powered.",
        'action': "Address gameplay balance imbalances by adjusting parameters of outliers (nerf/buff cycles). Publish balance logs openly to reassure competitive players."
    })
# 6. Grind
if game['grind_complaint_ratio'] >= 0.15:
    recommendations.append({
        'theme': 'Grind/Progression Gate', 'prio': 'MEDIUM', 'ratio': game['grind_complaint_ratio'],
        'css': 'rec-medium',
        'desc': f"**Burnout Risk**: Grind and progression gating is frustrating players (**{game['grind_complaint_ratio']*100:.1f}%** of negative reviews).",
        'action': "Adjust XP gains, progression milestones, or crafting material requirements to reduce fatigue. Introduce weekly events/boosters."
    })

# If no recommendations matched (healthy game)
if not recommendations:
    st.success("✅ **Healthy Game Status**: Sentiment is positive, and no specific complaint theme meets the warning threshold. Recommendations: Maintain live-ops cadence and monitor player review velocity for future spikes.")
else:
    # Sort recommendations by ratio descending
    recommendations = sorted(recommendations, key=lambda x: x['ratio'], reverse=True)
    for rec in recommendations:
        st.markdown(f"""
        <div class="rec-box {rec['css']}">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <strong>📌 {rec['theme']}</strong>
                <span class="tag-badge" style="background:#0f172a; border: 1px solid #334155;">PRIORITY: {rec['prio']}</span>
            </div>
            <p style="margin: 8px 0; font-size:0.95rem; color:#cbd5e1;">{rec['desc']}</p>
            <div style="font-size:0.9rem; color:#38bdf8;"><strong>Recommended Action:</strong> {rec['action']}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("---")

# 5. Interactive Review Explorer
st.markdown("### 🔍 Semantic Review Explorer")

# Fetch all reviews for selected game with cleaned NLP data within the selected timeframe
reviews_query = """
    SELECT r.review_id, r.review_text, r.recommended, r.playtime_forever, r.votes_up,
           c.sentiment_score, c.is_performance, c.is_monetization, c.is_bugs,
           c.is_grind, c.is_balance, c.is_matchmaking
    FROM raw_reviews r
    LEFT JOIN reviews_cleaned c ON r.review_id = c.review_id
    WHERE r.appid = ? AND r.timestamp_created >= ?
"""
all_reviews = query_db(reviews_query, [appid, int(time.time()) - (timeframe_days * 86400)])

if all_reviews is not None and not all_reviews.empty:
    # Filters
    col_f1, col_f2, col_f3 = st.columns([1, 1, 2])
    with col_f1:
        sent_filter = st.selectbox("Filter Sentiment", ["All", "Recommended (Positive)", "Not Recommended (Negative)"])
    with col_f2:
        theme_filter = st.selectbox(
            "Filter by Theme", 
            ["All", "Performance", "Monetization", "Bugs", "Grind", "Balance", "Matchmaking"]
        )
    with col_f3:
        search_query = st.text_input("Search reviews by keyword", "")
        
    # Apply filters
    filtered_df = all_reviews.copy()
    
    if sent_filter == "Recommended (Positive)":
        filtered_df = filtered_df[filtered_df['recommended'] == True]
    elif sent_filter == "Not Recommended (Negative)":
        filtered_df = filtered_df[filtered_df['recommended'] == False]
        
    if theme_filter != "All":
        col_map = {
            "Performance": "is_performance",
            "Monetization": "is_monetization",
            "Bugs": "is_bugs",
            "Grind": "is_grind",
            "Balance": "is_balance",
            "Matchmaking": "is_matchmaking"
        }
        filtered_df = filtered_df[filtered_df[col_map[theme_filter]] == 1]
        
    if search_query:
        filtered_df = filtered_df[filtered_df['review_text'].str.contains(search_query, case=False, na=False)]
        
    st.write(f"Showing {len(filtered_df)} matching reviews of {len(all_reviews)} total.")
    
    # Display reviews list
    for idx, r in filtered_df.head(50).iterrows(): # Cap display at 50 for performance
        sent_color = "#10b981" if r['recommended'] else "#ef4444"
        sent_text = "RECOMMENDED" if r['recommended'] else "NOT RECOMMENDED"
        
        # Build badges
        badges = []
        if r['is_performance'] == 1: badges.append(('<span class="tag-badge" style="background:#7f1d1d; color:#fca5a5;">PERFORMANCE</span>'))
        if r['is_monetization'] == 1: badges.append(('<span class="tag-badge" style="background:#7c2d12; color:#ffedd5;">MONETIZATION</span>'))
        if r['is_bugs'] == 1: badges.append(('<span class="tag-badge" style="background:#1e3a8a; color:#bfdbfe;">BUG</span>'))
        if r['is_grind'] == 1: badges.append(('<span class="tag-badge" style="background:#064e3b; color:#a7f3d0;">GRIND</span>'))
        if r['is_balance'] == 1: badges.append(('<span class="tag-badge" style="background:#701a75; color:#fbcfe8;">BALANCE</span>'))
        if r['is_matchmaking'] == 1: badges.append(('<span class="tag-badge" style="background:#312e81; color:#c7d2fe;">MATCHMAKING</span>'))
        
        badge_html = " ".join(badges)
        
        st.markdown(f"""
        <div class="review-row">
            <div style="display:flex; justify-content:space-between; margin-bottom: 8px;">
                <span style="color: {sent_color}; font-weight:700; font-size:0.85rem;">{sent_text}</span>
                <span style="font-size:0.8rem; color:#64748b;">
                    Playtime: <strong>{r['playtime_forever']:.1f}h</strong> | Helpful: <strong>{int(r['votes_up'])}</strong>
                </span>
            </div>
            <p style="font-size:0.95rem; margin: 4px 0; color:#cbd5e1; line-height: 1.4;">"{r['review_text']}"</p>
            <div style="margin-top: 6px;">{badge_html}</div>
        </div>
        """, unsafe_allow_html=True)
else:
    st.info("No reviews available to explore. Please run the ingestion pipeline.")

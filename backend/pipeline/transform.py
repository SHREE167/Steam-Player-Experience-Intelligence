import os
import duckdb

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'steam_analytics.db')

def run_sql_transformations():
    """Executes DuckDB SQL queries to transform raw reviews and NLP annotations into game-level KPI tables."""
    print("Running SQL data transformation layer in DuckDB...")
    
    with duckdb.connect(DB_PATH) as conn:
        # Verify if reviews_cleaned exists. If not, create a placeholder view/table to avoid query crashes
        # reviews_cleaned is populated by models.py, but we define schemas here defensively
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews_cleaned (
                review_id VARCHAR PRIMARY KEY,
                sentiment_score DOUBLE,
                is_performance INTEGER,
                is_monetization INTEGER,
                is_bugs INTEGER,
                is_grind INTEGER,
                is_balance INTEGER,
                is_matchmaking INTEGER
            )
        """)
        
        # 1. Rebuild game_daily_kpis table using window functions and CTEs
        print("Computing daily KPIs and moving averages...")
        conn.execute("DROP TABLE IF EXISTS game_daily_kpis")
        conn.execute("""
            CREATE TABLE game_daily_kpis AS
            WITH daily_raw AS (
                SELECT 
                    r.appid,
                    CAST(to_timestamp(CAST(r.timestamp_created AS BIGINT)) AS DATE) as review_date,
                    COUNT(*) as review_count,
                    SUM(CASE WHEN r.recommended = TRUE THEN 1 ELSE 0 END) as positive_count,
                    SUM(CASE WHEN r.recommended = FALSE THEN 1 ELSE 0 END) as negative_count,
                    -- Playtime details
                    AVG(r.playtime_forever) as avg_playtime,
                    -- Veteran stats (playtime > 50 hours)
                    SUM(CASE WHEN r.playtime_forever > 50.0 THEN 1 ELSE 0 END) as veteran_count,
                    SUM(CASE WHEN r.playtime_forever > 50.0 AND r.recommended = FALSE THEN 1 ELSE 0 END) as veteran_churn_count,
                    -- Complaint categories from NLP predictions (only counted on negative reviews to prevent ratios > 100%)
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_performance ELSE 0 END), 0) as perf_complaints,
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_monetization ELSE 0 END), 0) as money_complaints,
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_bugs ELSE 0 END), 0) as bug_complaints,
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_grind ELSE 0 END), 0) as grind_complaints,
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_balance ELSE 0 END), 0) as balance_complaints,
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_matchmaking ELSE 0 END), 0) as mm_complaints
                FROM raw_reviews r
                LEFT JOIN reviews_cleaned c ON r.review_id = c.review_id
                GROUP BY r.appid, review_date
            ),
            daily_metrics AS (
                SELECT 
                    appid,
                    review_date,
                    review_count,
                    positive_count,
                    negative_count,
                    avg_playtime,
                    veteran_count,
                    veteran_churn_count,
                    perf_complaints,
                    money_complaints,
                    bug_complaints,
                    grind_complaints,
                    balance_complaints,
                    mm_complaints,
                    -- Daily ratios
                    CAST(positive_count AS DOUBLE) / NULLIF(review_count, 0) as daily_pos_rate,
                    CAST(perf_complaints AS DOUBLE) / NULLIF(negative_count, 0) as perf_neg_ratio,
                    CAST(money_complaints AS DOUBLE) / NULLIF(negative_count, 0) as money_neg_ratio,
                    CAST(bug_complaints AS DOUBLE) / NULLIF(negative_count, 0) as bug_neg_ratio,
                    CAST(grind_complaints AS DOUBLE) / NULLIF(negative_count, 0) as grind_neg_ratio,
                    CAST(balance_complaints AS DOUBLE) / NULLIF(negative_count, 0) as balance_neg_ratio,
                    CAST(mm_complaints AS DOUBLE) / NULLIF(negative_count, 0) as mm_neg_ratio
                FROM daily_raw
            )
            SELECT 
                *,
                -- Rolling 7-day metrics using SQL window functions
                AVG(daily_pos_rate) OVER (
                    PARTITION BY appid 
                    ORDER BY review_date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rolling_7d_pos_rate,
                SUM(review_count) OVER (
                    PARTITION BY appid 
                    ORDER BY review_date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rolling_7d_review_volume,
                AVG(perf_complaints) OVER (
                    PARTITION BY appid 
                    ORDER BY review_date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rolling_7d_perf_complaints,
                AVG(money_complaints) OVER (
                    PARTITION BY appid 
                    ORDER BY review_date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rolling_7d_money_complaints,
                AVG(bug_complaints) OVER (
                    PARTITION BY appid 
                    ORDER BY review_date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rolling_7d_bug_complaints,
                AVG(grind_complaints) OVER (
                    PARTITION BY appid 
                    ORDER BY review_date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rolling_7d_grind_complaints,
                AVG(balance_complaints) OVER (
                    PARTITION BY appid 
                    ORDER BY review_date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rolling_7d_balance_complaints,
                AVG(mm_complaints) OVER (
                    PARTITION BY appid 
                    ORDER BY review_date 
                    ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
                ) as rolling_7d_mm_complaints
            FROM daily_metrics
        """)

        # 2. Build game_kpi_summary for dashboard reporting
        print("Computing overall game summary metrics...")
        conn.execute("DROP TABLE IF EXISTS game_kpi_summary")
        conn.execute("""
            CREATE TABLE game_kpi_summary AS
            WITH overall_stats AS (
                SELECT 
                    r.appid,
                    COUNT(*) as total_reviews,
                    SUM(CASE WHEN r.recommended = TRUE THEN 1 ELSE 0 END) as positive_reviews,
                    SUM(CASE WHEN r.recommended = FALSE THEN 1 ELSE 0 END) as negative_reviews,
                    AVG(r.playtime_forever) as avg_playtime,
                    
                    -- Veteran stats (veteran defined as > 50 hours of playtime)
                    SUM(CASE WHEN r.playtime_forever > 50.0 THEN 1 ELSE 0 END) as total_veterans,
                    SUM(CASE WHEN r.playtime_forever > 50.0 AND r.recommended = FALSE THEN 1 ELSE 0 END) as veteran_churners,
                    
                    -- Complaint themes totals (only among negative reviews to get mathematically correct ratios)
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_performance ELSE 0 END), 0) as total_perf_complaints,
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_monetization ELSE 0 END), 0) as total_money_complaints,
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_bugs ELSE 0 END), 0) as total_bug_complaints,
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_grind ELSE 0 END), 0) as total_grind_complaints,
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_balance ELSE 0 END), 0) as total_balance_complaints,
                    COALESCE(SUM(CASE WHEN r.recommended = FALSE THEN c.is_matchmaking ELSE 0 END), 0) as total_mm_complaints
                FROM raw_reviews r
                LEFT JOIN reviews_cleaned c ON r.review_id = c.review_id
                GROUP BY r.appid
            ),
            recency_stats AS (
                -- Calculate positive rate in last 7 days vs last 30 days to detect trend shifts
                SELECT 
                    r.appid,
                    SUM(CASE WHEN r.timestamp_created >= ((SELECT COALESCE(MAX(timestamp_created), epoch(now())) FROM raw_reviews) - 7 * 86400) AND r.recommended = TRUE THEN 1 ELSE 0 END) as pos_reviews_7d,
                    COUNT(CASE WHEN r.timestamp_created >= ((SELECT COALESCE(MAX(timestamp_created), epoch(now())) FROM raw_reviews) - 7 * 86400) THEN 1 ELSE NULL END) as total_reviews_7d,
                    SUM(CASE WHEN r.timestamp_created >= ((SELECT COALESCE(MAX(timestamp_created), epoch(now())) FROM raw_reviews) - 30 * 86400) AND r.recommended = TRUE THEN 1 ELSE 0 END) as pos_reviews_30d,
                    COUNT(CASE WHEN r.timestamp_created >= ((SELECT COALESCE(MAX(timestamp_created), epoch(now())) FROM raw_reviews) - 30 * 86400) THEN 1 ELSE NULL END) as total_reviews_30d
                FROM raw_reviews r
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
                CAST(o.positive_reviews AS DOUBLE) / o.total_reviews as overall_pos_rate,
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
            JOIN overall_stats o ON g.appid = o.appid
            LEFT JOIN recency_stats r ON g.appid = r.appid
        """)
        
    print("SQL data transformations completed successfully.")

if __name__ == '__main__':
    run_sql_transformations()

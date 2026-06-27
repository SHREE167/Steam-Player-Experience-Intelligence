import os
import argparse
import sys
import duckdb

# Add backend directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.pipeline.ingest import fetch_reviews_last_90_days, get_db_connection, setup_db_schemas, DB_PATH, search_steam_games
from backend.pipeline.models import train_and_run_nlp_models, compute_player_experience_risk_index
from backend.pipeline.transform import run_sql_transformations

def run_pipeline(appid=553850, clean_db=False):
    """Coordinates and runs the ingestion, cleaning, NLP modeling, SQL transformation, and PERI scoring."""
    print("======================================================================")
    print("   STEAM PLAYER EXPERIENCE INTELLIGENCE PIPELINE RUNNER   ")
    print("======================================================================\n")
    
    # 1. Ingestion / Data Setup
    if clean_db:
        print("Cleaning DuckDB database to remove old or synthetic datasets...")
        if os.path.exists(DB_PATH):
            try:
                os.remove(DB_PATH)
                print("Database file deleted for a clean run.")
            except Exception as e:
                print(f"Could not delete database file, clearing tables instead: {e}")
                conn = get_db_connection()
                conn.execute("DROP TABLE IF EXISTS raw_reviews")
                conn.execute("DROP TABLE IF EXISTS raw_games")
                conn.execute("DROP TABLE IF EXISTS reviews_cleaned")
                conn.execute("DROP TABLE IF EXISTS game_daily_kpis")
                conn.execute("DROP TABLE IF EXISTS game_kpi_summary")
                conn.close()
                
    conn = get_db_connection()
    setup_db_schemas(conn)
    conn.close()
    
    # Ingest real 90-day reviews
    print(f"Scraping real reviews of the last 90 days for AppID {appid} from Steam API...")
    fetch_reviews_last_90_days(appid)
        
    # 2. Run NLP Theme & Sentiment Classifier
    print("\n--- PHASE 2: ML & NLP CLASSIFICATION ---")
    train_and_run_nlp_models()
    
    # 3. Run SQL Transformation Layer (computes daily KPIs & moving averages)
    print("\n--- PHASE 3: SQL DATA TRANSFORMATION (DuckDB) ---")
    run_sql_transformations()
    
    # 4. Compute Player Experience Risk Index (PERI)
    print("\n--- PHASE 4: RISK SCORING ENGINE ---")
    compute_player_experience_risk_index()
    
    print("\n======================================================================")
    print("                  PIPELINE RUN COMPLETED SUCCESSFULLY                 ")
    print("======================================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Steam Player Experience Intelligence Ingestion & Model Pipeline")
    parser.add_argument('--appid', type=int, default=553850, help='Steam AppID to ingest real 30-day reviews for (default: 553850 for Helldivers 2)')
    parser.add_argument('--clean', action='store_true', help='Clean the database before fetching')
    parser.add_argument('--test', action='store_true', help='Run a quick check on the database')
    
    args = parser.parse_args()
    
    if args.test:
        print("Checking database files...")
        db_exists = os.path.exists(DB_PATH)
        print(f"Database exists: {db_exists}")
        if db_exists:
            conn = get_db_connection()
            tables = conn.execute("SHOW TABLES").fetchall()
            print("Tables found:")
            for t in tables:
                cnt = conn.execute(f"SELECT COUNT(*) FROM {t[0]}").fetchone()[0]
                print(f" - {t[0]}: {cnt} rows")
            conn.close()
        else:
            print("No database found. Run pipeline without --test first.")
    else:
        run_pipeline(appid=args.appid, clean_db=args.clean)

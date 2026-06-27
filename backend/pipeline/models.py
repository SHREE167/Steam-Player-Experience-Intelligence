import os
import re
import duckdb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'steam_analytics.db')

# Hand-crafted list of common English stopwords to avoid external downloads issues
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves",
    "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up",
    "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don",
    "should", "now"
])

LEXICON = {
    'is_performance': [
        r'\bfps\b', r'\bcrash(es|ed|ing)?\b', r'\blag(ged|ging|s)?\b', r'\bstutter(ing|s|ed)?\b', 
        r'\bfreeze\b', r'\bfreezing\b', r'\bperformance\b', r'\boptimization\b', r'\boptimize\b',
        r'\bgpu\b', r'\bcpu\b', r'\bframe rate\b', r'\bframe drop(s)?\b', r'\bunplayable\b'
    ],
    'is_monetization': [
        r'\bbattle\s?pass\b', r'\bmicrotransaction(s)?\b', r'\bgreedy\b', r'\bmoney\b', 
        r'\bwallet\b', r'\bcash\s?grab\b', r'\bpricing\b', r'\bpaywall\b', r'\bprice(s)?\b', 
        r'\bshop\b', r'\bstore\b', r'\bcost\b', r'\bpay\s?to\s?win\b', r'\bp2w\b', r'\bskin(s)?\b'
    ],
    'is_bugs': [
        r'\bbugs?(gy)?\b', r'\bglitch(es|ed|ing|y)?\b', r'\bbroken\b', r'\berror(s)?\b', 
        r'\bexploit(s)?\b', r'\bpatched\b', r'\bphysics\b', r'\bvisual glitch\b', r'\bclipping\b',
        r'\btexture(s)?\b', r'\bblack screen\b', r'\bfreeze\b'
    ],
    'is_grind': [
        r'\bgrind(y|ing|s)?\b', r'\bprogression\b', r'\bxp\b', r'\blevel(ing|up)?\b', 
        r'\btedious\b', r'\brepetitive\b', r'\bfarm(ing)?\b', r'\bmaterials\b', r'\bunlock(s|ed|ing)?\b'
    ],
    'is_balance': [
        r'\bbalance\b', r'\bbalanced\b', r'\bnerf(ed|s|ing)?\b', r'\bbuff(ed|s|ing)?\b', 
        r'\boverpowered\b', r'\bop\b', r'\bweapon balance\b', r'\bskills\b', r'\bmeta\b', 
        r'\bunfair\b', r'\bbroken combat\b'
    ],
    'is_matchmaking': [
        r'\bmatchmaking\b', r'\bping\b', r'\bserver(s)?\b', r'\bqueue(s)?\b', 
        r'\bconnection\b', r'\bdisconnect(ed|ing|s)?\b', r'\blobby\b', r'\bnetcode\b', 
        r'\bdisconnects\b', r'\bjoin lobby\b', r'\bhigh latency\b'
    ]
}

def clean_text(text: str) -> str:
    """Preprocess review text (lowercase, remove punctuation/numbers, remove stopwords)."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [w for w in words if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)

def weak_supervision_label(text: str) -> dict:
    """Assigns soft labels to a review text based on regex lexicons."""
    labels = {}
    lower_text = text.lower()
    for category, regex_list in LEXICON.items():
        matched = 0
        for regex in regex_list:
            if re.search(regex, lower_text):
                matched = 1
                break
        labels[category] = matched
    return labels

def train_and_run_nlp_models():
    """Trains a multi-label text classifier using Weak Supervision and saves predictions to DuckDB."""
    print("Loading review text from DuckDB...")
    conn = duckdb.connect(DB_PATH)
    
    # Read reviews
    df = conn.execute("SELECT review_id, review_text, recommended FROM raw_reviews").fetchdf()
    if df.empty:
        print("No reviews found in the database. Ingestion must run first.")
        conn.close()
        return
        
    print(f"Loaded {len(df)} reviews. Preprocessing text...")
    df['clean_text'] = df['review_text'].apply(clean_text)
    
    # Generate weak labels
    print("Generating weak labels using regex lexicons...")
    weak_labels = df['review_text'].apply(weak_supervision_label).tolist()
    labels_df = pd.DataFrame(weak_labels)
    
    # Combine data
    df = pd.concat([df, labels_df], axis=1)
    
    # Sentiment score logic:
    # We assign a baseline sentiment score based on recommendation (1 = positive, 0 = negative),
    # but we refine it based on text indicators. If it's recommended but has complaint keywords, score drops.
    # If it's negative but has praise, score increases.
    df['sentiment_score'] = df['recommended'].apply(lambda x: 0.8 if x else 0.2)
    # Simple modifier: subtract 0.1 for each complaint theme detected
    complaint_sum = df[list(LEXICON.keys())].sum(axis=1)
    df['sentiment_score'] = df['sentiment_score'] - (complaint_sum * 0.1)
    df['sentiment_score'] = df['sentiment_score'].clip(0.0, 1.0)
    
    # Train ML models to generalize beyond simple keywords
    categories = list(LEXICON.keys())
    
    # Split training and testing for validation output
    X = df['clean_text']
    y = df[categories]
    
    # Ensure we have at least some reviews to train
    if len(df) > 100:
        try:
            print("Fitting TF-IDF Vectorizer...")
            # Ignore extremely rare words
            vectorizer = TfidfVectorizer(max_features=1500, min_df=2)
            X_vec = vectorizer.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
            
            print("Training Multi-Label Classifier (Logistic Regression)...")
            # balanced weight to deal with class imbalances in specific complaint types
            clf = MultiOutputClassifier(LogisticRegression(class_weight='balanced', C=1.0, max_iter=500))
            clf.fit(X_train, y_train)
            
            # Test performance
            y_pred = clf.predict(X_test)
            
            print("\n--- ML Theme Classifier Performance Report (Evaluation Set) ---")
            for i, category in enumerate(categories):
                print(f"Category: {category}")
                # Ensure evaluation runs smoothly even if test set misses some positive cases
                try:
                    print(classification_report(y_test.iloc[:, i], y_pred[:, i], zero_division=0))
                except Exception as e:
                    print(f"Error printing report: {e}")
                    
            # Generate predictions for all reviews
            print("Running predictions across full dataset to generalize labeling...")
            y_all_pred = clf.predict(X_vec)
            
            # Overlay predictions (if a model predicts a category, or if it matched the lexicon, we flag it.
            # This hybrid approach gives us high recall and precision).
            for i, category in enumerate(categories):
                df[category] = np.maximum(df[category].values, y_all_pred[:, i])
        except Exception as e:
            print(f"ML Model training failed: {e}. Falling back to rule-based labeling.")
            
    else:
        print("Dataset too small for reliable ML training. Falling back to pure rule-based labels.")
        # We already computed df[category] based on weak labels, which is our fallback.
    
    # Re-insert results into reviews_cleaned table
    print("Writing processed reviews back to DuckDB table 'reviews_cleaned'...")
    conn.execute("DROP TABLE IF EXISTS reviews_cleaned")
    conn.execute("""
        CREATE TABLE reviews_cleaned (
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
    
    clean_records = df[['review_id', 'sentiment_score'] + categories].values.tolist()
    conn.executemany("""
        INSERT INTO reviews_cleaned VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, clean_records)
    
    conn.commit()
    conn.close()
    print("NLP and theme classification complete.")

def compute_player_experience_risk_index():
    """Computes the Player Experience Risk Index (PERI) and stores it in game_kpi_summary."""
    print("Calculating Player Experience Risk Index (PERI)...")
    conn = duckdb.connect(DB_PATH)
    
    # Read aggregated KPIs from game_kpi_summary
    kpis = conn.execute("SELECT * FROM game_kpi_summary").fetchdf()
    if kpis.empty:
        print("No game KPI summaries found. SQL transformations must run first.")
        conn.close()
        return
        
    peri_scores = []
    risk_tiers = []
    
    for idx, row in kpis.iterrows():
        # Component 1: Sentiment Risk (Ratio of negative reviews, 0-100)
        neg_rate = 1.0 - row['overall_pos_rate']
        sentiment_risk = neg_rate * 100
        
        # Component 2: Veteran Churn Risk (0-100)
        # Veterans are players with >50 hours playtime who have left a negative review
        veteran_churn_rate = row['veteran_churn_rate']
        if pd.isna(veteran_churn_rate) or row['total_veterans'] == 0:
            veteran_churn_risk = sentiment_risk # fallback if no veteran data
        else:
            veteran_churn_risk = veteran_churn_rate * 100
            
        # Component 3: Recent Sentiment Decline (0-100)
        # Compare last 7 days positive rate to last 30 days. If 7d is worse, calculate decay.
        pos_7d = row['pos_rate_7d']
        pos_30d = row['pos_rate_30d']
        
        if pd.isna(pos_7d) or pd.isna(pos_30d):
            momentum_risk = 0.0
        else:
            decline = pos_30d - pos_7d
            if decline > 0:
                # If positive rate dropped by 20% (0.2), risk is 0.2 * 200 = 40. Cap at 100.
                momentum_risk = min(decline * 200, 100.0)
            else:
                # If recovering/improving, we can reward them by lowering risk (up to -15)
                momentum_risk = max(decline * 50, -15.0)
                
        # Component 4: Complaint Severity Index (0-100)
        # Sum of complaint types, weighted by severity.
        # Performance/Crashes (0.35), Monetization (0.25), Matchmaking (0.20), Bugs (0.15), Balance (0.10), Grind (0.10)
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
        # Scale to 0-100
        severity_risk = min(severity_score * 100 * 2.0, 100.0) # Multiply to make it scale realistically
        
        # Combine into final PERI (Player Experience Risk Index)
        # Formula weights: Sentiment(35%), Veteran Churn(25%), Momentum(15%), Severity(25%)
        peri = (
            sentiment_risk * 0.35 +
            veteran_churn_risk * 0.25 +
            momentum_risk * 0.15 +
            severity_risk * 0.25
        )
        
        # Adjust/clamp to 0-100 range
        peri = max(0.0, min(100.0, peri))
        peri_scores.append(round(peri, 1))
        
        # Assign Risk Tier
        if peri >= 75.0:
            risk_tiers.append("CRITICAL")
        elif peri >= 55.0:
            risk_tiers.append("HIGH")
        elif peri >= 35.0:
            risk_tiers.append("MODERATE")
        elif peri >= 15.0:
            risk_tiers.append("STABLE")
        else:
            risk_tiers.append("HEALTHY")
            
    # Add columns to DuckDB game_kpi_summary
    kpis['peri_score'] = peri_scores
    kpis['risk_tier'] = risk_tiers
    
    print("\n--- Game Player Experience Risk Index (PERI) Scores ---")
    for idx, row in kpis.iterrows():
        print(f"Game: {row['title']} | PERI: {row['peri_score']} | Tier: {row['risk_tier']}")
        
    # Write back updated table to DuckDB
    # Update DuckDB by copying dataframe
    conn.execute("DROP TABLE IF EXISTS game_kpi_summary_temp")
    conn.register("df_temp", kpis)
    conn.execute("CREATE TABLE game_kpi_summary_temp AS SELECT * FROM df_temp")
    conn.execute("DROP TABLE IF EXISTS game_kpi_summary")
    conn.execute("ALTER TABLE game_kpi_summary_temp RENAME TO game_kpi_summary")
    
    conn.commit()
    conn.close()
    print("PERI scores successfully integrated into database.")

if __name__ == '__main__':
    train_and_run_nlp_models()
    compute_player_experience_risk_index()

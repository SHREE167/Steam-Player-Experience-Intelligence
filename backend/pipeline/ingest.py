import os
import re
import time
import datetime
import requests
import duckdb

# Database path
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'steam_analytics.db')

# Standard browser headers to bypass Steam Store API blockages
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9'
}

def get_db_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return duckdb.connect(DB_PATH)

def setup_db_schemas(conn):
    """Create the raw tables if they do not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_games (
            appid INTEGER PRIMARY KEY,
            title VARCHAR,
            developer VARCHAR,
            publisher VARCHAR,
            genres VARCHAR,
            price DOUBLE,
            release_date VARCHAR,
            header_image VARCHAR
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS raw_reviews (
            review_id VARCHAR PRIMARY KEY,
            appid INTEGER,
            review_text VARCHAR,
            recommended BOOLEAN,
            playtime_forever DOUBLE, -- in hours
            votes_up INTEGER,
            timestamp_created INTEGER
        )
    """)
    print("Database schemas initialized.")

def search_steam_games(query: str) -> tuple:
    """Search for games by name using Steam's storesearch API (with User-Agent headers). Returns (results, error_message)."""
    # Sanitize query by removing TM / Registered / Copyright symbols
    sanitized_query = re.sub(r'[™®©]', ' ', query).strip()
    sanitized_query = re.sub(r'\s+', ' ', sanitized_query) # remove duplicate spaces
    
    url = f"https://store.steampowered.com/api/storesearch/?term={requests.utils.quote(sanitized_query)}&l=english&cc=US"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code != 200:
            return [], f"HTTP Error {response.status_code}"
            
        data = response.json()
        if data and 'items' in data:
            items = data.get('items', [])
            results = []
            for item in items:
                # Replace residual TM character representations in names returned
                title = item['name'].replace('', '').replace('™', '').replace('®', '').strip()
                results.append({
                    'appid': int(item['id']),
                    'title': title,
                    'header_image': f"https://shared.akamai.steamstatic.com/store_item_assets/steam/apps/{item['id']}/header.jpg"
                })
            return results, None
        return [], "No items found in Steam store response"
    except Exception as e:
        print(f"Error searching Steam store for '{query}': {e}")
        return [], str(e)

def fetch_game_metadata(appid: int) -> dict:
    """Fetch game metadata from Steam Store API with headers."""
    url = f"https://store.steampowered.com/api/appdetails?appids={appid}"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data and str(appid) in data and data[str(appid)].get('success', False):
                game_data = data[str(appid)]['data']
                genres = ", ".join([g['description'] for g in game_data.get('genres', [])])
                price_info = game_data.get('price_overview', {})
                price = price_info.get('final_formatted', '$0.00') if price_info else 'Free'
                if isinstance(price, str) and price.startswith('$'):
                    try:
                        price = float(price.replace('$', '').replace(' ', ''))
                    except ValueError:
                        price = 0.0
                else:
                    price = 0.0
                
                title = game_data.get('name', 'Unknown Game').replace('', '').replace('™', '').replace('®', '').strip()
                return {
                    'appid': appid,
                    'title': title,
                    'developer': ", ".join(game_data.get('developers', ['Unknown'])),
                    'publisher': ", ".join(game_data.get('publishers', ['Unknown'])),
                    'genres': genres,
                    'price': price if isinstance(price, float) else 0.0,
                    'release_date': game_data.get('release_date', {}).get('date', 'Unknown'),
                    'header_image': game_data.get('header_image', '')
                }
    except Exception as e:
        print(f"Error fetching metadata for AppID {appid}: {e}")
    
    return {
        'appid': appid,
        'title': f"Steam App {appid}",
        'developer': "Unknown Developer",
        'publisher': "Unknown Publisher",
        'genres': "Action, Adventure",
        'price': 0.0,
        'release_date': "Unknown",
        'header_image': ""
    }

def fetch_reviews_last_90_days(appid: int, max_limit: int = 3000) -> int:
    """Fetch real reviews from the last 90 days from the Steam API (with headers) and write to DuckDB."""
    conn = get_db_connection()
    setup_db_schemas(conn)
    
    # Save game metadata first
    meta = fetch_game_metadata(appid)
    conn.execute("""
        INSERT OR REPLACE INTO raw_games VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (meta['appid'], meta['title'], meta['developer'], meta['publisher'], 
          meta['genres'], meta['price'], meta['release_date'], meta['header_image']))
    
    # Define 90-day time threshold
    now_ts = int(time.time())
    cutoff_ts = now_ts - (90 * 86400) # 90 days in seconds
    
    cursor = '*'
    reviews_fetched = 0
    failures = 0
    hit_cutoff = False
    
    print(f"Starting Ingestion for {meta['title']} (AppID: {appid}).")
    print(f"Goal: Fetch reviews from last 90 days (Cutoff timestamp: {cutoff_ts} / {datetime.datetime.fromtimestamp(cutoff_ts)})")
    
    while reviews_fetched < max_limit and failures < 3 and not hit_cutoff:
        url = f"https://store.steampowered.com/appreviews/{appid}?json=1&cursor={requests.utils.quote(cursor)}&language=english&filter=recent&num_per_page=100"
        try:
            response = requests.get(url, headers=HEADERS, timeout=10)
            if response.status_code != 200:
                failures += 1
                time.sleep(2)
                continue
                
            data = response.json()
            if not data or 'reviews' not in data or len(data['reviews']) == 0:
                print("No more reviews returned by Steam API.")
                break
                
            cursor = data.get('cursor', '*')
            reviews = data['reviews']
            
            inserted_count = 0
            for r in reviews:
                review_id = str(r['recommendationid'])
                review_text = r.get('review', '')
                recommended = r.get('voted_up', True)
                playtime = r.get('author', {}).get('playtime_forever', 0) / 60.0 # convert min to hours
                votes_up = r.get('votes_up', 0)
                timestamp = r.get('timestamp_created', int(time.time()))
                
                # Check 90-day boundary
                if timestamp < cutoff_ts:
                    print(f"Reached review older than 90 days ({datetime.datetime.fromtimestamp(timestamp)}). Stopping fetch.")
                    hit_cutoff = True
                    break
                
                # Skip reviews with empty text
                if not review_text.strip():
                    continue
                
                conn.execute("""
                    INSERT OR REPLACE INTO raw_reviews VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (review_id, appid, review_text, recommended, playtime, votes_up, timestamp))
                reviews_fetched += 1
                inserted_count += 1
                
                if reviews_fetched >= max_limit:
                    break
            
            print(f"Ingested chunk: {inserted_count} reviews in this query. Total for game: {reviews_fetched}")
            
            if hit_cutoff or reviews_fetched >= max_limit:
                break
                
            time.sleep(1) # Rate limit protection
            
        except Exception as e:
            print(f"Error fetching reviews chunk: {e}")
            failures += 1
            time.sleep(2)
            
    conn.close()
    print(f"Completed ingestion. Total real reviews fetched: {reviews_fetched}")
    return reviews_fetched

if __name__ == '__main__':
    # Default to fetching Helldivers 2 reviews if run directly
    fetch_reviews_last_90_days(553850)

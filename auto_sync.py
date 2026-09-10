import os
import sys
import io
import time
import requests
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import torch
import faiss
from sqlalchemy import create_engine, text

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["HF_HOME"] = "./.hf_cache"

DB_URL = "postgresql://postgres:Jadequest%403009@3.111.57.216:5432/jaxmart_db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_FILE = os.path.join(BASE_DIR, "visual_search_index.faiss")
MAPPING_FILE = os.path.join(BASE_DIR, "image_mapping.pkl")

def extract_listing_id(path_str):
    if not path_str:
        return None
    parts = os.path.normpath(str(path_str)).replace('/', '\\').split('\\')
    for part in reversed(parts):
        if len(part) == 36 and part.count('-') == 4:
            return part
    if len(parts) >= 2:
        return parts[-2]
    return str(path_str)

def get_already_indexed_listing_ids():
    """Returns a set of listing_ids currently present in the FAISS mapping."""
    if not os.path.exists(MAPPING_FILE):
        return set()
    try:
        with open(MAPPING_FILE, 'rb') as f:
            mapping = pickle.load(f)
        return set(extract_listing_id(p) for p in mapping if p)
    except Exception as e:
        print(f"Error reading mapping file: {e}")
        return set()

def fetch_image_in_memory(image_url):
    """Fetches image bytes directly into RAM (Zero Disk Download)."""
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        resp = requests.get(image_url, headers=headers, timeout=8)
        if resp.status_code == 200 and len(resp.content) > 100:
            img = Image.open(io.BytesIO(resp.content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
    except Exception:
        pass
    return None

def incremental_sync():
    """
    Zero-Download Auto-Sync:
    Checks for new products in DB, streams their images directly in RAM,
    computes CLIP vector embeddings, and appends to FAISS in 1-2 seconds.
    Zero bytes written to hard disk!
    """
    indexed_ids = get_already_indexed_listing_ids()
    
    # Fetch all listings from DB
    engine = create_engine(DB_URL)
    query = """
    SELECT DISTINCT ON (l.id)
        l.id as listing_id,
        l.title,
        lm.url as image_url
    FROM listings l
    INNER JOIN listing_media lm ON l.id = lm."listingId"
    WHERE lm.url IS NOT NULL AND lm.url != '' AND lm.url NOT ILIKE '%file:///%'
    ORDER BY l.id, lm."isPrimary" DESC, lm."sortOrder" ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
        
    new_rows = df[~df['listing_id'].astype(str).isin(indexed_ids)]
    
    if new_rows.empty:
        return 0
        
    print(f"--- In-Memory Sync: Found {len(new_rows)} new products in DB. Processing in RAM ---")
    
    from visual_search import VisualSearchEngine
    engine_vs = VisualSearchEngine(index_file=INDEX_FILE, mapping_file=MAPPING_FILE)
    engine_vs.load_model()
    engine_vs.load_index()
    
    new_embeddings = []
    new_ids = []
    
    for _, row in new_rows.iterrows():
        lid = str(row['listing_id'])
        url = str(row['image_url'])
        
        img = fetch_image_in_memory(url)
        if img is not None:
            try:
                emb = engine_vs.get_image_embedding(img)
                new_embeddings.append(emb)
                new_ids.append(lid) # Store listing UUID directly
            except Exception as e:
                pass
                
    if new_embeddings:
        matrix = np.vstack(new_embeddings)
        engine_vs.index.add(matrix)
        engine_vs.image_paths.extend(new_ids)
        
        # Save updated FAISS index and UUID mapping
        faiss.write_index(engine_vs.index, INDEX_FILE)
        with open(MAPPING_FILE, 'wb') as f:
            pickle.dump(engine_vs.image_paths, f)
            
        print(f"Successfully added {len(new_ids)} new products to FAISS via In-Memory Streaming!")
        return len(new_ids)
        
    return 0

def run_scheduler(interval_seconds=180):
    """Runs incremental sync periodically in the background (default every 3 minutes)."""
    print(f"Zero-Download Auto-Sync Scheduler started! (Checking every {interval_seconds}s)")
    while True:
        try:
            incremental_sync()
        except Exception as e:
            print(f"Error during auto sync: {e}")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--watch":
        run_scheduler(interval_seconds=180)
    else:
        incremental_sync()

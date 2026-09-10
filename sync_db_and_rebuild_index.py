import os
import sys
import time
import requests
import concurrent.futures
from PIL import Image
import pandas as pd
from sqlalchemy import create_engine, text

# Set environment variables for huggingface and duplicate libs
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["HF_HOME"] = "./.hf_cache"

DB_URL = "postgresql://postgres:Jadequest%403009@3.111.57.216:5432/jaxmart_db"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_DIR = os.path.join(BASE_DIR, "products")
os.makedirs(PRODUCTS_DIR, exist_ok=True)

def fetch_db_listings():
    print("Connecting to PostgreSQL Database...")
    engine = create_engine(DB_URL)
    query = """
    SELECT DISTINCT ON (l.id)
        l.id as listing_id,
        l.title,
        lm.url as image_url
    FROM listings l
    INNER JOIN listing_media lm ON l.id = lm."listingId"
    WHERE lm.url IS NOT NULL AND lm.url != ''
    ORDER BY l.id, lm."isPrimary" DESC, lm."sortOrder" ASC
    """
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    print(f"Total listings with images found in Database: {len(df)}")
    return df

def download_single_image(row_tuple):
    listing_id, image_url = row_tuple
    if not listing_id or not image_url or pd.isna(image_url):
        return None
        
    listing_folder = os.path.join(PRODUCTS_DIR, str(listing_id))
    os.makedirs(listing_folder, exist_ok=True)
    
    # Check if image already exists
    for ext in ['jpg', 'jpeg', 'png', 'webp']:
        existing_file = os.path.join(listing_folder, f"1.{ext}")
        if os.path.exists(existing_file) and os.path.getsize(existing_file) > 100:
            return existing_file
            
    # Download image
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        resp = requests.get(image_url, headers=headers, timeout=10)
        if resp.status_code == 200 and len(resp.content) > 100:
            # Determine extension
            ext = "jpg"
            if ".png" in image_url.lower():
                ext = "png"
            elif ".webp" in image_url.lower():
                ext = "webp"
            elif ".jpeg" in image_url.lower():
                ext = "jpeg"
                
            file_path = os.path.join(listing_folder, f"1.{ext}")
            with open(file_path, "wb") as f:
                f.write(resp.content)
            return file_path
    except Exception as e:
        # print(f"Failed to download {image_url}: {e}")
        pass
        
    return None

def download_missing_images(df):
    tasks = list(zip(df['listing_id'], df['image_url']))
    print(f"Checking & downloading images for {len(tasks)} products...")
    
    downloaded_count = 0
    already_existing = 0
    failed_count = 0
    
    # Check how many exist
    missing_tasks = []
    for lid, url in tasks:
        folder = os.path.join(PRODUCTS_DIR, str(lid))
        exists = False
        if os.path.exists(folder):
            for ext in ['jpg', 'jpeg', 'png', 'webp']:
                f_path = os.path.join(folder, f"1.{ext}")
                if os.path.exists(f_path) and os.path.getsize(f_path) > 100:
                    exists = True
                    break
        if exists:
            already_existing += 1
        else:
            missing_tasks.append((lid, url))
            
    print(f"Already have locally: {already_existing} images.")
    print(f"Downloading {len(missing_tasks)} missing images with multithreading...")
    
    if missing_tasks:
        with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
            results = list(executor.map(download_single_image, missing_tasks))
            for res in results:
                if res:
                    downloaded_count += 1
                else:
                    failed_count += 1
                    
    print(f"Download complete: {downloaded_count} newly downloaded, {already_existing} existing, {failed_count} failed.")

def rebuild_faiss_index():
    print("\nRebuilding FAISS Vector Index using CLIP...")
    from visual_search import VisualSearchEngine
    
    # Remove old index to build a clean complete one
    index_file = os.path.join(BASE_DIR, "visual_search_index.faiss")
    mapping_file = os.path.join(BASE_DIR, "image_mapping.pkl")
    
    if os.path.exists(index_file):
        os.remove(index_file)
    if os.path.exists(mapping_file):
        os.remove(mapping_file)
        
    engine = VisualSearchEngine(index_file=index_file, mapping_file=mapping_file)
    engine.build_index_from_directory(PRODUCTS_DIR, batch_size=64)
    print("FAISS Vector Index successfully rebuilt with all live database products!")

if __name__ == "__main__":
    df = fetch_db_listings()
    download_missing_images(df)
    rebuild_faiss_index()

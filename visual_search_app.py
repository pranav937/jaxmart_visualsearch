import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# Use a local cache directory for HuggingFace to avoid disk space issues while remaining cross-platform compatible
os.environ["HF_HOME"] = "./.hf_cache"

import streamlit as st
import pandas as pd
from PIL import Image as PILImage
from visual_search import VisualSearchEngine
from sqlalchemy import create_engine

# Streamlit UI Configuration
st.set_page_config(page_title="JaxMart Visual Search", page_icon="🔍", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #B0BEC5;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔍 JaxMart Visual Search</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload an image to find similar products in the Live Database</div>', unsafe_allow_html=True)
st.markdown("---")

DB_URL = "postgresql://postgres:Jadequest%403009@3.111.57.216:5432/jaxmart_db"

@st.cache_data
def load_dataset():
    try:
        # Fetch from database
        engine = create_engine(DB_URL)
        query = '''
        SELECT 
            l.id as listing_id,
            l.title as "Product Name",
            c.name as "Category",
            c.name as "Subcategory",
            bp."businessName" as "Company Name",
            addr.city as "Location",
            pd."pricePerUnit" as "Price",
            pd."minOrderQty" as "MOQ",
            l."avgRating" as "Rating",
            l."reviewCount" as "Reviews",
            lm.url as "Image URL"
        FROM listings l
        LEFT JOIN categories c ON l."categoryId" = c.id
        LEFT JOIN product_details pd ON l.id = pd."listingId"
        LEFT JOIN business_profiles bp ON l."sellerId" = bp."userId"
        LEFT JOIN (
            SELECT DISTINCT ON ("userId") "userId", city 
            FROM addresses 
            ORDER BY "userId", "isPrimary" DESC
        ) addr ON l."sellerId" = addr."userId"
        LEFT JOIN (
            SELECT DISTINCT ON ("listingId") "listingId", url 
            FROM listing_media 
            ORDER BY "listingId", "isPrimary" DESC, "sortOrder" ASC
        ) lm ON l.id = lm."listingId"
        '''
        df = pd.read_sql(query, engine)
        if 'Product Name' in df.columns:
            df = df.dropna(subset=['Product Name'])
        return df

    except Exception as e:
        st.error(f"Error loading data from Database: {e}")
        return pd.DataFrame()

# Load Dataset (Once)
df_global = load_dataset()

@st.cache_resource
def load_visual_search_engine():
    try:
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_CACHE = "./.hf_cache"
        os.environ["HF_HOME"] = "./.hf_cache"
        os.environ["TRANSFORMERS_CACHE"] = "./.hf_cache"
        
        engine = VisualSearchEngine()
        if engine.load_index():
            return engine
        return None
    except Exception as e:
        st.error(f"Error loading visual search engine: {e}")
        return None

vs_engine = load_visual_search_engine()

# Silent Automatic Background Sync (No manual clicks needed)
@st.cache_resource
def start_background_auto_sync():
    import threading
    def auto_sync_worker():
        import time
        from auto_sync import incremental_sync
        while True:
            try:
                incremental_sync()
            except Exception:
                pass
            time.sleep(180) # Check every 3 minutes silently
    
    t = threading.Thread(target=auto_sync_worker, daemon=True)
    t.start()
    return True

start_background_auto_sync()

def extract_listing_id(path_str):
    if not path_str:
        return None
    parts = os.path.normpath(str(path_str)).replace('/', '\\').split('\\')
    for part in reversed(parts):
        if len(part) == 36 and part.count('-') == 4:
            return part
    if len(parts) >= 2:
        return parts[-2]
    return None

def render_product_card(row):
    with st.container(border=True):
        img_url = row.get('Image URL', None)
        
        if pd.notna(img_url) and str(img_url).strip() and str(img_url).lower() not in ['none', 'nan', 'n/a']:
            try:
                st.image(str(img_url), use_container_width=True)
            except Exception:
                pass
        
        p_name = row.get('Product Name', 'Unknown')
        if pd.isna(p_name):
            p_name = 'Unknown Product'
        p_name = str(p_name)
        if len(p_name) > 40:
            p_name = p_name[:37] + "..."
            
        st.markdown(f"<h4 style='color: #FF4B2B; margin-bottom: 5px; min-height: 45px;'>{p_name}</h4>", unsafe_allow_html=True)
        
        price_val = row.get('Price', 'N/A')
        price_str = f"₹{price_val}" if pd.notna(price_val) and str(price_val) not in ['nan', 'None', 'N/A'] else "Contact for Price"
        st.markdown(f"**💰 Price:** <span style='color: #4CAF50; font-weight: bold;'>{price_str}</span>", unsafe_allow_html=True)
        
        comp_name = row.get('Company Name', 'N/A')
        st.markdown(f"**🏢 Company:** {comp_name if pd.notna(comp_name) and str(comp_name) != 'None' else 'N/A'}")
        
        loc_val = row.get('Location', 'N/A')
        st.markdown(f"**📍 Location:** {loc_val if pd.notna(loc_val) and str(loc_val) != 'None' else 'N/A'}")
        
        moq_val = row.get('MOQ', 'N/A')
        st.markdown(f"**📦 MOQ:** {moq_val if pd.notna(moq_val) and str(moq_val) != 'None' else 'N/A'}")
        
        rating_val = row.get('Rating', '0')
        rev_val = row.get('Reviews', '0')
        st.markdown(f"**⭐ Rating:** {rating_val if pd.notna(rating_val) else '0'} ({rev_val if pd.notna(rev_val) else '0'} reviews)")
        st.markdown("<br>", unsafe_allow_html=True)

uploaded_image = st.file_uploader("Upload a product image", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None and vs_engine is not None:
    st.image(uploaded_image, width=300, caption="Your Uploaded Image")
    st.markdown("---")
    
    with st.spinner("Analyzing image and searching visually similar products..."):
        try:
            img = PILImage.open(uploaded_image)
            best_match_results = vs_engine.search_similar_images(img, top_k=10)
            
            if best_match_results:
                matched_rows_list = []
                matched_subcategory = None
                
                # Retrieve direct matching product rows from df_global
                for img_path, score in best_match_results:
                    lid = extract_listing_id(img_path)
                    if lid and not df_global.empty:
                        matched_df = df_global[df_global['listing_id'] == lid]
                        if not matched_df.empty:
                            p_row = matched_df.iloc[0]
                            matched_rows_list.append((p_row, score))
                            if not matched_subcategory:
                                sub = p_row.get('Subcategory', None)
                                if pd.isna(sub) or str(sub).strip() in ['', 'nan', 'None']:
                                    sub = p_row.get('Category', None)
                                if pd.notna(sub) and str(sub).strip() not in ['', 'nan', 'None']:
                                    matched_subcategory = str(sub).strip()
                
                # 1. Show Top Visually Similar Products
                if matched_rows_list:
                    st.success(f"**Found {len(matched_rows_list)} Visually Similar Products!** (Category: {matched_subcategory if matched_subcategory else 'General'})")
                    st.markdown("### 🎯 Top Visually Similar Products:")
                    
                    cols = st.columns(3)
                    for i, (p_row, score) in enumerate(matched_rows_list[:6]):
                        with cols[i % 3]:
                            render_product_card(p_row)
                            
                    st.markdown("---")
                
                # 2. Show More Products from the Same Category/Subcategory
                if matched_subcategory:
                    st.markdown(f"### 📦 More Products in '{matched_subcategory}':")
                    subcat_df = df_global[(df_global['Subcategory'] == matched_subcategory) | (df_global['Category'] == matched_subcategory)]
                    
                    # Exclude already shown listing IDs
                    shown_ids = {r[0]['listing_id'] for r in matched_rows_list[:6]}
                    subcat_df = subcat_df[~subcat_df['listing_id'].isin(shown_ids)]
                    subcat_df = subcat_df.drop_duplicates(subset=['Product Name']).head(12)
                    
                    if not subcat_df.empty:
                        cols = st.columns(3)
                        for i, (_, row) in enumerate(subcat_df.iterrows()):
                            with cols[i % 3]:
                                render_product_card(row)
                elif not matched_rows_list:
                    st.warning("Could not identify matching products for this image in the database.")
            else:
                st.warning("Visual Search index is empty. Please wait for the background indexing to finish.")
        except Exception as e:
            st.error(f"Visual search failed: {e}")

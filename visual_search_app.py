import os
import io
import importlib
import threading
import time
from typing import List, Tuple, Union, Optional, Dict
import numpy as np
import pandas as pd
from PIL import Image as PILImage
import streamlit as st
from sqlalchemy import create_engine

import visual_search
importlib.reload(visual_search)
from visual_search import VisualSearchEngine


# Environment configuration
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["HF_HOME"] = "./.hf_cache"

# ==========================================
# 1. APPLICATION & UI CONFIGURATION
# ==========================================
st.set_page_config(page_title="JaxMart Visual Search", page_icon="🔍", layout="wide")

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

# ==========================================
# 2. GLOBAL CONSTANTS & MAPPINGS
# ==========================================
DB_URL = "postgresql://postgres:Jadequest%403009@3.111.57.216:5432/jaxmart_db"

SPELLING_MAP = {
    'wight': 'white',
    'whit': 'white',
    'blck': 'black',
    'blak': 'black',
    'tshirt': 't-shirt',
    't shirt': 't-shirt',
    'tee': 't-shirt',
    'bleu': 'blue',
    'yello': 'yellow',
    'slv': 'sleeve',
    'coton': 'cotton',
    'purpl': 'purple',
    'oraneg': 'orange'
}

# Note: CATEGORY_CLUSTERS dictionary is no longer hardcoded!
# Categories are now dynamically clustered using zero-shot semantic text embeddings on live DB categories.

# ==========================================
# 3. DATA & MODEL LOADING
# ==========================================
@st.cache_data(show_spinner=False)
def load_dataset() -> pd.DataFrame:
    """Fetches full product catalogue, category, seller, and image details from PostgreSQL."""
    try:
        engine = create_engine(DB_URL)
        query = '''
        SELECT 
            l.id as listing_id,
            l.title as "Product Name",
            l.description as "Description",
            l."shortDesc" as "Short Description",
            l.tags as "Tags",
            l."isFeatured" as "Is Featured",
            l."isVerified" as "Is Verified",
            l."avgRating" as "Rating",
            l."reviewCount" as "Reviews",
            l."viewCount" as "Views",
            l."enquiryCount" as "Enquiries",
            l."publishedAt" as "Published Date",
            c.name as "Category",
            c.slug as "Category Slug",
            pd.brand as "Brand",
            pd.model as "Model",
            pd.sku as "SKU",
            pd."priceType" as "Price Type",
            pd."pricePerUnit" as "Price",
            pd."priceRangeMin" as "Min Price",
            pd."priceRangeMax" as "Max Price",
            pd.currency as "Currency",
            pd."unitOfMeasure" as "Unit",
            pd."minOrderQty" as "MOQ",
            pd."maxOrderQty" as "Max Order Qty",
            pd."stockAvailable" as "Stock Available",
            pd."totalStock" as "Total Stock",
            pd."leadTimeDays" as "Lead Time Days",
            pd."deliveryTime" as "Delivery Time",
            pd."countryOfOrigin" as "Country of Origin",
            pd.specifications as "Specifications",
            pd.warranty as "Warranty",
            pd."returnPolicy" as "Return Policy",
            pd."paymentTerms" as "Payment Terms",
            pd."packagingDetails" as "Packaging Details",
            pd."packagingUnit" as "Packaging Unit",
            pd."sampleAvailable" as "Sample Available",
            pd."samplePrice" as "Sample Price",
            pd.certifications as "Certifications",
            pd."hsnCode" as "HSN Code",
            pd."gstRate" as "GST Rate",
            pd."supplyAbility" as "Supply Ability",
            bp."businessName" as "Company Name",
            bp."businessType" as "Business Type",
            bp."establishedYear" as "Established Year",
            bp."employeeRange" as "Employee Range",
            bp."annualTurnover" as "Annual Turnover",
            bp.gstin as "GSTIN",
            bp.pan as "PAN",
            bp.website as "Website",
            bp."description" as "Company Description",
            addr.line1 as "Address Line",
            addr.city as "Location",
            addr.state as "State",
            addr.pincode as "Pincode",
            addr.country as "Country",
            u."fullName" as "Seller Name",
            u.phone as "Seller Phone",
            u.email as "Seller Email",
            u."trustScore" as "Trust Score",
            lm.url as "Image URL"
        FROM listings l
        LEFT JOIN categories c ON l."categoryId" = c.id
        LEFT JOIN product_details pd ON l.id = pd."listingId"
        LEFT JOIN business_profiles bp ON l."sellerId" = bp."userId"
        LEFT JOIN users u ON l."sellerId" = u.id
        LEFT JOIN (
            SELECT DISTINCT ON ("userId") "userId", line1, city, state, pincode, country 
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


df_global = load_dataset()


@st.cache_resource(show_spinner=False)
def load_visual_search_engine(cache_version="v8.0_dynamic_clustering"):
    """Initializes and caches the VisualSearchEngine instance."""
    try:
        engine = VisualSearchEngine()
        engine.load_model()
        if engine.load_index():
            if 'df_global' in globals() and not df_global.empty and 'Category' in df_global.columns:
                all_cats = df_global['Category'].dropna().unique().tolist()
                engine.build_category_index(all_cats)
            return engine
        return None
    except Exception as e:
        st.error(f"Error loading visual search engine: {e}")
        return None


vs_engine = load_visual_search_engine()
if vs_engine and (not hasattr(vs_engine, 'category_embeddings') or vs_engine.category_embeddings is None):
    if not df_global.empty and 'Category' in df_global.columns:
        all_cats = df_global['Category'].dropna().unique().tolist()
        vs_engine.build_category_index(all_cats)


# Background Silent Sync Service
@st.cache_resource
def start_background_auto_sync():
    def auto_sync_worker():
        from auto_sync import incremental_sync
        while True:
            try:
                incremental_sync()
            except Exception:
                pass
            time.sleep(180)

    t = threading.Thread(target=auto_sync_worker, daemon=True)
    t.start()
    return True


start_background_auto_sync()

# ==========================================
# 4. UTILITY & RENDERING HELPERS
# ==========================================
def extract_listing_id(path_str: str) -> Optional[str]:
    """Extracts listing UUID from a product image file path."""
    if not path_str:
        return None
    parts = os.path.normpath(str(path_str)).replace('/', '\\').split('\\')
    for part in reversed(parts):
        if len(part) == 36 and part.count('-') == 4:
            return part
    if len(parts) >= 2:
        return parts[-2]
    return None


def is_val_valid(val) -> bool:
    """Validates if a database field has non-empty, displayable value."""
    if val is None:
        return False
    if isinstance(val, (list, tuple, np.ndarray)):
        return len(val) > 0
    if isinstance(val, dict):
        return len(val) > 0
    if isinstance(val, (bool, np.bool_)):
        return bool(val)
    try:
        na_check = pd.isna(val)
        if isinstance(na_check, (bool, np.bool_)) and na_check:
            return False
    except Exception:
        pass
    s = str(val).strip().lower()
    return s not in ['', 'nan', 'none', 'n/a', 'null', '0', '0.0', '[]', '{}']


def render_product_card(row: pd.Series):
    """Renders a comprehensive e-commerce product card in Streamlit."""
    with st.container(border=True):
        img_url = row.get('Image URL', None)
        if is_val_valid(img_url):
            try:
                st.image(str(img_url), use_container_width=True)
            except Exception:
                pass

        # Badges (Category, Stock, Verified, Featured)
        badge_html = []
        cat_val = row.get('Category', '')
        if is_val_valid(cat_val):
            badge_html.append(f"<span style='background:#263238; color:#ECEFF1; font-size:11px; padding:2px 7px; border-radius:12px; margin-right:4px;'>📂 {cat_val}</span>")
        if row.get('Is Verified') is True or str(row.get('Is Verified')).lower() == 'true':
            badge_html.append("<span style='background:#1B5E20; color:#E8F5E9; font-size:11px; padding:2px 7px; border-radius:12px; margin-right:4px;'>✅ Verified</span>")
        if row.get('Is Featured') is True or str(row.get('Is Featured')).lower() == 'true':
            badge_html.append("<span style='background:#E65100; color:#FFF3E0; font-size:11px; padding:2px 7px; border-radius:12px; margin-right:4px;'>⭐ Featured</span>")
        if row.get('Stock Available') is True or str(row.get('Stock Available')).lower() == 'true':
            badge_html.append("<span style='background:#004D40; color:#E0F2F1; font-size:11px; padding:2px 7px; border-radius:12px;'>🟢 In Stock</span>")

        if badge_html:
            st.markdown(f"<div style='margin-bottom: 6px;'>{''.join(badge_html)}</div>", unsafe_allow_html=True)

        # Title
        p_name = row.get('Product Name', 'Unknown Product')
        p_name_str = str(p_name) if is_val_valid(p_name) else 'Unknown Product'
        display_title = p_name_str if len(p_name_str) <= 50 else p_name_str[:47] + "..."
        st.markdown(f"<h4 style='color: #FF4B2B; margin-top:2px; margin-bottom: 8px; line-height: 1.3;' title='{p_name_str}'>{display_title}</h4>", unsafe_allow_html=True)

        # Price
        curr_code = str(row.get('Currency', 'INR')).strip()
        curr_sym = "₹" if curr_code.upper() in ['INR', ''] else f"{curr_code} "
        unit_str = f" / {row.get('Unit')}" if is_val_valid(row.get('Unit')) else ""

        if is_val_valid(row.get('Price')):
            st.markdown(f"**Price:** <span style='font-size: 1.2rem; font-weight: 700; color: #00E676;'>{curr_sym}{row.get('Price')}{unit_str}</span>", unsafe_allow_html=True)
        elif is_val_valid(row.get('Min Price')) and is_val_valid(row.get('Max Price')):
            st.markdown(f"**Price Range:** <span style='font-size: 1.1rem; font-weight: 700; color: #00E676;'>{curr_sym}{row.get('Min Price')} - {curr_sym}{row.get('Max Price')}{unit_str}</span>", unsafe_allow_html=True)

        # Details Accordion / Tabs
        with st.expander("📋 View Complete Specifications & Details", expanded=False):
            tabs = st.tabs(["📌 Specs", "📦 Orders", "🏢 Seller", "📝 Overview"])

            # Tab 1: Product Specs
            with tabs[0]:
                if is_val_valid(row.get('Brand')):
                    st.markdown(f"**🏷️ Brand:** {row.get('Brand')}")
                if is_val_valid(row.get('Model')):
                    st.markdown(f"**Model:** {row.get('Model')}")
                if is_val_valid(row.get('SKU')):
                    st.markdown(f"**SKU:** `{row.get('SKU')}`")
                if is_val_valid(row.get('Country of Origin')):
                    st.markdown(f"**🌍 Origin:** {row.get('Country of Origin')}")
                if is_val_valid(row.get('Warranty')):
                    st.markdown(f"**🛡️ Warranty:** {row.get('Warranty')}")
                if is_val_valid(row.get('HSN Code')):
                    st.markdown(f"**HSN Code:** `{row.get('HSN Code')}`")
                if is_val_valid(row.get('GST Rate')):
                    st.markdown(f"**GST Rate:** {row.get('GST Rate')}%")

            # Tab 2: Ordering & Stock
            with tabs[1]:
                if is_val_valid(row.get('MOQ')):
                    st.markdown(f"**📦 Minimum Order Qty (MOQ):** {row.get('MOQ')} {row.get('Unit', '')}")
                if is_val_valid(row.get('Delivery Time')):
                    st.markdown(f"**🚚 Delivery Time:** {row.get('Delivery Time')}")
                if is_val_valid(row.get('Payment Terms')):
                    st.markdown(f"**💳 Payment Terms:** {row.get('Payment Terms')}")
                if is_val_valid(row.get('Return Policy')):
                    st.markdown(f"**🔄 Return Policy:** {row.get('Return Policy')}")

            # Tab 3: Seller Profile
            with tabs[2]:
                if is_val_valid(row.get('Company Name')):
                    st.markdown(f"**🏢 Business:** {row.get('Company Name')}")
                if is_val_valid(row.get('Seller Name')):
                    st.markdown(f"**👤 Contact:** {row.get('Seller Name')}")
                if is_val_valid(row.get('Seller Phone')):
                    st.markdown(f"**📞 Phone:** `{row.get('Seller Phone')}`")
                if is_val_valid(row.get('Seller Email')):
                    st.markdown(f"**✉️ Email:** {row.get('Seller Email')}")
                if is_val_valid(row.get('Location')) or is_val_valid(row.get('State')):
                    st.markdown(f"**📍 Location:** {row.get('Location', '')}, {row.get('State', '')}")

            # Tab 4: Description & Tags
            with tabs[3]:
                desc = row.get('Description') or row.get('Short Description') or "No description provided."
                st.markdown(f"**Description:**\n{desc}")
                tags_data = row.get('Tags')
                if is_val_valid(tags_data):
                    if isinstance(tags_data, list):
                        tag_list = tags_data
                    else:
                        tag_list = str(tags_data).replace('{', '').replace('}', '').replace('[', '').replace(']', '').replace('"', '').split(',')
                    tag_badges = [f"`#{t.strip()}`" for t in tag_list if str(t).strip()]
                    if tag_badges:
                        st.markdown(f"**🏷️ Tags:** {' '.join(tag_badges)}")
                st.caption(f"Listing ID: `{row.get('listing_id')}`")


# ==========================================
# 5. CORE SEARCH PIPELINE
# ==========================================
def execute_multimodal_search(
    query_image: PILImage.Image,
    active_text: str,
    df: pd.DataFrame,
    engine: VisualSearchEngine,
    top_k: int = 9
) -> Tuple[Optional[str], List[pd.Series]]:
    """
    Executes Category-Anchored Multimodal Search combining:
      1. Zero-shot CLIP category detection across 200+ database categories.
      2. Dynamic text modifier embeddings for colors, styles, and attributes.
      3. Fast in-memory FAISS cosine similarity re-ranking.
    """
    # 1. Image Embedding
    img_emb = engine.get_image_embedding(query_image)
    img_emb = img_emb / np.linalg.norm(img_emb)

    # 2. Zero-Shot Category Classification & Dynamic Semantic Clustering
    detected_category_name = None
    allowed_categories = []
    if hasattr(engine, 'classify_categories'):
        detected_cats = engine.classify_categories(query_image, top_n=3)
        if detected_cats:
            detected_category_name = detected_cats[0][0]
            if hasattr(engine, 'get_dynamic_cluster'):
                # 100% Dynamic Semantic Auto-Clustering (Zero Hardcoding)
                allowed_categories = engine.get_dynamic_cluster(detected_category_name, threshold=0.80)
            else:
                allowed_categories = [detected_category_name]

    allowed_cat_set = set(allowed_categories)

    # 3. Process Text Modifier & Build Multimodal Embedding
    user_mod = (active_text or "").strip().lower()
    for k, v in SPELLING_MAP.items():
        if k in user_mod:
            user_mod = user_mod.replace(k, v)

    modifier_emb = None
    mod_tokens = set()
    if user_mod:
        cat_ctx = detected_category_name.lower() if detected_category_name else "product"
        prompt = f"a product photo of {user_mod} {cat_ctx}"
        modifier_emb = engine.get_text_embedding(prompt)
        modifier_emb = modifier_emb / np.linalg.norm(modifier_emb)
        for w in user_mod.replace('-', ' ').replace(',', ' ').split():
            if len(w) > 1:
                mod_tokens.add(w)

    lid_vec_map = getattr(engine, 'lid_vec_map', {})

    # 4. Multi-Tier Scoring
    scored_candidates = []
    for _, row in df.iterrows():
        p_cat = str(row.get('Category', ''))
        if p_cat not in allowed_cat_set:
            continue

        lid = str(row.get('listing_id', ''))
        p_name = str(row.get('Product Name', '')).lower()
        p_desc = str(row.get('Description', '')).lower()
        p_tags = str(row.get('Tags', '')).lower()
        p_specs = str(row.get('Specifications', '')).lower()

        # Text keyword match
        text_matches = 0
        if mod_tokens:
            for tok in mod_tokens:
                if tok in p_name:
                    text_matches += 3
                if tok in p_specs:
                    text_matches += 2
                if tok in p_tags:
                    text_matches += 2
                if tok in p_desc:
                    text_matches += 1

        # FAISS vector similarity
        prod_vec = lid_vec_map.get(lid, None)
        img_sim = 0.0
        color_sim = 0.0
        if prod_vec is not None:
            img_sim = float((prod_vec @ img_emb.T)[0][0])
            if modifier_emb is not None:
                color_sim = float((prod_vec @ modifier_emb.T)[0][0])

        cat_prio = 10.0 if p_cat == detected_category_name else 5.0

        if user_mod:
            final_score = (text_matches * 100.0) + (color_sim * 60.0) + (img_sim * 10.0) + cat_prio
        else:
            final_score = (img_sim * 50.0) + cat_prio

        scored_candidates.append((row, final_score))

    scored_candidates.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate results
    matched_rows_list = []
    seen_ids = set()
    primary_category = None

    for row, final_sc in scored_candidates:
        lid = row.get('listing_id')
        if lid not in seen_ids:
            seen_ids.add(lid)
            matched_rows_list.append(row)
            if not primary_category and is_val_valid(row.get('Category')):
                primary_category = str(row.get('Category')).strip()

    if not primary_category and detected_category_name:
        primary_category = detected_category_name

    return primary_category, matched_rows_list[:top_k]


# ==========================================
# 6. STREAMLIT UI & INTERACTION
# ==========================================
if 'prev_img_sig' not in st.session_state:
    st.session_state['prev_img_sig'] = None
if 'search_text_input' not in st.session_state:
    st.session_state['search_text_input'] = ""
if 'trigger_search' not in st.session_state:
    st.session_state['trigger_search'] = False

with st.container(border=True):
    st.markdown("### 📷 Upload & Search")
    uploaded_image = st.file_uploader("1. Upload product image", type=['jpg', 'jpeg', 'png'], key="img_uploader_box")

    current_img_sig = (uploaded_image.name, uploaded_image.size) if uploaded_image is not None else None
    if current_img_sig != st.session_state['prev_img_sig']:
        st.session_state['prev_img_sig'] = current_img_sig
        st.session_state['search_text_input'] = ""
        st.session_state['trigger_search'] = False
        st.rerun()

    st.markdown("**✍️ 2. Color / Style / Keyword (Optional)**")
    user_keyword = st.text_input(
        "Enter color, style, or attribute:",
        placeholder="e.g. black, white, cotton, round neck, formal...",
        label_visibility="collapsed",
        key="search_text_input"
    )
    st.caption("💡 *Tip: Upload an image, optionally type attributes (e.g. 'black'), and click 'Search Products'.*")

    if uploaded_image is not None:
        st.image(uploaded_image, width=150, caption="Uploaded Image Preview")

    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("🚀 Search Products", type="primary", use_container_width=True)

if search_clicked:
    if uploaded_image is None and not st.session_state.get('search_text_input', '').strip():
        st.warning("⚠️ Please upload an image or enter a keyword before searching.")
    else:
        st.session_state['trigger_search'] = True

if uploaded_image is not None and vs_engine is not None and st.session_state.get('trigger_search', False):
    st.markdown("---")
    col_preview, col_query_info = st.columns([1, 3])
    with col_preview:
        st.image(uploaded_image, width=220, caption="Uploaded Query Image")
    with col_query_info:
        active_text = st.session_state.get('search_text_input', '').strip()
        if active_text:
            st.info(f"🔍 **Searching with Image + Filter:** `{active_text}`")
        else:
            st.info("🔍 **Searching by AI Visual Match & Category Recognition**")

    with st.spinner("Analyzing image and searching matching products in Live Database..."):
        try:
            query_pil = PILImage.open(uploaded_image).convert('RGB')
            active_text = st.session_state.get('search_text_input', '').strip()

            primary_cat, matched_products = execute_multimodal_search(
                query_image=query_pil,
                active_text=active_text,
                df=df_global,
                engine=vs_engine,
                top_k=9
            )

            if matched_products:
                cat_header = f" (Category: **{primary_cat}**)" if primary_cat else ""
                st.success(f"🎯 **Found {len(matched_products)} Matching Products!**{cat_header}")
                st.markdown("### 🎯 Matching Products in Live Database:")

                cols = st.columns(3)
                for i, p_row in enumerate(matched_products):
                    with cols[i % 3]:
                        render_product_card(p_row)

                st.markdown("---")

                # Show More Products from Same Category
                if primary_cat:
                    subcat_df = df_global[df_global['Category'] == primary_cat]
                    shown_ids = {r['listing_id'] for r in matched_products}
                    subcat_df = subcat_df[~subcat_df['listing_id'].isin(shown_ids)].drop_duplicates(subset=['Product Name']).head(9)

                    if not subcat_df.empty:
                        st.markdown(f"### 📦 More Products in **'{primary_cat}'**:")
                        cols_more = st.columns(3)
                        for i, (_, row) in enumerate(subcat_df.iterrows()):
                            with cols_more[i % 3]:
                                render_product_card(row)
            else:
                st.warning("No matching products found for this search. Try uploading another image or entering a keyword.")
        except Exception as e:
            st.error(f"Visual search failed: {e}")

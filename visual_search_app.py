import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["HF_HOME"] = "./.hf_cache"

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image as PILImage
from sqlalchemy import create_engine
import importlib
import visual_search
importlib.reload(visual_search)
from visual_search import VisualSearchEngine

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
        # Fetch comprehensive product, seller, and category details from database
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

# Load Dataset (Once)
df_global = load_dataset()

@st.cache_resource(show_spinner=False)
def load_visual_search_engine(cache_version="v3.0_universal_category_engine"):
    try:
        import huggingface_hub.constants
        huggingface_hub.constants.HF_HUB_CACHE = "./.hf_cache"
        os.environ["HF_HOME"] = "./.hf_cache"
        os.environ["TRANSFORMERS_CACHE"] = "./.hf_cache"
        
        importlib.reload(visual_search)
        engine = visual_search.VisualSearchEngine()
        engine.load_model()
        if engine.load_index():
            # Build zero-shot category index for all categories in DB
            if 'df_global' in globals() and not df_global.empty and 'Category' in df_global.columns:
                all_cats = df_global['Category'].dropna().unique().tolist()
                engine.build_category_index(all_cats)
            return engine
        return None
    except Exception as e:
        st.error(f"Error loading visual search engine: {e}")
        return None

vs_engine = load_visual_search_engine()
if vs_engine and not hasattr(vs_engine, 'category_embeddings') and not df_global.empty and 'Category' in df_global.columns:
    all_cats = df_global['Category'].dropna().unique().tolist()
    vs_engine.build_category_index(all_cats)



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

def is_val_valid(val):
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

def render_product_card(row):
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
            
        # Product Title
        p_name = row.get('Product Name', 'Unknown Product')
        if not is_val_valid(p_name):
            p_name = 'Unknown Product'
        p_name_str = str(p_name)
        display_title = p_name_str if len(p_name_str) <= 50 else p_name_str[:47] + "..."
        st.markdown(f"<h4 style='color: #FF4B2B; margin-top:2px; margin-bottom: 8px; line-height: 1.3;' title='{p_name_str}'>{display_title}</h4>", unsafe_allow_html=True)
        
        # Price
        curr_code = str(row.get('Currency', 'INR')).strip()
        curr_sym = "₹" if curr_code.upper() in ['INR', ''] else f"{curr_code} "
        unit_str = f" / {row.get('Unit')}" if is_val_valid(row.get('Unit')) else ""
        
        price_val = row.get('Price', None)
        min_p = row.get('Min Price', None)
        max_p = row.get('Max Price', None)
        
        if is_val_valid(min_p) and is_val_valid(max_p):
            price_display = f"{curr_sym}{min_p} - {curr_sym}{max_p}{unit_str}"
        elif is_val_valid(price_val):
            try:
                p_float = float(price_val)
                p_formatted = f"{p_float:,.2f}".rstrip('0').rstrip('.')
            except Exception:
                p_formatted = str(price_val)
            price_display = f"{curr_sym}{p_formatted}{unit_str}"
        else:
            price_display = "Contact for Price"
            
        st.markdown(f"**💰 Price:** <span style='color: #4CAF50; font-size: 1.1rem; font-weight: bold;'>{price_display}</span>", unsafe_allow_html=True)
        
        # MOQ
        moq_val = row.get('MOQ', None)
        moq_unit = str(row.get('Unit', '')).strip() if is_val_valid(row.get('Unit')) else "units"
        moq_str = f"{moq_val} {moq_unit}" if is_val_valid(moq_val) else "N/A"
        st.markdown(f"**📦 MOQ:** {moq_str}")
        
        # Company & Location
        comp_name = row.get('Company Name', 'N/A')
        comp_str = str(comp_name) if is_val_valid(comp_name) else "N/A"
        
        loc_val = row.get('Location', '')
        state_val = row.get('State', '')
        loc_parts = [str(x).strip() for x in [loc_val, state_val] if is_val_valid(x)]
        loc_str = ", ".join(loc_parts) if loc_parts else "N/A"
        
        st.markdown(f"**🏢 Company:** {comp_str}")
        st.markdown(f"**📍 Location:** {loc_str}")
        
        # Rating
        rating_val = row.get('Rating', '0')
        rev_val = row.get('Reviews', '0')
        rating_clean = rating_val if is_val_valid(rating_val) else '0'
        rev_clean = rev_val if is_val_valid(rev_val) else '0'
        st.markdown(f"**⭐ Rating:** {rating_clean} ({rev_clean} reviews)")
        
        # Expandable All-Data Section
        with st.expander("📋 View All Product & Seller Details"):
            tabs = st.tabs(["🏷️ Specs & Product", "🚚 Stock & Delivery", "🏢 Seller & Contact", "📝 Description"])
            
            # Tab 1: Specs & Product
            with tabs[0]:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Brand:** {row.get('Brand') if is_val_valid(row.get('Brand')) else 'N/A'}")
                    st.markdown(f"**Model:** {row.get('Model') if is_val_valid(row.get('Model')) else 'N/A'}")
                    st.markdown(f"**SKU:** `{row.get('SKU') if is_val_valid(row.get('SKU')) else 'N/A'}`")
                with col_b:
                    st.markdown(f"**Country of Origin:** {row.get('Country of Origin') if is_val_valid(row.get('Country of Origin')) else 'N/A'}")
                    st.markdown(f"**HSN Code:** {row.get('HSN Code') if is_val_valid(row.get('HSN Code')) else 'N/A'}")
                    st.markdown(f"**GST Rate:** {row.get('GST Rate') if is_val_valid(row.get('GST Rate')) else 'N/A'}")
                
                # Extra Specifications dictionary if present
                specs = row.get('Specifications', None)
                if specs and isinstance(specs, dict) and len(specs) > 0:
                    st.markdown("**Detailed Specifications:**")
                    spec_items = []
                    for k, v in specs.items():
                        if is_val_valid(v) and k not in ['sku', 'hsnCode', 'gstRate']:
                            spec_items.append(f"- **{k.capitalize()}:** {v}")
                    if spec_items:
                        st.markdown("\n".join(spec_items))
                        
                if is_val_valid(row.get('Warranty')):
                    st.markdown(f"**🛡️ Warranty:** {row.get('Warranty')}")
                if is_val_valid(row.get('Return Policy')):
                    st.markdown(f"**🔄 Return Policy:** {row.get('Return Policy')}")
            
            # Tab 2: Stock, Logistics & Pricing
            with tabs[1]:
                col_c, col_d = st.columns(2)
                with col_c:
                    st.markdown(f"**Price Type:** {row.get('Price Type') if is_val_valid(row.get('Price Type')) else 'FIXED'}")
                    st.markdown(f"**Stock Available:** {'Yes ✅' if row.get('Stock Available') is True else 'No ❌'}")
                    st.markdown(f"**Total Stock Units:** {row.get('Total Stock') if is_val_valid(row.get('Total Stock')) else 'N/A'}")
                    st.markdown(f"**Max Order Qty:** {row.get('Max Order Qty') if is_val_valid(row.get('Max Order Qty')) else 'N/A'}")
                with col_d:
                    st.markdown(f"**Lead Time:** {row.get('Lead Time Days')} days" if is_val_valid(row.get('Lead Time Days')) else "**Lead Time:** N/A")
                    st.markdown(f"**Delivery Time:** {row.get('Delivery Time') if is_val_valid(row.get('Delivery Time')) else 'N/A'}")
                    st.markdown(f"**Sample Available:** {'Yes ✅' if row.get('Sample Available') is True else 'No ❌'}")
                    if is_val_valid(row.get('Sample Price')):
                        st.markdown(f"**Sample Price:** ₹{row.get('Sample Price')}")
                        
                if is_val_valid(row.get('Supply Ability')):
                    st.markdown(f"**Supply Ability:** {row.get('Supply Ability')}")
                if is_val_valid(row.get('Packaging Details')):
                    st.markdown(f"**Packaging:** {row.get('Packaging Details')}")
                if is_val_valid(row.get('Payment Terms')):
                    st.markdown(f"**Payment Terms:** {row.get('Payment Terms')}")

            # Tab 3: Seller & Company Info
            with tabs[2]:
                st.markdown(f"**🏢 Business Name:** {row.get('Company Name') if is_val_valid(row.get('Company Name')) else 'N/A'}")
                if is_val_valid(row.get('Business Type')):
                    st.markdown(f"**Business Type:** {row.get('Business Type')}")
                if is_val_valid(row.get('Seller Name')):
                    st.markdown(f"**👤 Contact Person:** {row.get('Seller Name')}")
                if is_val_valid(row.get('Seller Phone')):
                    st.markdown(f"**📞 Phone:** `{row.get('Seller Phone')}`")
                if is_val_valid(row.get('Seller Email')):
                    st.markdown(f"**✉️ Email:** {row.get('Seller Email')}")
                if is_val_valid(row.get('GSTIN')):
                    st.markdown(f"**GSTIN:** `{row.get('GSTIN')}`")
                if is_val_valid(row.get('PAN')):
                    st.markdown(f"**PAN:** `{row.get('PAN')}`")
                if is_val_valid(row.get('Established Year')):
                    st.markdown(f"**Established Year:** {row.get('Established Year')}")
                if is_val_valid(row.get('Employee Range')):
                    st.markdown(f"**Team Size:** {row.get('Employee Range')}")
                if is_val_valid(row.get('Annual Turnover')):
                    st.markdown(f"**Annual Turnover:** {row.get('Annual Turnover')}")
                
                # Full Address
                addr_parts = [row.get('Address Line'), row.get('Location'), row.get('State'), row.get('Pincode'), row.get('Country')]
                valid_addr = [str(a).strip() for a in addr_parts if is_val_valid(a)]
                if valid_addr:
                    st.markdown(f"**📍 Full Address:** {', '.join(valid_addr)}")
                
                if is_val_valid(row.get('Website')):
                    web_url = str(row.get('Website')).strip()
                    if not web_url.startswith('http'):
                        web_url = f"https://{web_url}"
                    st.markdown(f"**🌐 Website:** [{row.get('Website')}]({web_url})")
                if is_val_valid(row.get('Trust Score')):
                    st.markdown(f"**🛡️ Trust Score:** {row.get('Trust Score')}/100")

            # Tab 4: Description & Tags
            with tabs[3]:
                desc_text = row.get('Description')
                if not is_val_valid(desc_text):
                    desc_text = row.get('Short Description')
                if is_val_valid(desc_text):
                    st.markdown(f"**Description:**\n{desc_text}")
                else:
                    st.markdown("*No description provided.*")
                    
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
        st.markdown("<br>", unsafe_allow_html=True)

# Session state management for Image tracking and Auto-Resetting Text Box
if 'prev_img_sig' not in st.session_state:
    st.session_state['prev_img_sig'] = None
if 'search_text_input' not in st.session_state:
    st.session_state['search_text_input'] = ""
if 'trigger_search' not in st.session_state:
    st.session_state['trigger_search'] = False

# Search UI Section
with st.container(border=True):
    st.markdown("### 📷 Upload & Search")
    uploaded_image = st.file_uploader("1. Upload product image", type=['jpg', 'jpeg', 'png'], key="img_uploader_box")
    
    # Check if a new image was uploaded -> Reset Text box and WAIT for user to click Search
    current_img_sig = (uploaded_image.name, uploaded_image.size) if uploaded_image is not None else None
    if current_img_sig != st.session_state['prev_img_sig']:
        st.session_state['prev_img_sig'] = current_img_sig
        st.session_state['search_text_input'] = "" # Reset text box on new image
        st.session_state['trigger_search'] = False # Do NOT search automatically
        st.rerun()

    st.markdown("**✍️ 2. Color / Style / Keyword (Optional)**")
    user_keyword = st.text_input(
        "Enter color, style, or attribute:",
        placeholder="e.g. black color, cotton, formal, round neck...",
        label_visibility="collapsed",
        key="search_text_input"
    )
    st.caption("💡 *Tip: Upload an image, optionally type attributes (e.g. 'black'), and click 'Search Products'.*")
    
    # Show small thumbnail preview of uploaded image before search
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
            img = PILImage.open(uploaded_image).convert('RGB')
            active_text = st.session_state.get('search_text_input', '').strip()
            
            # Ensure category embeddings are built
            if hasattr(vs_engine, 'build_category_index') and (not hasattr(vs_engine, 'category_embeddings') or vs_engine.category_embeddings is None):
                if not df_global.empty and 'Category' in df_global.columns:
                    all_cats = df_global['Category'].dropna().unique().tolist()
                    vs_engine.build_category_index(all_cats)

            # Step 1: Detect matching categories using CLIP zero-shot across all 200+ DB categories
            detected_category_name = None
            top_target_categories = []
            if hasattr(vs_engine, 'classify_categories'):
                detected_cats = vs_engine.classify_categories(img, top_n=4)
                if detected_cats:
                    detected_category_name = detected_cats[0][0]
                    top_target_categories = [c[0] for c in detected_cats]

            # Step 2: Query FAISS for top 500 visual nearest neighbors
            img_emb = vs_engine.get_image_embedding(img)
            distances, indices = vs_engine.index.search(img_emb.astype(np.float32), min(500, vs_engine.index.ntotal))
            
            faiss_scores = {}
            for dist, idx in zip(distances[0], indices[0]):
                if idx != -1 and idx < len(vs_engine.image_paths):
                    p = vs_engine.image_paths[idx]
                    lid = extract_listing_id(p)
                    if lid:
                        sim = 1.0 / (1.0 + float(dist))
                        if lid not in faiss_scores or sim > faiss_scores[lid]:
                            faiss_scores[lid] = sim

            # Step 3: Filter and Rank Candidates
            scored_candidates = []
            
            if active_text:
                # User provided keyword: Strict Keyword Matching + Visual Similarity Ranking
                tokens = [w.lower().strip() for w in active_text.replace('-', ' ').replace(',', ' ').split() if len(w.strip()) > 1]
                if any(t in active_text.lower() for t in ['tshirt', 't-shirt', 't shirt', 'tee']):
                    tokens.extend(['tshirt', 't-shirt', 't shirt', 't-shirts', 'tshirts', 'tee', 'tees', 'shirt', 'shirts', 'mens t-shirts', 'polyester tshirt', 'cotton t-shirt', 'apparel'])
                
                for _, row in df_global.iterrows():
                    lid = str(row.get('listing_id', ''))
                    p_name = str(row.get('Product Name', '')).lower()
                    p_cat = str(row.get('Category', '')).lower()
                    p_desc = str(row.get('Description', '')).lower()
                    p_tags = str(row.get('Tags', '')).lower()
                    
                    matches = sum(1 for t in tokens if t in p_name or t in p_cat or t in p_tags or t in p_desc)
                    if matches > 0:
                        vis_sc = faiss_scores.get(lid, 0.0)
                        final_sc = (matches * 15.0) + (vis_sc * 5.0)
                        scored_candidates.append((row, final_sc))
            else:
                # Pure Image Search: Target products from AI detected categories, sorted by visual match
                target_cat_set = set(top_target_categories)
                for _, row in df_global.iterrows():
                    lid = str(row.get('listing_id', ''))
                    p_cat = str(row.get('Category', ''))
                    vis_sc = faiss_scores.get(lid, 0.0)
                    
                    if p_cat in target_cat_set:
                        cat_rank = top_target_categories.index(p_cat) # 0, 1, 2, 3
                        cat_weight = 4.0 - cat_rank # 4, 3, 2, 1
                        final_sc = (cat_weight * 10.0) + (vis_sc * 5.0)
                        scored_candidates.append((row, final_sc))
                    elif vis_sc > 0.85:
                        # Direct near-identical visual match in FAISS
                        final_sc = vis_sc * 2.0
                        scored_candidates.append((row, final_sc))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Deduplicate by listing_id
            matched_rows_list = []
            seen_ids = set()
            primary_category = None

            for row, final_sc in scored_candidates:
                lid = row.get('listing_id')
                if lid not in seen_ids:
                    seen_ids.add(lid)
                    matched_rows_list.append((row, final_sc))
                    if not primary_category:
                        sub = row.get('Category', None)
                        if is_val_valid(sub):
                            primary_category = str(sub).strip()

            if not primary_category and detected_category_name:
                primary_category = detected_category_name

            # Step 4: Display Results
            if matched_rows_list:
                cat_header = f" (Category: **{primary_category}**)" if primary_category else ""
                st.success(f"🎯 **Found {len(matched_rows_list[:9])} Matching Products!**{cat_header}")
                st.markdown("### 🎯 Matching Products in Live Database:")
                
                cols = st.columns(3)
                for i, (p_row, score) in enumerate(matched_rows_list[:9]):
                    with cols[i % 3]:
                        render_product_card(p_row)
                        
                st.markdown("---")
                
                # Show More Products from the Same Category
                if primary_category:
                    subcat_df = df_global[df_global['Category'] == primary_category]
                    shown_ids = {r[0]['listing_id'] for r in matched_rows_list[:9]}
                    subcat_df = subcat_df[~subcat_df['listing_id'].isin(shown_ids)].drop_duplicates(subset=['Product Name']).head(9)
                    
                    if not subcat_df.empty:
                        st.markdown(f"### 📦 More Products in **'{primary_category}'**:")
                        cols_more = st.columns(3)
                        for i, (_, row) in enumerate(subcat_df.iterrows()):
                            with cols_more[i % 3]:
                                render_product_card(row)
            else:
                st.warning("No matching products found for this search. Try uploading another image or entering a keyword.")
        except Exception as e:
            st.error(f"Visual search failed: {e}")



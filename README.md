# 🔍 JaxMart Visual Search

A state-of-the-art multimodal visual search engine for JaxMart e-commerce platform powered by **OpenAI CLIP (ViT-B/32)**, **FAISS Vector Search**, and **Streamlit**.

---

## 📁 Project Architecture

```text
visualsearch/
│
├── visual_search_app.py        # 🚀 Main Streamlit Application UI & Search Interface
├── visual_search.py            # 🧠 Core Visual Search & CLIP Vector Search Engine
├── auto_sync.py                # ⚡ Background Zero-Download Real-Time Sync Service
│
├── scripts/                    # 🛠️ Maintenance & Offline Utilities
│   ├── sync_db_and_rebuild_index.py  # Full database batch sync & FAISS index builder
│   └── scraper.py                    # Product metadata scraper
│
├── visual_search_index.faiss   # 📦 Precomputed FAISS Vector Database (512-dim)
├── image_mapping.pkl           # 🗺️ Mapping index for Listing UUIDs
├── requirements.txt            # 📄 Python Dependencies
└── README.md                   # 📖 Project Documentation
```

---

## ⚡ Features

1. **Multimodal Visual Search:**
   - Upload any product photo to find visually and semantically matching products in the live database.
   - Zero-shot classification across **200+ product categories** (T-shirts, Sarees, Bubble Wrap, Safety Shoes, Vitrified Tiles, etc.).

2. **Category-Anchored Color & Attribute Modifiers:**
   - Type attributes like `black`, `white`, `cotton`, `round neck`, `formal` to rank specific variants at the top (#1).
   - Prevents cross-category contamination (e.g. typing `black` on a T-shirt photo will NEVER return black tiles or black gloves).

3. **High-Speed In-Memory Vector Search:**
   - Millisecond-level nearest neighbor retrieval across 8,000+ live products using FAISS.

4. **Silent Real-Time Background Sync:**
   - Automatically detects new products added to the PostgreSQL database and streams their image embeddings directly into RAM every 3 minutes.

---

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Visual Search Web Application
```bash
streamlit run visual_search_app.py
```

### 3. (Optional) Rebuild FAISS Vector Index from Scratch
```bash
python scripts/sync_db_and_rebuild_index.py
```

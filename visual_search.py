import os
import io
import pickle
import requests
import numpy as np
import pandas as pd
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
from typing import List, Tuple, Union, Optional, Dict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["HF_HOME"] = "./.hf_cache"


def _resolve_data_path(filename: str) -> str:
    """Resolves path for data files, checking data/ subfolder first, then root directory."""
    base = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base, "data", filename)
    if os.path.exists(data_path):
        return data_path
    return os.path.join(base, filename)


class VisualSearchEngine:
    """
    JaxMart Visual Search Engine
    Powered by OpenAI CLIP (ViT-B/32) and FAISS Vector Search.
    Supports:
      - High-speed Image Feature Extraction (512-dim)
      - Text/Prompt Feature Extraction
      - Zero-Shot Dynamic Category Classification across 200+ database categories
      - Fast In-Memory Vector Re-ranking and Multimodal Similarity
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        index_file: Optional[str] = None,
        mapping_file: Optional[str] = None
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.index_file = index_file if index_file else _resolve_data_path("visual_search_index.faiss")
        self.mapping_file = mapping_file if mapping_file else _resolve_data_path("image_mapping.pkl")

        self.model = None
        self.processor = None
        self.index = None
        self.image_paths: List[str] = []
        self.embedding_dim = 512

        # In-memory category zero-shot cache
        self.category_names: List[str] = []
        self.category_embeddings: Optional[np.ndarray] = None
        self.lid_vec_map: Dict[str, np.ndarray] = {}


    def load_model(self):
        """Loads and caches the CLIP model and processor on GPU/CPU."""
        if self.model is None:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.embedding_dim = self.model.config.projection_dim

    def load_index(self) -> bool:
        """Loads the FAISS index and listing ID mappings from disk."""
        if os.path.exists(self.index_file) and os.path.exists(self.mapping_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.mapping_file, 'rb') as f:
                self.image_paths = pickle.load(f)
            self._build_vector_cache()
            return True
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.image_paths = []
            return False

    def _build_vector_cache(self):
        """Caches normalized FAISS vectors mapped by listing UUID for instant re-ranking."""
        if self.index is None or self.index.ntotal == 0:
            return

        self.lid_vec_map = {}
        for idx, p in enumerate(self.image_paths):
            parts = os.path.normpath(str(p)).replace('/', '\\').split('\\')
            lid = None
            for part in reversed(parts):
                if len(part) == 36 and part.count('-') == 4:
                    lid = part
                    break
            if lid and idx < self.index.ntotal:
                try:
                    vec = self.index.reconstruct(int(idx)).reshape(1, -1)
                    vec = vec / np.linalg.norm(vec)
                    self.lid_vec_map[lid] = vec
                except Exception:
                    pass

    def get_image_embedding(self, image: Union[str, Image.Image, bytes, io.BytesIO]) -> np.ndarray:
        """Computes normalized 512-dim CLIP embedding for an image."""
        self.load_model()
        if isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
        elif isinstance(image, io.BytesIO):
            image = Image.open(image)
        elif isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                headers = {'User-Agent': 'Mozilla/5.0'}
                resp = requests.get(image, headers=headers, timeout=10)
                image = Image.open(io.BytesIO(resp.content))
            else:
                image = Image.open(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
            if hasattr(features, 'pooler_output'):
                features = features.pooler_output
            elif hasattr(features, 'image_embeds'):
                features = features.image_embeds

        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()

    def get_text_embedding(self, text: str) -> np.ndarray:
        """Computes normalized 512-dim CLIP text embedding."""
        self.load_model()
        inputs = self.processor(text=[text], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            if hasattr(features, 'pooler_output'):
                features = features.pooler_output
            elif hasattr(features, 'text_embeds'):
                features = features.text_embeds

        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()

    def build_category_index(self, categories: List[str]):
        """Precomputes normalized text embeddings and semantic similarity matrix for all live DB categories."""
        self.load_model()
        valid_cats = sorted(list(set([str(c).strip() for c in categories if str(c).strip() and str(c).lower() not in ['nan', 'none', 'null']])))
        self.category_names = valid_cats

        prompts = [f"a photo of {c.lower()}" for c in valid_cats]

        inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            cat_feats = self.model.get_text_features(**inputs)
            if hasattr(cat_feats, 'pooler_output'):
                cat_feats = cat_feats.pooler_output
            elif hasattr(cat_feats, 'text_embeds'):
                cat_feats = cat_feats.text_embeds
            cat_feats = cat_feats / cat_feats.norm(p=2, dim=-1, keepdim=True)

        self.category_embeddings = cat_feats.cpu().numpy()
        self.category_sim_matrix = self.category_embeddings @ self.category_embeddings.T

    def get_dynamic_cluster(self, category_name: str, threshold: float = 0.88, max_cluster_size: int = 4) -> List[str]:
        """
        Dynamically finds semantically related categories from live DB without hardcoding.
        Example: 'Mobile Phone' -> ['Mobile Phone', 'Smartphone']
                 'Mens T-Shirts' -> ['Mens T-Shirts', 'Mens Shirts', 'Kids Wear']
                 'Glass Doors' -> ['Glass Doors', 'Steel Doors', 'Doors & Windows', 'Flush Doors']
        """
        if not category_name or self.category_embeddings is None or len(self.category_names) == 0:
            return [category_name] if category_name else []

        if category_name not in self.category_names:
            return [category_name]

        cat_idx = self.category_names.index(category_name)
        sims = self.category_sim_matrix[cat_idx]

        matches = []
        for i, sim in enumerate(sims):
            other_cat = self.category_names[i]
            if other_cat != category_name and sim >= threshold:
                matches.append((other_cat, sim))

        matches.sort(key=lambda x: x[1], reverse=True)
        cluster = [category_name] + [m[0] for m in matches[:max_cluster_size - 1]]
        return cluster

    def classify_categories(self, query_image: Union[str, Image.Image], top_n: int = 4) -> List[Tuple[str, float]]:
        """Classifies a query image against all precomputed database categories."""
        if self.category_embeddings is None or len(self.category_names) == 0:
            return []

        img_emb = self.get_image_embedding(query_image)
        sims = (img_emb @ self.category_embeddings.T)[0]
        top_indices = np.argsort(sims)[::-1][:top_n]

        return [(self.category_names[i], float(sims[i])) for i in top_indices]

    def search_similar_images(self, query_image: Union[str, Image.Image], top_k: int = 10) -> List[Tuple[str, float]]:
        """Searches FAISS for top-k visual nearest neighbors."""
        if self.index is None or self.index.ntotal == 0:
            return []

        query_embedding = self.get_image_embedding(query_image)
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(dist)))

        return results

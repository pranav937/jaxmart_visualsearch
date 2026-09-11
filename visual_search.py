import os
import io
import requests
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss
import pickle
from typing import List, Tuple, Union, Optional

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ["HF_HOME"] = "./.hf_cache"

class VisualSearchEngine:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", index_file: str = "visual_search_index.faiss", mapping_file: str = "image_mapping.pkl"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.index_file = index_file
        self.mapping_file = mapping_file
        
        self.model = None
        self.processor = None
        self.index = None
        self.image_paths = [] # Holds listing_ids or paths
        self.embedding_dim = 512

    def load_model(self):
        if self.model is None:
            self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.embedding_dim = self.model.config.projection_dim

    def load_index(self) -> bool:
        """Loads FAISS index and listing ID mapping."""
        if os.path.exists(self.index_file) and os.path.exists(self.mapping_file):
            self.index = faiss.read_index(self.index_file)
            with open(self.mapping_file, 'rb') as f:
                self.image_paths = pickle.load(f)
            return True
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            self.image_paths = []
            return False

    def get_image_embedding(self, image: Union[str, Image.Image, bytes, io.BytesIO]) -> np.ndarray:
        """Computes CLIP embedding directly from PIL Image, File Path, Bytes, or In-Memory Stream."""
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
        """Computes normalized CLIP text embedding."""
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
        """Precomputes normalized CLIP text embeddings for all database categories."""
        self.load_model()
        valid_cats = sorted(list(set([str(c).strip() for c in categories if str(c).strip() and str(c).lower() not in ['nan', 'none', 'null']])))
        self.category_names = valid_cats
        
        # Enhanced descriptive prompt templates for each category
        prompts = []
        for cat in valid_cats:
            c_low = cat.lower()
            if 'saree' in c_low:
                prompts.append(f"a product photograph of an indian woman saree clothing garment, {c_low}")
            elif 't-shirt' in c_low or 't shirt' in c_low:
                prompts.append(f"a product photograph of a men t-shirt, round neck polo tee, {c_low}")
            elif 'wrap' in c_low or 'bubble' in c_low:
                prompts.append(f"a product photograph of bubble wrap plastic packaging sheet roll, {c_low}")
            elif 'glove' in c_low:
                prompts.append(f"a product photograph of industrial safety gloves hand protection, {c_low}")
            elif 'tile' in c_low:
                prompts.append(f"a product photograph of vitrified floor tiles ceramic marble, {c_low}")
            elif 'shoe' in c_low or 'footwear' in c_low:
                prompts.append(f"a product photograph of industrial safety shoes footwear, {c_low}")
            elif 'fabric' in c_low or 'denim' in c_low:
                prompts.append(f"a product photograph of textile cloth material denim fabric roll, {c_low}")
            else:
                prompts.append(f"a product photograph of {c_low}")

        inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            cat_feats = self.model.get_text_features(**inputs)
            if hasattr(cat_feats, 'pooler_output'):
                cat_feats = cat_feats.pooler_output
            elif hasattr(cat_feats, 'text_embeds'):
                cat_feats = cat_feats.text_embeds
            cat_feats = cat_feats / cat_feats.norm(p=2, dim=-1, keepdim=True)
            
        self.category_embeddings = cat_feats.cpu().numpy()

    def classify_categories(self, query_image: Union[str, Image.Image], top_n: int = 5) -> List[Tuple[str, float]]:
        """Classifies an image across all 200+ database categories."""
        if not hasattr(self, 'category_embeddings') or self.category_embeddings is None:
            return []
            
        img_emb = self.get_image_embedding(query_image)
        sims = (img_emb @ self.category_embeddings.T)[0]
        top_indices = np.argsort(sims)[::-1][:top_n]
        
        return [(self.category_names[i], float(sims[i])) for i in top_indices]

    def classify_domain(self, query_image: Union[str, Image.Image]) -> Tuple[str, List[str], float]:
        """Backward compatible domain classifier."""
        top_cats = self.classify_categories(query_image, top_n=3)
        if top_cats:
            return top_cats[0][0], [c[0] for c in top_cats], top_cats[0][1]
        return "General Product", [], 0.5


    def search_similar_images(self, query_image: Union[str, Image.Image], top_k: int = 10) -> List[Tuple[str, float]]:
        """Searches FAISS for top-k similar images."""
        if self.index is None or self.index.ntotal == 0:
            return []
            
        query_embedding = self.get_image_embedding(query_image)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(dist)))
                
        return results

    def search_multimodal(self, query_image: Optional[Union[str, Image.Image]] = None, query_text: Optional[str] = None, text_weight: float = 0.5, top_k: int = 50) -> List[Tuple[str, float]]:
        """Combines image and text queries to search FAISS index with balanced weights."""
        if self.index is None or self.index.ntotal == 0:
            return []

        emb = None
        if query_image is not None and query_text and query_text.strip():
            img_emb = self.get_image_embedding(query_image)
            txt_emb = self.get_text_embedding(query_text.strip())
            emb = (1.0 - text_weight) * img_emb + text_weight * txt_emb
            emb = emb / np.linalg.norm(emb, axis=-1, keepdims=True)
        elif query_image is not None:
            emb = self.get_image_embedding(query_image)
        elif query_text and query_text.strip():
            emb = self.get_text_embedding(query_text.strip())
        else:
            return []

        distances, indices = self.index.search(emb.astype(np.float32), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(dist)))
        return results



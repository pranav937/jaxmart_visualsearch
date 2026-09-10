import os
import io
import requests
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
import faiss
import pickle
from typing import List, Tuple, Union

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
            self.model = CLIPVisionModelWithProjection.from_pretrained(self.model_name).to(self.device)
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
            outputs = self.model(**inputs)
            features = outputs.image_embeds
            
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return features.cpu().numpy()

    def search_similar_images(self, query_image: Union[str, Image.Image], top_k: int = 10) -> List[Tuple[str, float]]:
        """Searches FAISS for top-k similar images and returns list of (listing_id/path, distance)."""
        if self.index is None or self.index.ntotal == 0:
            return []
            
        query_embedding = self.get_image_embedding(query_image)
        distances, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(dist)))
                
        return results

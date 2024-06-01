# assetSearch/backend/data_processor.py
import clip
import torch
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans


class DataProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", self.device)

    def compute_clip_embeddings(self, image_paths):
        images = [self.preprocess(Image.open(image_path)).unsqueeze(0) for image_path in image_paths]
        images = torch.cat(images).to(self.device)
        with torch.no_grad():
            embeddings = self.model.encode_image(images).cpu().numpy()
        return embeddings

    def compute_text_embeddings(self, texts):
        tokens = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            text_embeddings = self.model.encode_text(tokens).cpu().numpy()
        return text_embeddings

    def quantize_embeddings(self, embeddings):
        quantized = np.clip((embeddings + 1.0) * 127.5, 0, 255).astype(np.uint8)
        return quantized

    def perform_kmeans_clustering(self, embeddings, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters)
        cluster_indices = kmeans.fit_predict(embeddings)
        distances = kmeans.transform(embeddings)  # Distance to each cluster centroid
        min_distances = np.min(distances, axis=1)  # Minimum distance for each point
        return cluster_indices, min_distances

# assetSearch/backend/data_processor.py
import clip
import torch
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from .clip_model import CLIPModel
from .resnet_model import ResNetModel


class DataProcessor:
    def __init__(self, model_type="clip"):
        if model_type == "clip":
            self.model = CLIPModel()
        elif model_type == "resnet152":
            self.model = ResNetModel()
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def compute_image_embeddings(self, image_paths):
        return self.model.compute_image_embeddings(image_paths)

    def compute_text_embeddings(self, texts):
        return self.model.compute_text_embeddings(texts)

    def compute_embedding_from_pixels(self, img):
        return self.model.compute_embedding_from_pixels(img)

    def quantize_embeddings(self, embeddings):
        return self.model.quantize_embeddings(embeddings)

    def perform_kmeans_clustering(self, embeddings, num_clusters):
        kmeans = KMeans(n_clusters=num_clusters)
        cluster_indices = kmeans.fit_predict(embeddings)
        distances = kmeans.transform(embeddings)  # Distance to each cluster centroid
        min_distances = np.min(distances, axis=1)  # Minimum distance for each point
        return cluster_indices, min_distances

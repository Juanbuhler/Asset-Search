# backend/nearest_neighbors.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .database import session, ImageAsset
from .data_processor import DataProcessor
from backend.config import DEFAULT_MODEL_TYPE
import cv2

class NearestNeighborsSearch:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto', metric='cosine')
        self.embeddings = []
        self.uris = []
        self.current_dataset = None
        self.current_embedding_type = DEFAULT_MODEL_TYPE
        self.processor = DataProcessor()  # For embedding text queries

    def load_data(self, dataset_name=None, embedding_type=DEFAULT_MODEL_TYPE):
        if dataset_name == self.current_dataset and embedding_type == self.current_embedding_type:
            # Data already loaded, no need to reload
            return

        self.current_dataset = dataset_name
        self.current_embedding_type = embedding_type
        query = session.query(ImageAsset)
        if dataset_name:
            if dataset_name == "%":
                # Return all assets without filtering
                assets = query.all()
            else:
                dataset_name = f"%{dataset_name}%"  # Use % as the wildcard character
                query = query.filter(ImageAsset.dataset.like(dataset_name))

        # Proceed to query the assets
        assets = query.all()

        self.embeddings = []
        self.uris = []

        embedding_attr = DEFAULT_MODEL_TYPE
        if self.current_embedding_type == "clip":
            embedding_attr = "embeddings"
        elif self.current_embedding_type == "resnet152":
            embedding_attr = "resnet_embeddings"
        else:
            raise ValueError(f"Unsupported model type: {self.current_embedding_type}")

        for asset in assets:
            embeddings = getattr(asset, embedding_attr)
            quantized_embeddings = np.frombuffer(embeddings, dtype=np.uint8)
            self.embeddings.append((quantized_embeddings - 127.5)/255)
            self.uris.append(asset.uri)

        self.embeddings = np.vstack(self.embeddings)
        self.fit()

    def fit(self):
        self.model.fit(self.embeddings)

    def search(self, query, bias=""):
        print('***')
        print(bias)
        print('***')

        final_bias = 0
        if bias:
            # Split the bias into "/more of/less of/factor" to do biasing
            query_parts = bias.split("/")[:4]
            factor = 1
            if len(query_parts) == 4:
                factor = float(query_parts[3])
                query_parts = query_parts[1:3]  # Ignore the first part for biasing

            print(query_parts)
            embeddings = self.processor.compute_text_embeddings(query_parts)
            more_of = embeddings[0]

            less_of = more_of * 0
            if len(embeddings) > 1:
                less_of = embeddings[1]
            final_bias = (more_of - less_of) * factor
            print(final_bias)
        if isinstance(query, str):  # Text query
            # Split the query into "query/more of/less of/factor" to do biasing
            query_parts = query.split("/")[:4]
            factor = 1
            if len(query_parts) == 4:
                factor = float(query_parts[3])
                query_parts = query_parts[:3]

            embeddings = self.processor.compute_text_embeddings(query_parts)
            target_embedding = embeddings[0]

            more_of = less_of = target_embedding * 0
            if len(embeddings) > 1:
                more_of = embeddings[1]
            if len(embeddings) > 2:
                less_of = embeddings[2]
            final_embedding = target_embedding + (more_of - less_of) * factor
            query_embedding = (self.processor.quantize_embeddings(final_embedding) - 127.5) / 255
        elif len(query.shape) == 3:
            embeddings = self.processor.compute_embedding_from_pixels(query)
            query_embedding = (self.processor.quantize_embeddings(embeddings) - 127.5) / 255
        else:  # Image embedding query
            query_embedding = query

        query_embedding = query_embedding + final_bias
        distances, indices = self.model.kneighbors(query_embedding.reshape(1, -1), n_neighbors=self.n_neighbors)

        # Sort results by distance (similarity)
        sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1])

        # Convert indices back to URIs
        results = [(self.uris[idx], dist) for idx, dist in sorted_results]

        return results

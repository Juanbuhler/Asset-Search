# backend/nearest_neighbors.py
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .database import session, ImageAsset
from .data_processor import DataProcessor


class NearestNeighborsSearch:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.model = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='auto')
        self.embeddings = []
        self.uris = []
        self.current_dataset = None
        self.processor = DataProcessor()  # For embedding text queries

    def load_data(self, dataset_name=None):
        if dataset_name == self.current_dataset:
            # Data already loaded, no need to reload
            return

        self.current_dataset = dataset_name
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

        for asset in assets:
            quantized_embeddings = np.frombuffer(asset.embeddings, dtype=np.uint8)
            self.embeddings.append(quantized_embeddings)
            self.uris.append(asset.uri)

        self.embeddings = np.vstack(self.embeddings)
        self.fit()

    def fit(self):
        self.model.fit(self.embeddings)

    def search(self, query):
        if isinstance(query, str):  # Text query
            embedding = self.processor.compute_text_embeddings([query])[0]
            query_embedding = self.processor.quantize_embeddings(embedding)
        else:  # Image embedding query
            query_embedding = query

        distances, indices = self.model.kneighbors(query_embedding.reshape(1, -1), n_neighbors=self.n_neighbors)

        # Sort results by distance (similarity)
        sorted_results = sorted(zip(indices[0], distances[0]), key=lambda x: x[1])

        # Convert indices back to URIs
        results = [(self.uris[idx], dist) for idx, dist in sorted_results]

        return results

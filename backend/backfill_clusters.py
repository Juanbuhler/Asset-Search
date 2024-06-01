# backend/backfill_clusters.py
from .database import session, ImageAsset, add_cluster_column
from .data_processor import DataProcessor
import numpy as np


def backfill_clusters(dataset_name, num_clusters):
    add_cluster_column()

    processor = DataProcessor()

    # Fetch data from the database
    assets = session.query(ImageAsset).filter_by(dataset=dataset_name).all()
    embeddings = np.array([np.frombuffer(asset.embeddings, dtype=np.uint8) for asset in assets])

    # Perform KMeans clustering
    cluster_indices, distances = processor.perform_kmeans_clustering(embeddings, num_clusters)

    # Update database with cluster indices
    for asset, cluster_index, distance in zip(assets, cluster_indices, distances):
        asset.cluster_index = cluster_index
        asset.distance_to_centroid = distance
    session.commit()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python backfill_clusters.py <dataset_name> <num_clusters>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    num_clusters = int(sys.argv[2])

    backfill_clusters(dataset_name, num_clusters)

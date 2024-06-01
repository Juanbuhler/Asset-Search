# backend/search_manager.py
import os
import numpy as np
import urllib.parse
import threading
from http.server import SimpleHTTPRequestHandler
import socketserver

from sqlalchemy import func
from backend.database import session, ImageAsset
from backend.nearest_neighbors import NearestNeighborsSearch
from backend.data_processor import DataProcessor
import streamlit as st


@st.cache_resource(show_spinner=False)
class SearchManager:
    def __init__(self):
        self.searcher = NearestNeighborsSearch()

    def perform_similarity_search(self, query, dataset_name, n_neighbors):

        if dataset_name == "All":
            dataset_name = "%"
        self.searcher.n_neighbors = n_neighbors
        self.searcher.load_data(dataset_name=dataset_name)  # Only loads data if the dataset has changed

        if query.startswith("file://") or query.startswith("http://"):  # URI query
            query_image_asset = session.query(ImageAsset).filter(
                ImageAsset.uri.like(f"%{query}%")
            ).first()

            if query_image_asset:
                query_embedding = np.frombuffer(query_image_asset.embeddings, dtype=np.uint8)
                results = self.searcher.search(query_embedding)
            else:
                print("Query image not found in the database.")
                return []
        else:  # Text query
            print("Performing text-based search...")
            results = self.searcher.search(query)

        return results

    def perform_cluster_search(self, cluster_id, dataset_name, N):
        results = (
            session.query(ImageAsset)
            .filter_by(dataset=dataset_name, cluster_index=cluster_id)
            .order_by(ImageAsset.distance_to_centroid)
            .limit(N)
            .all()
        )
        return [(result.uri, result.distance_to_centroid) for result in results]

    def perform_clustering(self, dataset_name, num_clusters):
        assets = session.query(ImageAsset).filter_by(dataset=dataset_name).all()
        embeddings = np.array([np.frombuffer(asset.embeddings, dtype=np.uint8) for asset in assets])

        processor = DataProcessor()
        cluster_indices, distances = processor.perform_kmeans_clustering(embeddings, num_clusters)

        for asset, cluster_index, distance in zip(assets, cluster_indices, distances):
            asset.cluster_index = int(cluster_index)
            asset.distance_to_centroid = distance

        session.commit()

    def get_cluster_indices(self, dataset_name):
        clusters = session.query(ImageAsset.cluster_index).filter_by(dataset=dataset_name).distinct().all()
        return [cluster[0] for cluster in clusters if cluster[0] is not None]

    def get_available_datasets(self):
        datasets = session.query(ImageAsset.dataset).distinct().all()
        datasets = [dataset[0] for dataset in datasets]
        return datasets

    def count_assets_in_dataset(self, dataset_name):
        if dataset_name == "All":
            dataset_name = "%"
        return session.query(func.count(ImageAsset.id)).filter(ImageAsset.dataset.like(dataset_name)).scalar()

    def get_thumbnails(self, dataset_name,
                       n_images=None,
                       search_results=None,
                       base_url=None,
                       base_path=None,
                       page_id=0,
                       ):
        def convert_file_uri_to_http_url(file_uri):
            parsed_uri = urllib.parse.urlparse(file_uri)
            relative_path = os.path.relpath(parsed_uri.path, base_path)
            return urllib.parse.urljoin(base_url, relative_path)

        if dataset_name == "All":
            dataset_name = "%"
        if search_results:
            thumbnails = [convert_file_uri_to_http_url(thumbnail) for thumbnail in search_results]
            uris = search_results
        else:
            # Calculate the offset based on the page_id and n_assets
            offset_value = page_id * n_images

            # Query to fetch the specific page of results
            query_results = (session.query(ImageAsset.uri)
                             .filter(ImageAsset.dataset.like(dataset_name))
                             .offset(offset_value)
                             .limit(n_images)
                             .all())

            thumbnails = [convert_file_uri_to_http_url(thumbnail[0]) for thumbnail in query_results]
            uris = [thumbnail[0] for thumbnail in query_results]

        return thumbnails, uris

    def start_http_server(self, image_folder, port):
        os.chdir(image_folder)

        class Handler(SimpleHTTPRequestHandler):
            pass

        def serve(port):
            # We catch the OSError that would happen if the port is already in use
            # This will break is the port is in use for another purpose, but will be ok
            # in most cases where the streamlit app is running and the user wants to run a notebook as well
            with socketserver.TCPServer(("", port), Handler) as httpd:
                print(f"Serving HTTP on port {port}")
                httpd.serve_forever()


        thread = threading.Thread(target=serve, args=(port,))
        thread.daemon = True  # Ensure the thread exits when the main program does
        thread.start()
        return port

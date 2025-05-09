# backend/search_manager.py
import os
import numpy as np
import urllib.parse
import shutil
import threading
from http.server import SimpleHTTPRequestHandler
import socketserver

from sqlalchemy import func
from sqlalchemy.orm.exc import NoResultFound
from backend.database import session, ImageAsset
from backend.nearest_neighbors import NearestNeighborsSearch
from backend.data_processor import DataProcessor
from backend.config import DEFAULT_MODEL_TYPE, DATABASE_URI, ROOT_IMAGE_DIRECTORY
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

@st.cache_resource(show_spinner=False)
class SearchManager:
    def __init__(self):
        self.searcher = NearestNeighborsSearch()

    def perform_similarity_search(self, query, dataset_name, n_neighbors, embedding_type=DEFAULT_MODEL_TYPE, bias_query=""):

        if dataset_name == "All":
            dataset_name = "%"
        self.searcher.n_neighbors = n_neighbors

        embedding_attr = "embeddings"
        if embedding_type == "clip":
            embedding_attr = "embeddings"
        elif embedding_type == "resnet152":
            embedding_attr = "resnet_embeddings"
        else:
            raise ValueError(f"Unsupported model type: {embedding_type}")

        self.searcher.load_data(dataset_name=dataset_name, embedding_type=embedding_type)  # Only loads data if the dataset has changed

        if type(query) == np.ndarray:  # OpenCV image query
            results = self.searcher.search(query, bias=bias_query)
        elif query.startswith("file://") or query.startswith("http://"):  # URI query
            query_image_asset = session.query(ImageAsset).filter(
                ImageAsset.uri.like(f"%{query}%")
            ).first()

            if query_image_asset:
                embeddings = getattr(query_image_asset, embedding_attr)

                query_embedding = (np.frombuffer(embeddings, dtype=np.uint8) - 127.5) / 255
                results = self.searcher.search(query_embedding, bias=bias_query)
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

    def perform_outlier_search(self, dataset_name, N):
        results = (
            session.query(ImageAsset)
            .filter_by(dataset=dataset_name)
            .order_by(ImageAsset.distance_to_centroid.desc())
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

    def delete_image_assets_by_uris(self, uri_list):
        for uri in uri_list:
            try:
                # Find the image asset by URI
                image_asset = session.query(ImageAsset).filter(ImageAsset.uri == uri).one()
                # Delete the image asset
                session.delete(image_asset)
            except NoResultFound:
                print(f"No image asset found with URI: {uri}")
            except Exception as e:
                print(f"Error occurred while deleting URI: {uri}. Error: {e}")
        # Commit the changes to the database
        session.commit()

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

    def get_captions(self, uris):
        """
        Retrieve captions for a list of image URIs from the database.

        Args:
            uris (list): List of image URIs as stored in the database.

        Returns:
            dict: Mapping of URIs to their corresponding captions.
        """
        engine = create_engine(f'sqlite:///{DATABASE_URI}')
        Session = sessionmaker(bind=engine)
        session = Session()
        # Retrieve all relevant captions with a single query
        result = {img.uri: img.caption for img in
                  session.query(ImageAsset).filter(ImageAsset.uri.in_(uris)).all()}
        return result

    def copy_image_assets(self, uris, new_dataset_name):
        """
        Copies database entries corresponding to the given URIs to a new dataset
        and copies the image files to a new location.

        Args:
            uris (list): List of image URIs to copy.
            new_dataset_name (str): The name of the new dataset.
            DATABASE_URI (str): The URI of the database to copy to.
            ROOT_IMAGE_DIRECTORY (str): The root directory for image storage.
        """
        try:
            # Retrieve the ImageAsset objects from the database
            assets_to_copy = session.query(ImageAsset).filter(ImageAsset.uri.in_(uris)).all()

            # Create the new dataset directory if it doesn't exist
            new_dataset_directory = os.path.join(ROOT_IMAGE_DIRECTORY, new_dataset_name)
            os.makedirs(new_dataset_directory, exist_ok=True)

            # Create new ImageAsset objects with the new dataset name and copy files
            new_assets = []
            for asset in assets_to_copy:
                # Extract the file path from the URI
                file_path = asset.uri
                if file_path.startswith("file://"):
                    file_path = file_path[7:]  # Remove "file://" prefix

                source_file = os.path.join(os.path.dirname(file_path), os.path.basename(file_path))
                destination_file = os.path.join(new_dataset_directory, os.path.basename(file_path))

                # Copy the file
                shutil.copy2(source_file, destination_file)  # copy2 preserves metadata

                # Create new ImageAsset object
                new_asset = ImageAsset(**{key: value for key, value in asset.__dict__.items()
                                          if key != 'id' and key != '_sa_instance_state'})
                new_asset.dataset = new_dataset_name
                new_asset.uri = "file://" + destination_file  # Store the new file path
                new_assets.append(new_asset)

            # Add the new assets to the session
            session.add_all(new_assets)

            # Commit the changes to the database
            session.commit()

            print(f"Successfully copied {len(new_assets)} assets to dataset '{new_dataset_name}'")

        except Exception as e:
            session.rollback()
            print(f"Error copying assets: {e}")

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

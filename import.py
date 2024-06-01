# main.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from backend.database import initialize_database, add_cluster_column
from backend.data_ingestion import ingest_images_from_directory


def main():
    # Initialize the database and add cluster_index column
    initialize_database()
    add_cluster_column()
    print("Database initialized and cluster_index column ensured.")

    # Ingest images from a directory
    directory_path = input("Enter the path to the directory containing images: ")
    dataset_name = input("Enter the dataset name: ")
    ingest_images_from_directory(directory_path, dataset_name)
    print(f"Image URIs from {directory_path} have been successfully ingested into the '{dataset_name}' dataset.")


if __name__ == "__main__":
    main()

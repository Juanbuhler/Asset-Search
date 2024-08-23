# import.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from backend.database import initialize_database, add_cluster_column, add_resnet_column
from backend.data_ingestion import ingest_images_from_directory
from backend.config import DEFAULT_MODEL_TYPE


def main():
    initialize_database()
    add_cluster_column()
    add_resnet_column()
    print("Database initialized and cluster_index column ensured.")

    directory_path = input("Enter the path to the directory containing images: ")
    dataset_name = input("Enter the dataset name: ")
    model_type = input(f"Enter the model type (clip/resnet152) [{DEFAULT_MODEL_TYPE}]: ") or DEFAULT_MODEL_TYPE
    ingest_images_from_directory(directory_path, dataset_name, model_type)
    print(f"Image URIs from {directory_path} have been successfully ingested into the '{dataset_name}' dataset.")


if __name__ == "__main__":
    main()

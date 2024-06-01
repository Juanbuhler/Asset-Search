# assetSearch/backend/data_ingestion.py
import os
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from .database import session, ImageAsset
from .data_processor import DataProcessor
from .config import ROOT_IMAGE_DIRECTORY

# Set the batch size for processing images
BATCH_SIZE = 25


def process_batch(image_paths, image_uris, dataset_name, processor):
    total_images = len(image_paths)
    with tqdm(total=total_images, desc="Processing images") as pbar:
        for i in range(0, total_images, BATCH_SIZE):
            batch_paths = image_paths[i:i+BATCH_SIZE]
            batch_uris = image_uris[i:i+BATCH_SIZE]
            valid_paths = []
            valid_uris = []
            for image_path, image_uri in zip(batch_paths, batch_uris):
                try:
                    # Try opening the image to check for corruption
                    with Image.open(image_path) as img:
                        img.verify()  # Check if the image can be opened
                    valid_paths.append(image_path)
                    valid_uris.append(image_uri)
                except (IOError, UnidentifiedImageError, OSError) as e:
                    print(f"Warning: Skipping corrupted image {image_path}. Error: {e}")
            if valid_paths:
                embeddings = processor.compute_clip_embeddings(valid_paths)
                quantized_embeddings = processor.quantize_embeddings(embeddings)
                for uri, emb in zip(valid_uris, quantized_embeddings):
                    image_asset = ImageAsset(uri=uri, dataset=dataset_name, embeddings=emb.tobytes())
                    session.add(image_asset)
                session.commit()
            pbar.update(len(batch_paths))


def ingest_images_from_directory(directory_path, dataset_name):
    image_paths = []
    image_uris = []
    existing_uris = {asset.uri for asset in session.query(ImageAsset.uri).filter_by(dataset=dataset_name).all()}

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_uri = f"file://{image_path}"
                if image_uri not in existing_uris:
                    image_paths.append(image_path)
                    image_uris.append(image_uri)

    processor = DataProcessor()

    # Process images in batches
    process_batch(image_paths, image_uris, dataset_name, processor)

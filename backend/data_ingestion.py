# assetSearch/backend/data_ingestion.py
import os
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError
from .database import session, ImageAsset
from .data_processor import DataProcessor
from .config import ROOT_IMAGE_DIRECTORY, DEFAULT_MODEL_TYPE

# Set the batch size for processing images
BATCH_SIZE = 25


def process_batch(image_paths, image_uris, dataset_name, processor, model_type):
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
                embeddings = processor.compute_image_embeddings(valid_paths)
                quantized_embeddings = processor.quantize_embeddings(embeddings)

                # Define a dictionary to map model types to the corresponding attribute name
                embedding_attr_map = {
                    'clip': 'embeddings',
                    'resnet152': 'resnet_embeddings'
                }

                # Get the appropriate attribute name based on processor.model_type
                embedding_attr = embedding_attr_map.get(model_type, 'embeddings')

                for uri, emb in zip(valid_uris, quantized_embeddings):
                    # Check if the asset already exists
                    image_asset = session.query(ImageAsset).filter_by(uri=uri).first()

                    if image_asset:
                        # Update the existing asset
                        setattr(image_asset, embedding_attr, emb.tobytes())
                    else:
                        # Create a new asset
                        image_asset = ImageAsset(uri=uri, dataset=dataset_name)
                        setattr(image_asset, embedding_attr, emb.tobytes())
                        session.add(image_asset)
                session.commit()
            pbar.update(len(batch_paths))


def ingest_images_from_directory(directory_path, dataset_name, model_type=DEFAULT_MODEL_TYPE):
    image_paths = []
    image_uris = []
    existing_uris = {asset.uri for asset in session.query(ImageAsset.uri).filter_by(dataset=dataset_name).all()}

    for root, dirs, files in os.walk(directory_path):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                image_uri = f"file://{image_path}"
                if image_uri not in existing_uris or model_type != DEFAULT_MODEL_TYPE:
                    image_paths.append(image_path)
                    image_uris.append(image_uri)

    processor = DataProcessor(model_type=model_type)

    # Process images in batches
    process_batch(image_paths, image_uris, dataset_name, processor, model_type)

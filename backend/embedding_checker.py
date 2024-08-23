# backend/embedding_checker.py
from sqlalchemy import func, and_
from backend.database import session, ImageAsset


def check_populated_embeddings(dataset_name):
    """
    Checks which embeddings are populated for a given dataset.

    Args:
        dataset_name (str): The name of the dataset to check.

    Returns:
        list: A list of strings indicating which embeddings are populated.
    """
    populated_embeddings = []

    clip_populated = session.query(func.count(ImageAsset.id)).filter(
        and_(
            ImageAsset.dataset == dataset_name,
            ImageAsset.embeddings != None
        )
    ).scalar()

    resnet_populated = session.query(func.count(ImageAsset.id)).filter(
        and_(
            ImageAsset.dataset == dataset_name,
            ImageAsset.resnet_embeddings != None
        )
    ).scalar()

    if clip_populated > 0:
        populated_embeddings.append("clip")
    if resnet_populated > 0:
        populated_embeddings.append("resnet152")

    return populated_embeddings

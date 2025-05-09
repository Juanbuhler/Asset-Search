# backend/database.py
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, text, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
from sqlalchemy.exc import OperationalError
from .config import DATABASE_URI

Base = declarative_base()


class ImageAsset(Base):
    __tablename__ = 'image_assets'
    id = Column(Integer, primary_key=True)
    uri = Column(String, unique=True, nullable=False)
    dataset = Column(String, nullable=False)
    embeddings = Column(LargeBinary)  # CLIP, these are default
    resnet_embeddings = Column(LargeBinary)
    perceptual_embeddings = Column(LargeBinary)
    cluster_index = Column(Integer, nullable=True)  # New column for cluster indices
    distance_to_centroid = Column(Float, nullable=True)
    caption = Column(String, nullable=False)

    def __init__(self, uri, dataset, embeddings=None, resnet_embeddings=None, perceptual_embeddings=None,
                 cluster_index=None, distance_to_centroid=None, caption=None):
        self.uri = uri
        self.dataset = dataset
        self.embeddings = embeddings
        self.resnet_embeddings = resnet_embeddings
        self.perceptual_embeddings = perceptual_embeddings
        self.cluster_index = cluster_index
        self.distance_to_centroid = distance_to_centroid
        self.caption = caption


engine = create_engine(f'sqlite:///{DATABASE_URI}')
Session = sessionmaker(bind=engine)
session = Session()


def initialize_database():
    Base.metadata.create_all(engine)


def add_cluster_column():
    connection = engine.connect()
    try:
        connection.execute(text('ALTER TABLE image_assets ADD COLUMN distance_to_centroid FLOAT'))
        connection.execute(text('ALTER TABLE image_assets ADD COLUMN cluster_index INTEGER'))
    except OperationalError as e:
        if "duplicate column name" in str(e):
            # Column already exists
            pass
        else:
            raise
    finally:
        connection.close()


def add_resnet_column():
    connection = engine.connect()
    try:
        connection.execute(text('ALTER TABLE image_assets ADD COLUMN resnet_embeddings BLOB'))
        connection.execute(text('ALTER TABLE image_assets ADD COLUMN perceptual_embeddings BLOB'))
    except OperationalError as e:
        if "duplicate column name" in str(e):
            # Column already exists
            pass
        else:
            raise
    finally:
        connection.close()


def add_caption_column():
    connection = engine.connect()
    try:
        connection.execute(text('ALTER TABLE image_assets ADD COLUMN caption STRING'))
    except OperationalError as e:
        if "duplicate column name" in str(e):
            # Column already exists
            pass
        else:
            raise
    finally:
        connection.close()



# Call this function once to ensure the column exists
#add_cluster_column()

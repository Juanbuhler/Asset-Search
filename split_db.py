
import re
import os
import sqlalchemy as sa
from sqlalchemy.orm import sessionmaker, declarative_base, make_transient
from sqlalchemy.exc import OperationalError

# Define your database URI here
DATABASE_URI = 'backend/asset_search.db'

Base = declarative_base()


class ImageAsset(Base):
    __tablename__ = 'image_assets'
    id = sa.Column(sa.Integer, primary_key=True)
    uri = sa.Column(sa.String, unique=True, nullable=False)
    dataset = sa.Column(sa.String, nullable=False)
    embeddings = sa.Column(sa.LargeBinary)
    resnet_embeddings = sa.Column(sa.LargeBinary)
    perceptual_embeddings = sa.Column(sa.LargeBinary)
    cluster_index = sa.Column(sa.Integer, nullable=True)
    distance_to_centroid = sa.Column(sa.Float, nullable=True)

    def __init__(self, uri, dataset, embeddings=None):
        self.uri = uri
        self.dataset = dataset
        self.embeddings = embeddings


# Create engine and session for the existing database
engine = sa.create_engine(f'sqlite:///{DATABASE_URI}')
Session = sessionmaker(bind=engine)
session = Session()


def sanitize_dataset_name(dataset_name):
    # Replace non-alphanumeric characters with underscores
    return re.sub(r'\W+', '_', dataset_name)


def create_new_database_for_dataset(dataset_name, records):
    sanitized_name = sanitize_dataset_name(dataset_name)
    new_db_name = f'{sanitized_name}.db'

    # Create a new SQLite database
    new_engine = sa.create_engine(f'sqlite:///{new_db_name}')
    Base.metadata.create_all(new_engine)

    NewSession = sessionmaker(bind=new_engine)
    new_session = NewSession()

    # Make records transient before adding them to the new session
    for record in records:
        session.expunge(record)
        make_transient(record)

    # Add records to the new database
    new_session.add_all(records)
    new_session.commit()
    new_session.close()


def split_database_by_dataset():
    # Get all unique dataset names
    datasets = session.query(ImageAsset.dataset).distinct().all()

    for dataset in datasets:
        dataset_name = dataset[0]

        # Query all records belonging to this dataset
        records = session.query(ImageAsset).filter(ImageAsset.dataset == dataset_name).all()

        # Create a new database and add these records to it
        create_new_database_for_dataset(dataset_name, records)


# Split the database
split_database_by_dataset()

# Close the session
session.close()

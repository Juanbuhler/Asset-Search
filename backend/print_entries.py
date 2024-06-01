# assetSearch/backend/print_entries.py
from .database import session, ImageAsset

def print_database_entries(dataset_name=None, limit=10):
    query = session.query(ImageAsset)
    if dataset_name:
        query = query.filter_by(dataset=dataset_name)
    entries = query.limit(limit).all()
    for entry in entries:
        print(f"ID: {entry.id}, URI: {entry.uri}, Dataset: {entry.dataset}, Cluster: {entry.cluster_index}")

if __name__ == "__main__":
    import sys
    dataset_name = None
    limit = 10  # Default limit
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        try:
            limit = int(sys.argv[2])
        except ValueError:
            print("Usage: python print_entries.py [dataset_name] [limit]")
            sys.exit(1)

    print_database_entries(dataset_name, limit)

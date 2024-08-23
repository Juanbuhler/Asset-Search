# assetSearch/delete.py
import sys
import os
from tqdm import tqdm
from backend.database import session, ImageAsset


def delete_dataset(dataset_name):
    # Confirm deletion
    confirmation = input(f"Are you sure you want to delete the dataset '{dataset_name}'? This action cannot be undone. (yes/no): ")
    if confirmation.lower() != 'yes':
        print("Deletion cancelled.")
        return

    # Query all entries with the given dataset name
    entries = session.query(ImageAsset).filter_by(dataset=dataset_name).all()
    total_entries = len(entries)
    if total_entries == 0:
        print(f"No entries found for dataset '{dataset_name}'.")
        return

    # Delete entries with a progress bar
    with tqdm(total=total_entries, desc="Deleting entries") as pbar:
        for entry in entries:
            session.delete(entry)
            session.commit()
            pbar.update(1)

    print(f"All entries for dataset '{dataset_name}' have been deleted.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python delete.py <dataset_name>")
        sys.exit(1)

    dataset_name = sys.argv[1]
    delete_dataset(dataset_name)

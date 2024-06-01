
# Asset Search

Asset Search is a tool for managing and searching a collection of image assets using CLIP embeddings and KMeans clustering. It includes a Streamlit web application for interactive searching and viewing of image assets.

## Folder Structure

```
.
├── Asset Search.ipynb
├── README.md
├── app.py
├── import.py
├── backend
│   ├── __init__.py
│   ├── asset_search.db
│   ├── backfill_clusters.py
│   ├── config.py
│   ├── data_ingestion.py
│   ├── data_processor.py
│   ├── database.py
│   ├── nearest_neighbors.py
│   ├── print_entries.py
│   └── search_manager.py
```

## Getting Started

### Installation

1. Clone the repository and install dependencies:
   ```bash
   git clone https://github.com/yourusername/asset-search.git
   cd asset-search
   ```
    ```bash
    pip install -r requirements.txt
    ```   

2. Edit ROOT_IMAGE_DIRECTORY in backend/config.py to point to the root directory where your image datasets are. 

3. Set up the database and ingest images:
   ```bash
   python import.py
   ```
   Follow the prompts to enter the path to your image directory and provide a dataset name.

### Running the Application

To run the Streamlit application, execute:

```bash
streamlit run app.py
```

### Usage

1. Open the Streamlit web application in your browser (usually at `http://localhost:8501`).

2. Use the sidebar to select a dataset, adjust the number of assets to show, and thumbnail size.

3. Enter text queries to perform a similarity search using CLIP embeddings or select a cluster to view similar images.

4. Click on thumbnails to perform further similarity searches based on the selected image.

### Files Description

- `app.py`: Main entry point for the Streamlit web application.
- `main.py`: Script for initializing the database and ingesting images.
- `backend/`: Directory containing backend modules and scripts.
  - `config.py`: Configuration file for database paths.
  - `data_ingestion.py`: Script for ingesting images and computing embeddings.
  - `data_processor.py`: Handles image and text embedding computations using CLIP.
  - `database.py`: Database setup using SQLAlchemy.
  - `nearest_neighbors.py`: Implements nearest neighbors search using sklearn.
  - `search_manager.py`: Manages search operations and clustering.
  - `backfill_clusters.py`: Script to backfill clusters for existing datasets.
  - `print_entries.py`: Utility to print database entries for debugging.

### TO DO

- Add different embedding types
- Support other data modalities (at least short snippets of text, using CLIP embeddings)
- Add dimensionality reduction and a projector view.

### License

This project is licensed under the MIT License.

### Acknowledgements

- [OpenAI CLIP](https://github.com/openai/CLIP)
- [Streamlit](https://streamlit.io/)
- [SQLAlchemy](https://www.sqlalchemy.org/)
- [scikit-learn](https://scikit-learn.org/)

```

Feel free to customize this `README.md` with additional details specific to your project as needed.
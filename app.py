# app.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from st_clickable_images import clickable_images
from backend.search_manager import SearchManager
from backend.config import ROOT_IMAGE_DIRECTORY
from backend.util import get_collage

st.set_page_config(layout='wide')


@st.cache_resource(show_spinner=False)
def get_search_manager():
    search_manager = SearchManager()
    return search_manager


if "port" not in st.session_state:
    search_manager = get_search_manager()
    port = 8000
    st.session_state.port = search_manager.start_http_server(ROOT_IMAGE_DIRECTORY, port)


def convert_file_uri_to_data_url(file_uri):
    return file_uri.replace(f"file://{ROOT_IMAGE_DIRECTORY}", "http://localhost:8000")



def main():
    search_manager = get_search_manager()
    available_datasets = search_manager.get_available_datasets()

    display_section = st.sidebar.expander('Display Controls',)
    cluster_section = st.sidebar.expander('Clustering Controls')

    if "current_dataset" not in st.session_state:
        st.session_state["current_dataset"] = available_datasets[0]
        st.session_state["search_results"] = None
        st.session_state["clicked"] = -1
        st.session_state["text_query"] = ""
        st.session_state["clip_key"] = 1
        st.session_state["num_clusters"] = 10


    selected_dataset = display_section.selectbox("Select Dataset", ["All"] + available_datasets)
    n_neighbors = display_section.slider("Number of Assets to Show", min_value=10, value=50)
    thumbnail_size = display_section.slider("Thumbnail Size", min_value=50, max_value=250, step=50, value=100)

    if st.session_state.current_dataset != selected_dataset:
        st.session_state.current_dataset = selected_dataset
        st.session_state.search_results = None

    st.title(selected_dataset)
    num_assets = search_manager.count_assets_in_dataset(selected_dataset)
    st.write(f"{num_assets} images")

    clip_key = st.session_state.clip_key
    text_query = st.text_input("Enter text to search", key=f"{clip_key}")
    if text_query != "":
        clip_key += 1
        st.session_state.clip_key = clip_key
        similarity_results = search_manager.perform_similarity_search(text_query, selected_dataset, n_neighbors)
        st.session_state.search_results = [result[0] for result in similarity_results]
        st.rerun()

    display_page = 0
    if st.session_state.search_results:
        if st.button("Reset Search"):
            st.session_state.search_results = None
            st.session_state.clicked = -1
            st.session_state.text_query = ""
            st.rerun()
    else:
        display_page = st.slider('Page', min_value=0, max_value=int(num_assets/n_neighbors))

    # Cluster search input
    if "cluster_selection" not in st.session_state:
        st.session_state.cluster_selection = "None"

    cluster_id = cluster_section.selectbox("Select Cluster", ["None"] + [str(i) for i in range(0, 20)])
    if cluster_id != st.session_state.cluster_selection:
        if cluster_id == "None":
            st.session_state.cluster_selection = "None"
            st.session_state.search_results = None

        else:
            cluster_results = search_manager.perform_cluster_search(int(cluster_id), selected_dataset, n_neighbors)
            st.session_state.search_results = [result[0] for result in cluster_results]
            st.session_state.cluster_selection = cluster_id
        st.rerun()

    # Clustering controls
    num_clusters = cluster_section.slider("Number of Clusters", min_value=5, max_value=200, step=5,
                                     value=st.session_state.num_clusters)
    if cluster_section.button("Compute Clusters"):
        st.session_state.num_clusters = num_clusters
        search_manager.perform_clustering(selected_dataset, num_clusters)
        st.success(f"Computed {num_clusters} clusters for dataset '{selected_dataset}'")
        st.session_state.search_results = None
        st.session_state.clicked = -1
        st.rerun()

    # Cluster Collages
    cluster_section.divider()
    cluster_section.text("Cluster Collages")
    N = cluster_section.slider("Images per side", min_value=4, max_value=20, value=5)
    if cluster_section.button("Make Cluster Collages"):
        clusters = search_manager.get_cluster_indices(selected_dataset)
        if clusters:
            n_cols = 5
            cols = st.columns(n_cols)
            i_col = 0
            for cluster_id in clusters:
                imgs = search_manager.perform_cluster_search(cluster_id, selected_dataset, N*N)
                img_paths = [im[0][7:] for im in imgs]
                collage = get_collage(img_paths, N)
                cols[i_col].image(collage)
                i_col += 1
                if i_col >= n_cols:
                    i_col = 0
        else:
            st.text("Please compute clusters first.")

    if cluster_section.button("Display outliers"):
        clusters = search_manager.get_cluster_indices(selected_dataset)
        if clusters:
            outlier_results = search_manager.perform_outlier_search(selected_dataset, n_neighbors)
            st.session_state.search_results = [result[0] for result in outlier_results]
        else:
            st.text("Please compute clusters first.")
        st.rerun()

    # Displaying thumbnails
    thumbnails, image_uris = search_manager.get_thumbnails(selected_dataset, n_images=n_neighbors,
                                                           search_results=st.session_state.search_results,
                                                           page_id=display_page)

    data_urls = [convert_file_uri_to_data_url(uri) for uri in image_uris]

    if 'clicked' not in st.session_state:
        st.session_state["clicked"] = -1
    old_clicked = st.session_state.clicked

    clicked = clickable_images(
        data_urls,
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": f"{thumbnail_size}px"},
        key=f'thumbnails{clip_key}'
    )

    if clicked > -1 and clicked != old_clicked:
        st.session_state.clip_key += 1
        selected_image_uri = image_uris[clicked]
        similarity_results = search_manager.perform_similarity_search(selected_image_uri, selected_dataset, n_neighbors)
        st.session_state.search_results = [result[0] for result in similarity_results]
        st.session_state.clicked = clicked
        st.rerun()


if __name__ == "__main__":
    main()

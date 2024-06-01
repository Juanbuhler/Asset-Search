# app.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from st_clickable_images import clickable_images
from backend.search_manager import SearchManager
from backend.config import ROOT_IMAGE_DIRECTORY


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


    if "current_dataset" not in st.session_state:
        st.session_state["current_dataset"] = available_datasets[0]
        st.session_state["search_results"] = None
        st.session_state["clicked"] = -1
        st.session_state["text_query"] = ""
        st.session_state["clip_key"] = 1
        st.session_state["num_clusters"] = 10


    selected_dataset = st.sidebar.selectbox("Select Dataset", ["All"] + available_datasets)
    n_neighbors = st.sidebar.slider("Number of Assets to Show", min_value=10, value=50)
    thumbnail_size = st.sidebar.slider("Thumbnail Size", min_value=50, max_value=250, step=50, value=100)

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

    cluster_id = st.sidebar.selectbox("Select Cluster", ["None"] + [str(i) for i in range(0, 20)])
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
    num_clusters = st.sidebar.slider("Number of Clusters", min_value=2, max_value=20, step=1,
                                     value=st.session_state.num_clusters)
    if st.sidebar.button("Compute Clusters"):
        st.session_state.num_clusters = num_clusters
        search_manager.perform_clustering(selected_dataset, num_clusters)
        st.success(f"Computed {num_clusters} clusters for dataset '{selected_dataset}'")
        st.session_state.search_results = None
        st.session_state.clicked = -1
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

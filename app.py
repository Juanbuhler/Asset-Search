# app.py
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from st_click_detector import click_detector
from backend.search_manager import SearchManager
from backend.config import ROOT_IMAGE_DIRECTORY, DATABASE_URI
from backend.util import get_collage
from backend.embedding_checker import check_populated_embeddings
from PIL import Image
import requests
from io import BytesIO
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2


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

    display_section = st.sidebar.expander('Search Controls', expanded=True)
    cluster_section = st.sidebar.expander('Clustering Controls')
    danger_section = st.sidebar.expander('Dataset Management')

    if "current_dataset" not in st.session_state:
        st.session_state["current_dataset"] = available_datasets[0]
        st.session_state["search_results"] = None
        st.session_state["clicked"] = -1
        st.session_state["text_query"] = ""
        st.session_state["clip_key"] = 1
        st.session_state["num_clusters"] = 10

    selected_dataset = display_section.selectbox("Select Dataset", available_datasets)
    available_embeddings = check_populated_embeddings(selected_dataset)
    embedding_type = display_section.radio("Select Embedding", available_embeddings)
    n_neighbors = display_section.slider("Number of Assets to Show", min_value=10, value=20)
    bias_query = display_section.checkbox("Use Text Query to Bias Search")
    thumbnail_size = display_section.slider("Thumbnail Size", min_value=50, max_value=300, step=50, value=200)
    show_captions = display_section.checkbox("Show Captions")

    if "targeted_search_selection" not in st.session_state:
        st.session_state["targeted_search_selection"] = False

    if "do_targeted_search" not in st.session_state:
        st.session_state["do_targeted_search"] = False

    if st.session_state["do_targeted_search"]:
        if display_section.button("Cancel targeted search"):
            st.session_state["do_targeted_search"] = False
            st.rerun()
    else:
        if display_section.button("Select one image for targeted search"):
            st.session_state["targeted_search_selection"] = True

    if st.session_state.current_dataset != selected_dataset:
        st.session_state.current_dataset = selected_dataset
        st.session_state.search_results = None

    if st.session_state["targeted_search_selection"]:
        st.title("Click One Image for Targeted Search")
    else:
        st.title(selected_dataset)
    num_assets = search_manager.count_assets_in_dataset(selected_dataset)
    st.write(f"{num_assets} images")

    clip_key = st.session_state.clip_key

    # Only show text query if embedding is CLIP
    if embedding_type == "clip":
        text_query = st.text_input("Enter text to search", key=f"{clip_key}", value=st.session_state.text_query)
        if text_query != "" and text_query != st.session_state.text_query and not bias_query:
            clip_key += 1
            st.session_state.clip_key = clip_key
            st.session_state.text_query = text_query
            similarity_results = search_manager.perform_similarity_search(text_query,
                                                                          selected_dataset,
                                                                          n_neighbors,
                                                                          embedding_type=embedding_type)
            st.session_state.search_results = [result[0] for result in similarity_results]
            st.rerun()
        else:
            st.session_state.text_query = text_query

    if st.session_state["do_targeted_search"]:
        # Specify canvas parameters in application
        stroke_width = 3
        stroke_color = '#f22'
        bg_color = "#eee"
        drawing_mode = "rect"
        realtime_update = True

        target_image = st.session_state["target_image"].split('://')[1]
        img = cv2.imread(target_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.1)",  # Fixed fill color with some opacity
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            update_streamlit=realtime_update,
            width=800,
            height=600,
            drawing_mode=drawing_mode,
            key="canvas",
            background_image=Image.open(target_image) if target_image else None,
        )
        if canvas_result.json_data is not None:
            if canvas_result.json_data["objects"]:
                rect = canvas_result.json_data["objects"][-1]
                x = rect['left']
                y = rect['top']
                w = rect['width']
                h = rect['height']
                x = int(x * img.shape[1] / 800)
                y = int(y * img.shape[0] / 600)
                w = int(w * img.shape[1] / 800)
                h = int(h * img.shape[0] / 600)
                cropped = img[y:y + h, x:x + w]
                similarity_results = search_manager.perform_similarity_search(cv2.resize(cropped, (224, 224)),
                                                                              selected_dataset,
                                                                              n_neighbors,
                                                                              embedding_type=embedding_type)
                st.session_state.search_results = [result[0] for result in similarity_results]
                st.session_state.clicked = -1


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

    new_dataset = danger_section.text_input("New Dataset Name")
    if danger_section.button("Copy images to new dataset"):
        search_manager.copy_image_assets(image_uris, new_dataset)
    if danger_section.button("Delete Displayed Images From Database"):
        if st.session_state.search_results:
            search_manager.delete_image_assets_by_uris(st.session_state.search_results)
            st.session_state.search_results = None
            st.session_state.clicked = -1
            st.session_state.text_query = ""
            st.rerun()
        else:
            st.title("Perform a search before trying to delete assets")

    if 'clicked' not in st.session_state:
        st.session_state["clicked"] = -1
    old_clicked = st.session_state.clicked

    if "images_selection" not in st.session_state:
        st.session_state["images_selection"] = []

    if "select_images" not in st.session_state:
        st.session_state["select_images"] = False

    if danger_section.button("Select Images"):
        st.session_state["select_images"] = True


    html_images = ''' <style>
        .image-container {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #111;
            display: inline-block;
            max-width: 300px;
            width: auto;
        }
        .image-container-selected {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #111;
            border-color: #FFF;
            border: 100px;
            display: inline-block;
            max-width: 300px;
            width: auto;
        }
        .image-container img {
            display: block;
            margin-right: 0;
            margin-bottom: 10px;
        }

        .caption {
            overflow-wrap: break-word;
            word-wrap: break-word;
            word-break: break-all;
            width: 100%;
            white-space: normal;
        }
    </style> '''

    if show_captions:
        captions = search_manager.get_captions(image_uris)
    else:
        captions = {uri: "" for uri in image_uris}

    for id, url in enumerate(data_urls):
        html_image = f'''
            <div class="image-container">
                <a href="#" id="{id}">
                    <img style="height: {thumbnail_size}px; border: 10px solid #111111" src="{url}" alt="Image {id}">
                </a>
                <div class="caption">
                    <p>{captions[image_uris[id]]}</p>
                </div>
            </div>
        '''
        if url in st.session_state["images_selection"]:
            html_image = f'''
                <div class="image-container">
                    <a href="#" id="{id}">
                        <img style="height: {thumbnail_size}px; border: 10px solid #FF1111" src="{url}" alt="Image {id}">
                    </a>
                    <div class="caption">
                        <p>{captions[image_uris[id]]}</p>
                    </div>
                </div>
            '''
        html_images += html_image

    clicked = click_detector(html_images)

    if st.session_state["select_images"]:
        if clicked and clicked != old_clicked:
            if image_uris[int(clicked)] in st.session_state["images_selection"]:
                st.session_state["images_selection"] = [u for u in st.session_state["images_selection"]
                                                        if u != image_uris[int(clicked)]]
            else:
                st.session_state["images_selection"].append(image_uris[int(clicked)])
            print(st.session_state["images_selection"])
            st.session_state.clicked = clicked
            st.rerun()

    #clicked = clickable_images(
    #    data_urls,
    #    div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
    #   img_style={"margin": "5px", "height": f"{thumbnail_size}px"},
    #    key=f'thumbnails{clip_key}'
    #)


    if clicked and clicked != old_clicked:
        if st.session_state["targeted_search_selection"]:
            st.session_state["targeted_search_selection"] = False
            st.session_state["do_targeted_search"] = True
            st.session_state["target_image"] = image_uris[int(clicked)]
            st.rerun()
        else:
            bias_query_text = ""
            if bias_query:
                bias_query_text = text_query
            else:
                st.session_state.text_query = ""
            st.session_state.clip_key += 1
            selected_image_uri = image_uris[int(clicked)]
            similarity_results = search_manager.perform_similarity_search(selected_image_uri,
                                                                          selected_dataset,
                                                                          n_neighbors, bias_query=bias_query_text,
                                                                          embedding_type=embedding_type)
            st.session_state.search_results = [result[0] for result in similarity_results]
            st.session_state.clicked = clicked
            st.rerun()


if __name__ == "__main__":
    main()

import streamlit as st
import pickle
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
import os
import faiss

@st.cache_resource
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    return model, processor

model, processor = load_model()

@st.cache_data
def load_vectors():
    data_path = r"frame_vectors.pkl"
    with open(data_path, "rb") as f:
        vectors, names = pickle.load(f)
    return vectors, names

frame_vectors, frame_names = load_vectors()
frame_folder = r"frames"

@st.cache_resource
def build_faiss_index(vectors):
    norm_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    dim = norm_vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(norm_vectors.astype('float32'))
    return index

faiss_index = build_faiss_index(frame_vectors)


def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt", truncation=True)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features.squeeze().cpu().numpy()


def search_with_faiss(query_vector, top_k=50):
    q_norm = query_vector / np.linalg.norm(query_vector)
    D, I = faiss_index.search(np.expand_dims(q_norm.astype('float32'), axis=0), top_k)
    return I[0], D[0]


def temporal_match(indices, scores, frame_names, max_gap=5):
    frame_ids = [int(os.path.splitext(frame_names[i])[0].split('_')[-1]) for i in indices]
    sorted_data = sorted(zip(frame_ids, indices, scores))  

    n = len(sorted_data)
    dp = [s for _, _, s in sorted_data]
    prev = [-1] * n

    for i in range(n):
        for j in range(i):
            if 0 < sorted_data[i][0] - sorted_data[j][0] <= max_gap:
                if dp[j] + sorted_data[i][2] > dp[i]:
                    dp[i] = dp[j] + sorted_data[i][2]
                    prev[i] = j


    max_idx = np.argmax(dp)
    best_path = []
    while max_idx != -1:
        best_path.append(sorted_data[max_idx][1])
        max_idx = prev[max_idx]

    return best_path[::-1] 


st.set_page_config(page_title="Video Search", layout="centered")
st.title("Semantic Video Search")


query = st.text_input(" EX: ", placeholder="SKIBIDI")

if query:
    with st.spinner("FINDING..."):
        q_vec = encode_text(query)
        top_indices, top_scores = search_with_faiss(q_vec)

        matched_indices = temporal_match(top_indices, top_scores, frame_names)

    st.success(f"Tìm thấy {len(matched_indices)} khung hình liên tiếp!")

    for idx in matched_indices:
        img_path = os.path.join(frame_folder, frame_names[idx])
        st.image(img_path, caption=frame_names[idx], use_column_width=True)

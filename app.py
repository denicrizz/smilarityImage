import os
import numpy as npd
import streamlit as st
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# Load pretrained VGG16 model
@st.cache_resource
def load_model():
    return VGG16(weights='imagenet', include_top=False)

model = load_model()

# Fungsi untuk mengekstraksi fitur dari gambar
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Fungsi untuk memuat gambar dan mengekstraksi fitur dari semua gambar
@st.cache_data
def load_images_and_extract_features(image_folder):
    image_features = []
    image_paths = []
    
    for img_file in os.listdir(image_folder):
        if img_file.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_folder, img_file)
            image_paths.append(img_path)
            features = extract_features(img_path, model)
            image_features.append(features)
    
    return image_paths, np.array(image_features)

# Fungsi untuk menemukan gambar serupa berdasarkan cosine similarity
def find_similar_images(input_features, features_list, image_paths):
    similarities = cosine_similarity([input_features], features_list)[0]
    sorted_indices = np.argsort(similarities)[::-1]
    return [(image_paths[i], similarities[i]) for i in sorted_indices]

# Streamlit app
def main():
    st.title("Similarity Image")

    # Pastikan folder uploads ada
    os.makedirs("uploads", exist_ok=True)

    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Simpan gambar yang diunggah
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        filepath = os.path.join("uploads", uploaded_file.name)

        # Tampilkan gambar yang diunggah
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

        # Ekstraksi fitur dari gambar input
        input_features = extract_features(filepath, model)

        # Memuat dataset gambar dan fitur mereka
        dataset_folder = 'static/images/'  # Folder dataset gambar
        image_paths, image_features = load_images_and_extract_features(dataset_folder)

        # Temukan gambar serupa
        similar_images = find_similar_images(input_features, image_features, image_paths)

        # Hanya ambil 5 gambar teratas yang mirip
        similar_images = similar_images[:5]

        # Tampilkan gambar serupa
        st.subheader("5 gambar yang mungkin sama:")
        cols = st.columns(5)
        for idx, (img_path, similarity) in enumerate(similar_images):
            with cols[idx]:
                st.image(img_path, caption=f"Kesamaan: {similarity:.2f}", use_column_width=True)

if __name__ == '__main__':
    main()

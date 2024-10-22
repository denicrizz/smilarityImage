import os
import numpy as np
import streamlit as st
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Load pretrained VGG16 model
@st.cache_resource
def load_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_model.layers:
        layer.trainable = False  # Freeze the base model layers
    return base_model

model = load_model()

# Function to extract features from an image
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Function to load images and extract features from all images
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

# Function to find similar images based on cosine similarity
def find_similar_images(input_features, features_list, image_paths):
    similarities = cosine_similarity([input_features], features_list)[0]
    return [(image_paths[i], similarities[i]) for i in range(len(similarities))]

# Streamlit app
def main():
    st.title("Klasifikasi Kemiripan Gambar Bunga Mawar Merah")

    # Ensure uploads folder exists
    os.makedirs("uploads", exist_ok=True)

    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Save uploaded image
        with open(os.path.join("uploads", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        filepath = os.path.join("uploads", uploaded_file.name)

        # Display uploaded image
        st.image(uploaded_file, caption="Gambar yang Diunggah", use_column_width=True)

        # Extract features from input image
        input_features = extract_features(filepath, model)

        # Load dataset images and their features
        dataset_folder = 'static/images/'  # Folder containing dataset images
        image_paths, image_features = load_images_and_extract_features(dataset_folder)

        # Find similar images
        similar_images = find_similar_images(input_features, image_features, image_paths)

        # Filter out images with similarity less than 20%
        similar_images = [(img_path, similarity) for img_path, similarity in similar_images if similarity >= 0.20]

        # Check if there are similar images
        if not similar_images:
            st.subheader("Tidak ada gambar yang mirip.")
        else:
            # Only take the top 5 similar images
            similar_images = sorted(similar_images, key=lambda x: x[1], reverse=True)[:5]

            # Display similar images with their similarity percentages
            st.subheader("5 gambar yang mungkin sama:")
            cols = st.columns(5)
            for idx, (img_path, similarity) in enumerate(similar_images):
                with cols[idx]:
                    st.image(img_path, caption=f"Kesamaan: {similarity * 100:.2f}%", use_column_width=True)

if __name__ == '__main__':
    main()
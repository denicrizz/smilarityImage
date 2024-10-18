import os
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import matplotlib.pyplot as plt

# Inisialisasi Flask app
app = Flask(__name__)

# Folder untuk menyimpan gambar yang diupload
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pretrained VGG16 model
model = VGG16(weights='imagenet', include_top=False)

# Fungsi untuk mengekstraksi fitur dari gambar
def extract_features(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_data = image.img_to_array(img)
    img_data = np.expand_dims(img_data, axis=0)
    img_data = preprocess_input(img_data)
    features = model.predict(img_data)
    return features.flatten()

# Fungsi untuk memuat gambar dan mengekstraksi fitur dari semua gambar
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

# Route untuk halaman utama
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Jika pengguna mengunggah gambar
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            # Simpan gambar yang diunggah
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Ekstraksi fitur dari gambar input
            input_features = extract_features(filepath, model)
            
            # Memuat dataset gambar dan fitur mereka
            dataset_folder = 'static/images/'  # Folder dataset gambar
            image_paths, image_features = load_images_and_extract_features(dataset_folder)
            
            # Temukan gambar serupa
            similar_images = find_similar_images(input_features, image_features, image_paths)
            
            # Hanya ambil 5 gambar teratas yang mirip
            similar_images = similar_images[:5]
            
            # Kirim data gambar serupa ke frontend
            return render_template('index.html', uploaded_image=filename, similar_images=similar_images)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

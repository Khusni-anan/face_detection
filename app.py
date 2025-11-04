import streamlit as st
import tensorflow as tf
import joblib
import json
import numpy as np
import cv2
from PIL import Image

# --- 1. Konfigurasi Halaman ---
# Mengatur judul tab dan ikon
st.set_page_config(page_title="Deteksi Ekspresi Wajah", page_icon="😊")

# --- 2. Fungsi Cache untuk Model (PENTING) ---
# @st.cache_resource akan menyimpan model di memori, 
# sehingga tidak perlu di-load ulang setiap kali ada interaksi.

@st.cache_resource
def load_cnn_model():
    """Memuat model CNN Keras untuk ekstraksi fitur."""
    try:
        # Muat model Keras yang sudah dilatih
        model = tf.keras.models.load_model('emotion_feature_extractor.h5')
        return model
    except Exception as e:
        st.error(f"Error memuat model CNN (emotion_feature_extractor.h5): {e}")
        st.error("Pastikan file tersebut ada di direktori yang sama dengan app.py")
        return None

@st.cache_resource
def load_rf_model():
    """Memuat model Random Forest (Scikit-learn)."""
    try:
        # Muat model .pkl
        model = joblib.load('emotion_rf_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error memuat model RF (emotion_rf_model.pkl): {e}")
        st.error("Pastikan file tersebut ada di direktori yang sama dengan app.py")
        return None

@st.cache_resource
def load_labels():
    """Memuat file JSON yang berisi pemetaan label emosi."""
    try:
        with open('emotion_labels.json', 'r') as f:
            labels = json.load(f)
        return labels
    except Exception as e:
        st.error(f"Error memuat label (emotion_labels.json): {e}")
        st.error("Pastikan file tersebut ada di direktori yang sama dengan app.py")
        return None

# --- 3. Fungsi Preprocessing dan Prediksi ---
def process_and_predict(image_bytes, cnn_model, rf_model, labels):
    """
    Mengambil byte gambar, memprosesnya, dan mengembalikan 
    prediksi emosi beserta tingkat keyakinannya.
    """
    try:
        # 1. Decode image bytes ke array NumPy
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Baca sebagai gambar berwarna (BGR)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 2. Konversi ke Grayscale (sesuai data training)
        gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        
        # 3. Resize ke 96x96 (sesuai data training)
        resized_img = cv2.resize(gray_img, (96, 96))
        
        # 4. Reshape untuk model CNN (batch_size, height, width, channels)
        img_array = np.reshape(resized_img, (1, 96, 96, 1))
        
        # 5. Normalisasi pixel (skala 0-1)
        img_array = img_array / 255.0
        
        # --- Rantai Prediksi ---
        # 6. Dapatkan fitur dari CNN
        # Model CNN akan mengubah gambar (1, 96, 96, 1) menjadi vektor fitur
        features = cnn_model.predict(img_array)
        
        # 7. Prediksi kelas menggunakan Random Forest
        # rf_model.predict() mengembalikan array, misal [3]
        prediction_idx = rf_model.predict(features)
        
        # 8. Dapatkan probabilitas untuk semua kelas
        probabilities = rf_model.predict_proba(features)
        
        # 9. Dapatkan hasil
        emotion_label = str(prediction_idx[0]) # Konversi ke string untuk key JSON
        emotion_name = labels.get(emotion_label, "Tidak Dikenali")
        
        # Ambil probabilitas dari kelas yang diprediksi
        confidence = probabilities[0][prediction_idx[0]]
        
        return emotion_name, confidence

    except Exception as e:
        st.error(f"Error saat memproses gambar: {e}")
        return None, None

# --- 4. Tampilan Utama (Main UI) ---
def main():
    st.title("😊 Aplikasi Deteksi Ekspresi Wajah")
    st.write("""
    Upload gambar wajah untuk mendeteksi ekspresi.
    Model ini dilatih pada dataset CK+ menggunakan arsitektur hibrida CNN + Random Forest.
    """)

    # Load semua model dan label saat aplikasi dimulai
    cnn_model = load_cnn_model()
    rf_model = load_rf_model()
    labels = load_labels()

    # Jika salah satu file gagal dimuat, hentikan aplikasi
    if cnn_model is None or rf_model is None or labels is None:
        st.warning("Model atau file label gagal dimuat. Aplikasi tidak dapat berjalan.")
        st.stop() # Menghentikan eksekusi script

    # --- Sidebar untuk Upload ---
    st.sidebar.header("Upload Gambar Anda")
    uploaded_file = st.sidebar.file_uploader(
        "Pilih file gambar (JPG, JPEG, PNG)...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Baca file yang di-upload sebagai bytes
        image_bytes = uploaded_file.getvalue()
        
        # Tampilkan gambar yang di-upload
        st.image(image_bytes, caption='Gambar yang Di-upload', use_column_width=True)
        
        # Tombol untuk memulai deteksi
        if st.button('Deteksi Ekspresi'):
            # Tampilkan spinner selagi model bekerja
            with st.spinner('Menganalisis ekspresi...'):
                emotion, confidence = process_and_predict(image_bytes, cnn_model, rf_model, labels)
            
            if emotion:
                st.success(f"**Ekspresi Terdeteksi:**")
                
                # Tampilkan hasil menggunakan st.metric untuk tampilan yang lebih bagus
                st.metric(label="Ekspresi", value=emotion.capitalize())
                st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2%}")

    else:
        st.info('Silakan upload gambar melalui sidebar untuk memulai deteksi.')
        
        st.subheader("Label Emosi yang Dikenali Model:")
        # Menampilkan mapping label dari file JSON
        st.json(labels)

if __name__ == "__main__":
    main()

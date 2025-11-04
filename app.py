import streamlit as st
import tensorflow as tf
import joblib
import json
import numpy as np
import cv2
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- 1. Konfigurasi Halaman ---
st.set_page_config(page_title="Deteksi Ekspresi Wajah", page_icon="😊")

# --- 2. Fungsi Cache untuk Model (PENTING) ---
@st.cache_resource
def load_cnn_model():
    """Memuat model CNN Keras untuk ekstraksi fitur."""
    try:
        model = tf.keras.models.load_model('emotion_feature_extractor.h5')
        return model
    except Exception as e:
        st.error(f"Error memuat model CNN (emotion_feature_extractor.h5): {e}")
        return None

@st.cache_resource
def load_rf_model():
    """Memuat model Random Forest (Scikit-learn)."""
    try:
        model = joblib.load('emotion_rf_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error memuat model RF (emotion_rf_model.pkl): {e}")
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
        return None

# --- 3. Muat Detektor Wajah (Haar Cascade) ---
# Kita perlu ini untuk menemukan wajah di dalam frame video
@st.cache_resource
def load_face_detector():
    """Memuat file Haar Cascade untuk deteksi wajah."""
    try:
        # File ini berisi data untuk mendeteksi wajah
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        return face_cascade
    except Exception as e:
        st.error(f"Error memuat Haar Cascade: {e}")
        return None

# --- 4. Fungsi Prediksi (Untuk Gambar Tunggal) ---
def predict_emotion(image_roi, cnn_model, rf_model, labels):
    """
    Mengambil ROI (Region of Interest) wajah yang sudah di-grayscale 
    dan mengembalikan prediksi.
    """
    try:
        # 1. Resize ke 96x96 (sesuai data training)
        resized_img = cv2.resize(image_roi, (96, 96))
        
        # 2. Reshape untuk model CNN (batch_size, height, width, channels)
        img_array = np.reshape(resized_img, (1, 96, 96, 1))
        
        # 3. Normalisasi pixel (skala 0-1)
        img_array = img_array / 255.0
        
        # --- Rantai Prediksi ---
        # 4. Dapatkan fitur dari CNN
        features = cnn_model.predict(img_array)
        
        # 5. Prediksi kelas menggunakan Random Forest
        prediction_idx = rf_model.predict(features)
        
        # 6. Dapatkan probabilitas (untuk keyakinan)
        probabilities = rf_model.predict_proba(features)
        
        # 7. Dapatkan hasil
        emotion_label = str(prediction_idx[0]) # Konversi ke string untuk key JSON
        emotion_name = labels.get(emotion_label, "Tidak Dikenali")
        confidence = probabilities[0][prediction_idx[0]]
        
        return emotion_name, confidence

    except Exception as e:
        st.warning(f"Error saat prediksi: {e}")
        return "Error", 0.0

# --- 5. Kelas Video Transformer (Inti Deteksi Real-time) ---
class EmotionTransformer(VideoTransformerBase):
    def __init__(self, cnn_model, rf_model, labels, face_detector):
        self.cnn_model = cnn_model
        self.rf_model = rf_model
        self.labels = labels
        self.face_detector = face_detector

    def recv(self, frame):
        # 1. Ubah frame video ke format OpenCV (BGR)
        img = frame.to_ndarray(format="bgr24")
        
        # 2. Konversi ke Grayscale (lebih cepat untuk deteksi wajah)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 3. Deteksi wajah di frame
        # scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        faces = self.face_detector.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        # 4. Loop untuk setiap wajah yang terdeteksi
        for (x, y, w, h) in faces:
            # 5. Ambil Region of Interest (ROI) wajah saja
            face_roi = gray[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                continue
            
            # 6. Lakukan prediksi emosi pada ROI
            emotion, confidence = predict_emotion(face_roi, self.cnn_model, self.rf_model, self.labels)
            
            # 7. Tampilkan teks dan kotak di frame video
            text = f"{emotion} ({confidence:.2f})"
            
            # Tentukan warna kotak (hijau)
            color = (0, 255, 0) # BGR
            
            # Gambar kotak di sekitar wajah
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            
            # Gambar latar belakang teks
            cv2.rectangle(img, (x, y-30), (x+w, y), color, -1)
            # Tulis teks emosi
            cv2.putText(img, text, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
        # 8. Kembalikan frame yang sudah dimodifikasi (DI LUAR LOOP FOR)
        return frame.from_ndarray(img, format="bgr24")

# --- 6. Tampilan Utama (Main UI) ---
def main():
    st.title("😊 Aplikasi Deteksi Ekspresi Wajah")
    st.write("""
    Aplikasi ini dapat mendeteksi ekspresi wajah baik melalui upload gambar 
    maupun secara real-time menggunakan kamera Anda.
    """)

    # --- Load Semua Model ---
    # Gunakan st.spinner agar terlihat proses loadingnya
    with st.spinner("Memuat semua model... Harap tunggu..."):
        cnn_model = load_cnn_model()
        rf_model = load_rf_model()
        labels = load_labels()
        face_detector = load_face_detector()

    if cnn_model is None or rf_model is None or labels is None or face_detector is None:
        st.error("Gagal memuat satu atau lebih file model. Aplikasi tidak dapat berjalan.")
        st.stop()
    
    # --- Pilihan Mode Aplikasi ---
    st.sidebar.header("Pilih Mode")
    app_mode = st.sidebar.selectbox("Mode Deteksi:", 
                                    ["Upload Gambar", "Deteksi Real-time"])

    # --- MODE 1: UPLOAD GAMBAR ---
    if app_mode == "Upload Gambar":
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
                with st.spinner('Menganalisis ekspresi...'):
                    # --- Logika deteksi untuk upload ---
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                    
                    # Deteksi wajah di gambar
                    faces = face_detector.detectMultiScale(gray_img, 1.1, 4)
                    
                    if len(faces) == 0:
                        st.warning("Tidak ada wajah yang terdeteksi di gambar.")
                    else:
                        st.success(f"Terdeteksi {len(faces)} wajah. Memproses wajah pertama...")
                        # Ambil wajah pertama saja
                        (x, y, w, h) = faces[0]
                        face_roi = gray_img[y:y+h, x:x+w]
                        
                        emotion, confidence = predict_emotion(face_roi, cnn_model, rf_model, labels)
                        
                        # Gambar kotak di gambar asli (buat salinan)
                        img_with_box = img_cv.copy()
                        cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        st.image(img_with_box, channels="BGR", caption="Wajah terdeteksi")
                        
                        st.metric(label="Ekspresi", value=emotion.capitalize())
                        st.metric(label="Tingkat Keyakinan", value=f"{confidence:.2%}")
        else:
            st.info('Silakan upload gambar melalui sidebar untuk memulai deteksi.')

    # --- MODE 2: DETEKSI REAL-TIME ---
    elif app_mode == "Deteksi Real-time":
        st.sidebar.warning("Pastikan Anda memberikan izin browser untuk mengakses kamera.")
        
        st.header("Deteksi Ekspresi Real-time")
        st.write("Arahkan wajah Anda ke kamera. Model akan mencoba mendeteksi ekspresi Anda.")

        webrtc_streamer(
            key="emotion_detection",
            video_transformer_factory=lambda: EmotionTransformer(
                cnn_model=cnn_model,
                rf_model=rf_model,
                labels=labels,
                face_detector=face_detector
            ),
            rtc_configuration={  # Konfigurasi untuk deploy (opsional tapi bagus)
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={"video": True, "audio": False}
        )

if __name__ == "__main__":
    main()

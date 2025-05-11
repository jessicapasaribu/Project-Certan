import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import gdown

st.set_page_config(page_title="Certan - Deteksi Penyakit Ayam", layout="wide")

# ---------------------------
# Tabs Navigasi (Gantikan Sidebar)
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Beranda", "📸 Deteksi Gambar", "ℹ️ Tentang"])

# ---------------------------
# Fungsi Preprocessing Gambar
# ---------------------------
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------
# Label dan Informasi
# ---------------------------
class_names = [
    "Chicken_Coccidiosis",
    "Chicken_Healthy",
    "Chicken_NewCastleDisease",
    "Chicken_Salmonella"
]

class_descriptions = {
    "Chicken_Coccidiosis": "⚠️ Coccidiosis adalah penyakit parasit usus yang disebabkan oleh protozoa.",
    "Chicken_Healthy": "✅ Ayam dalam kondisi sehat.",
    "Chicken_NewCastleDisease": "⚠️ Newcastle Disease adalah infeksi virus yang sangat menular.",
    "Chicken_Salmonella": "⚠️ Salmonella adalah infeksi bakteri yang dapat menyebar melalui makanan atau air."
}

@st.cache_resource
def load_model():
    MODEL_ID = "1-FobzoF_xu7OT3shK0UeQzQOp-BjDLPX"
    MODEL_PATH = "model_state_dict.pt"
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mengunduh model..."):
            url = f"https://drive.google.com/uc?id={MODEL_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            st.success("✅ Model berhasil diunduh!")
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

def predict(image, model):
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
        return label, class_descriptions[label]

def tampilkan_hasil(image):
    st.image(image, caption="Gambar yang Diproses", width=300)
    try:
        label, _ = predict(image, model)
        st.success(f"✅ Prediksi: {label.replace('_', ' ')}")

        if label == "Chicken_Coccidiosis":
            st.markdown("""### 🦠 Coccidiosis
            Infeksi usus oleh protozoa *Eimeria*. Bisa menyebabkan kematian.
            **Gejala:** diare berdarah, lesu, penurunan berat badan.
            **Solusi:** vaksin, sanitasi ketat, pemisahan ayam terinfeksi.
            """)

        elif label == "Chicken_Salmonella":
            st.markdown("""### 🧫 Salmonella
            Infeksi bakteri dari makanan/air kotor.
            **Gejala:** diare, nafsu makan menurun, kematian anak ayam.
            **Solusi:** vaksinasi, sanitasi, isolasi ayam sakit.
            """)

        elif label == "Chicken_NewCastleDisease":
            st.markdown("""### 🦠 Newcastle Disease
            Virus mematikan yang menyerang sistem saraf dan pernapasan.
            **Gejala:** batuk, lumpuh, kematian mendadak.
            **Solusi:** vaksinasi rutin, biosekuriti ketat.
            """)

        elif label == "Chicken_Healthy":
            st.markdown("""### ✅ Ayam Sehat
            Gambar tidak menunjukkan tanda penyakit.
            **Saran:** pertahankan pola makan, kebersihan, dan vaksinasi.
            """)
    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")

# ---------------------------
# Halaman Tab 1 - Beranda
# ---------------------------
with tab1:
    st.title("🐔 Certan")
    st.markdown("""
    ### Chicken Excreta Recognition & Analysis Tool
    _"Deteksi Dini, Produksi Terjaga"_ 🧪

    ---
    Certan adalah aplikasi AI untuk mendeteksi penyakit ayam dari gambar kotoran. Menggunakan **ResNet-50**, Certan mengklasifikasikan gambar ke dalam:
    
    - 🦠 **Coccidiosis**
    - 🧫 **Salmonella**
    - 🦠 **Newcastle Disease**
    - ✅ **Sehat**

    ---
    **Manfaat:**
    - 🔍 Deteksi dini
    - 💸 Hemat biaya pengobatan
    - 📈 Tingkatkan produktivitas ternak

    ---
    """)

# ---------------------------
# Halaman Tab 2 - Deteksi
# ---------------------------
with tab2:
    st.header("📸 Deteksi Penyakit Ayam dari Gambar Kotoran")
    pilihan = st.radio("Pilih Metode Input Gambar", ["Unggah Gambar", "Ambil dari Kamera"])

    if pilihan == "Unggah Gambar":
        uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                tampilkan_hasil(image)
            except Exception as e:
                st.error(f"Gagal memuat gambar: {e}")

    elif pilihan == "Ambil dari Kamera":
        camera_image = st.camera_input("Ambil gambar langsung")
        if camera_image:
            try:
                image = Image.open(camera_image).convert("RGB")
                tampilkan_hasil(image)
            except Exception as e:
                st.error(f"Gagal membuka gambar kamera: {e}")

# ---------------------------
# Halaman Tab 3 - Tentang
# ---------------------------
with tab3:
    st.subheader("Tentang Certan 🐔")
    st.markdown("""
    **Certan** adalah aplikasi AI untuk deteksi penyakit ayam lewat gambar kotoran.

    ### 🎯 Tujuan:
    - Deteksi penyakit secara cepat & praktis
    - Bantu edukasi peternak & mahasiswa
    - Dukung kesehatan unggas nasional

    ### 🧠 Teknologi:
    - Deep Learning (ResNet-50)
    - Dataset ±4000 gambar dilatih dengan augmentasi
    - Akurasi validasi model: ±91%

    ### 👩‍💻 Pengembang:
    Kelompok 19 - D3 Teknologi Informasi  
    - Jessi Pasaribu  
    - [Nama lainnya jika ada]

    📫 Kontak: [email@example.com]

    ⚠️ *Disclaimer: ini bukan alat diagnosis medis resmi. Konsultasikan ke dokter hewan untuk diagnosis akurat.*
    """)

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import gdown

# ---------------------------
# Konfigurasi Halaman
# ---------------------------
st.set_page_config(page_title="Certan - Deteksi Penyakit Ayam", layout="centered")

# ---------------------------
# Sidebar Navigasi
# ---------------------------
st.sidebar.title("Navigasi")
mode = st.sidebar.radio("Pilih Mode", ["🏠 Beranda", "📸 Deteksi Gambar", "ℹ️ Tentang"])

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
    "Chicken_Coccidiosis": "Infeksi protozoa di usus ayam.",
    "Chicken_Healthy": "Ayam dalam kondisi sehat.",
    "Chicken_NewCastleDisease": "Virus yang menyerang saraf dan pernapasan.",
    "Chicken_Salmonella": "Infeksi bakteri dari makanan/air kotor."
}

# ---------------------------
# Load Model
# ---------------------------
MODEL_ID = "1-FobzoF_xu7OT3shK0UeQzQOp-BjDLPX"
MODEL_PATH = "model_state_dict.pt"

@st.cache_resource
def load_model():
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

# ---------------------------
# Fungsi Prediksi
# ---------------------------
def predict(image, model):
    img_tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = torch.max(outputs, 1)
        label = class_names[predicted.item()]
        return label, class_descriptions[label]

# ---------------------------
# Fungsi Hasil Deteksi
# ---------------------------
def tampilkan_hasil(image):
    st.image(image, caption="📷 Gambar yang Diproses", width=300)

    try:
        label, _ = predict(image, model)
        st.success(f"✅ Prediksi: {label.replace('_', ' ')}")

        if label == "Chicken_Coccidiosis":
            st.markdown("""
                ## 🦠 Coccidiosis

                Infeksi usus serius akibat protozoa *Eimeria*.

                ---

                ### 🧩 Gejala:
                - 💩 Diare berdarah
                - 💤 Lesu
                - ⚖️ Berat badan turun
                - 🪶 Bulu kusam

                ### 🛡️ Solusi:
                - Obat anticoccidial
                - Kebersihan kandang
                - Hindari kepadatan tinggi
            """)

        elif label == "Chicken_Salmonella":
            st.markdown("""
                ## 🧫 Salmonella

                Infeksi bakteri dari air/pakan terkontaminasi.

                ---

                ### 🧩 Gejala:
                - 💩 Diare encer
                - 🐣 Kematian anak ayam
                - 🥚 Penurunan produksi telur

                ### 🛡️ Solusi:
                - Vaksinasi
                - Sanitasi kandang
                - Isolasi ayam sakit
            """)

        elif label == "Chicken_NewCastleDisease":
            st.markdown("""
                ## 🦠 Newcastle Disease

                Virus menular menyerang pernapasan, saraf & pencernaan.

                ---

                ### 🧩 Gejala:
                - 😮‍💨 Batuk, sesak napas
                - 🌀 Leher terpelintir
                - ☠️ Kematian massal

                ### 🛡️ Solusi:
                - Vaksinasi berkala
                - Karantina ketat
                - Pemusnahan (bila parah)
            """)

        elif label == "Chicken_Healthy":
            st.markdown("""
                ## ✅ Ayam Sehat

                Tidak ditemukan gejala penyakit utama.

                ---

                ### 🧩 Rekomendasi:
                - 🍽️ Pola makan baik
                - 🧼 Kandang bersih
                - 💉 Lanjutkan vaksinasi
            """)

    except Exception as e:
        st.error(f"❌ Gagal melakukan prediksi: {e}")

# ---------------------------
# Halaman Beranda
# ---------------------------
if mode == "🏠 Beranda":
    st.title("🐔 Certan")
    st.markdown("""
    ### Chicken Excreta Recognition & Analysis Tool  
    _"Deteksi Dini, Produksi Terjaga"_ 🧪

    ---

    Selamat datang di **Certan**, aplikasi AI untuk mendeteksi penyakit ayam dari gambar kotoran. 🚀

    ### Deteksi:
    - 🦠 Coccidiosis
    - 🧫 Salmonella
    - 🦠 Newcastle Disease
    - ✅ Sehat

    ### Kenapa Gunakan Certan?
    - 🔍 Deteksi cepat = penanganan dini
    - 💸 Hemat biaya pengobatan
    - 📈 Produktivitas meningkat

    ### Fitur:
    - 📸 Upload atau kamera langsung
    - 📚 Info penyakit ayam
    - ⏱️ Hasil instan
    """)

# ---------------------------
# Halaman Deteksi
# ---------------------------
elif mode == "📸 Deteksi Gambar":
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
                st.stop()

    elif pilihan == "Ambil dari Kamera":
        camera_image = st.camera_input("Ambil gambar langsung")
        if camera_image:
            try:
                image = Image.open(camera_image).convert("RGB")
                tampilkan_hasil(image)
            except Exception as e:
                st.error(f"Gagal membuka gambar kamera: {e}")
                st.stop()

# ---------------------------
# Halaman Tentang
# ---------------------------
elif mode == "ℹ️ Tentang":
    st.subheader("Tentang Certan 🐔")
    st.markdown("""
    **Certan** (*Chicken Excreta Recognition & Analysis Tool*) adalah aplikasi AI untuk deteksi dini penyakit ayam berdasarkan gambar kotoran.

    ---

    ### 🎯 Tujuan
    - Deteksi awal penyakit unggas
    - Edukasi peternak & mahasiswa
    - Meningkatkan kesehatan ternak

    ### 🧠 Teknologi
    - Model: ResNet-50
    - Klasifikasi: 4 kelas
    - Input: Gambar via upload/kamera

    ### 📊 Dataset
    - ±4000 gambar dilatih
    - Augmentasi: rotasi, flip, brightness
    - Akurasi validasi: ±91%

    ### 👨‍💻 Pengembang
    Proyek oleh:
    **Kelompok 19 - D3 Teknologi Informasi**

    - Jessi Pasaribu
    - [Anggota lainnya]

    📫 Kontak: [email@example.com]

    ### ⚠️ Disclaimer
    Certan bukan alat diagnosis resmi. Untuk diagnosis pasti, konsultasikan dengan dokter hewan.
    """)

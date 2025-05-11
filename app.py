import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io
import os
import urllib.request

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
# Kelas Label dan Informasi
# ---------------------------
class_names = [
    "Chicken_Coccidiosis",
    "Chicken_Healthy",
    "Chicken_NewCastleDisease",
    "Chicken_Salmonella"
]

class_descriptions = {
    "Chicken_Coccidiosis": "⚠️ Coccidiosis adalah penyakit parasit usus yang disebabkan oleh protozoa. Biasanya ditandai dengan diare berdarah dan penurunan berat badan.",
    "Chicken_Healthy": "✅ Ayam dalam kondisi sehat. Tidak ditemukan gejala penyakit dalam gambar kotoran yang diberikan.",
    "Chicken_NewCastleDisease": "⚠️ Newcastle Disease adalah infeksi virus yang sangat menular. Dapat menyebabkan gejala pernapasan, pencernaan, dan saraf.",
    "Chicken_Salmonella": "⚠️ Salmonella adalah infeksi bakteri yang dapat menyebar melalui makanan atau air yang terkontaminasi, menyebabkan diare dan masalah pencernaan."
}

risk = {
    "Chicken_Healthy": "🟢 Sangat Rendah",
    "Chicken_Coccidiosis": "🔴 Tinggi",
    "Chicken_Salmonella": "🟠 Sedang",
    "Chicken_NewCastleDisease": "🔴 Sangat Tinggi"
}

# ---------------------------
# Load Model (dengan state_dict)
# ---------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1-FobzoF_xu7OT3shK0UeQzQOp-BjDLPX"
MODEL_PATH = "Model-Certan-true-state.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mengunduh model dari Google Drive..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            st.success("✅ Model berhasil diunduh!")

    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    return model

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
# Halaman Beranda
# ---------------------------
if mode == "🏠 Beranda":
    st.title("Certan 🐔")
    st.markdown("""
    Selamat datang di **Certan**, aplikasi pintar untuk deteksi dini penyakit ayam berbasis gambar kotoran! 🧪🐓

    Dengan teknologi **Deep Learning (ResNet-50)**, Certan membantu peternak dan pemerhati unggas mengenali 4 kondisi:

    - ⚠️ *Coccidiosis*
    - ⚠️ *Salmonella*
    - ⚠️ *Newcastle Disease*
    - ✅ *Kondisi Sehat*

    **Kenapa Penting?**
    - Deteksi dini = pencegahan penyebaran
    - Hemat biaya pengobatan
    - Meningkatkan produktivitas ternak

    **Fitur Aplikasi:**
    - Deteksi otomatis dari gambar 📸
    - Info lengkap tentang penyakit 🧬
    - Input dari kamera langsung atau upload gambar 🖼️

    **Gunakan aplikasi ini untuk:**
    - Skrining cepat sebelum konsultasi dokter hewan
    - Edukasi peternak atau mahasiswa peternakan
    - Penelitian di bidang kesehatan hewan
    """)

# ---------------------------
# Halaman Deteksi Gambar
# ---------------------------
elif mode == "📸 Deteksi Gambar":
    st.header("📸 Deteksi Penyakit Ayam dari Gambar Kotoran")
    pilihan = st.radio("Pilih Metode Input Gambar", ["Unggah Gambar", "Ambil dari Kamera"])

    model = load_model()

    if pilihan == "Unggah Gambar":
        uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Gambar yang Diunggah", width=300)
            label, info = predict(image, model)
            st.success(f"✅ Prediksi: {label}")
            st.info(info)

            st.markdown(f"""
            ### 🔬 Ringkasan Deteksi
            - **Jenis**: {label.replace('_', ' ')}
            - **Risiko Penularan**: {risk.get(label, 'Tidak Diketahui')}
            - **Saran**: {'Segera isolasi ayam dan konsultasikan ke dokter hewan.' if label != 'Chicken_Healthy' else 'Pertahankan kebersihan dan pakan yang baik.'}
            """)

            st.markdown("""
            ---
            ### 📌 Fakta Cepat
            - Coccidiosis bisa membunuh ayam hanya dalam 2-3 hari bila tidak diobati.
            - Newcastle Disease menyebar melalui udara dan sangat menular.
            - Salmonella dapat menular ke manusia jika tidak ditangani dengan baik.

            📖 **Sumber**: Direktorat Jenderal Peternakan dan Kesehatan Hewan, 2022
            """)

    elif pilihan == "Ambil dari Kamera":
        camera_image = st.camera_input("Ambil gambar langsung")
        if camera_image:
            image = Image.open(camera_image).convert("RGB")
            st.image(image, caption="Gambar dari Kamera", width=300)
            label, info = predict(image, model)
            st.success(f"✅ Prediksi: {label}")
            st.info(info)

            st.markdown(f"""
            ### 🔬 Ringkasan Deteksi
            - **Jenis**: {label.replace('_', ' ')}
            - **Risiko Penularan**: {risk.get(label, 'Tidak Diketahui')}
            - **Saran**: {'Segera isolasi ayam dan konsultasikan ke dokter hewan.' if label != 'Chicken_Healthy' else 'Pertahankan kebersihan dan pakan yang baik.'}
            """)

            st.markdown("""
            ---
            ### 📌 Fakta Cepat
            - Coccidiosis bisa membunuh ayam hanya dalam 2-3 hari bila tidak diobati.
            - Newcastle Disease menyebar melalui udara dan sangat menular.
            - Salmonella dapat menular ke manusia jika tidak ditangani dengan baik.

            📖 **Sumber**: Direktorat Jenderal Peternakan dan Kesehatan Hewan, 2022
            """)

# ---------------------------
# Halaman Tentang
# ---------------------------
elif mode == "ℹ️ Tentang":
    st.subheader("Tentang Aplikasi")
    st.markdown("""
    **Certan** adalah aplikasi deteksi penyakit ayam berbasis gambar kotoran menggunakan **Deep Learning** (ResNet-50) dan **Streamlit**.

    **Fitur:**
    - Deteksi otomatis penyakit: Coccidiosis, Salmonella, Newcastle, Healthy
    - Input gambar via upload atau kamera
    - Real-time prediction menggunakan model terlatih

    **Pengembang:** Kelompok 19 - D3 Teknologi Informasi

    **Catatan:** Aplikasi ini bersifat edukatif dan bukan pengganti diagnosis dokter hewan profesional.
    """)

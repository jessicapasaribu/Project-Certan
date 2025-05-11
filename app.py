import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np  # WAJIB: untuk konversi gambar
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
MODEL_ID = "1-FobzoF_xu7OT3shK0UeQzQOp-BjDLPX"
MODEL_PATH = "model_state_dict.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("📥 Mengunduh model dari Google Drive..."):
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
# Fungsi Tampilkan Hasil Deteksi (diperbarui)
# ---------------------------
def tampilkan_hasil(image):
    st.image(image, caption="Gambar yang Diproses", width=300)
    st.write(f"Mode gambar: {image.mode}, Ukuran: {image.size}")

    try:
        label, _ = predict(image, model)

        st.success(f"✅ Prediksi: {label.replace('_', ' ')}")
        
        # Informasi mendalam berdasarkan label
        if label == "Chicken_Coccidiosis":
            st.markdown("""
            ### 🦠 Coccidiosis
            Coccidiosis adalah infeksi usus serius yang disebabkan oleh protozoa *Eimeria*. Penyakit ini umum terjadi pada ayam muda dan bisa menyebabkan kematian jika tidak segera ditangani.

            **Gejala Umum:**
            - Diare berdarah
            - Lesu dan kehilangan nafsu makan
            - Penurunan berat badan
            - Bulu kusam

            **Pencegahan & Pengobatan:**
            - Vaksinasi dan pemberian anticoccidial dalam pakan
            - Jaga kebersihan kandang dan sanitasi air
            - Hindari kepadatan populasi yang tinggi

            📌 *Segera isolasi ayam yang terinfeksi dan konsultasikan dengan dokter hewan.*
            """)

        elif label == "Chicken_Salmonella":
            st.markdown("""
            ### 🧫 Salmonella
            Salmonellosis adalah infeksi bakteri dari genus *Salmonella*, umumnya menyebar melalui makanan, air, atau peralatan yang terkontaminasi.

            **Gejala Umum:**
            - Diare dan kotoran berair
            - Nafsu makan menurun
            - Penurunan produksi telur
            - Kematian mendadak pada anak ayam

            **Pencegahan & Pengobatan:**
            - Gunakan vaksin dan biosekuriti yang ketat
            - Pisahkan ayam yang terinfeksi
            - Bersihkan kandang secara rutin

            ⚠️ *Beberapa jenis Salmonella bisa menular ke manusia (zoonosis), penting untuk penanganan higienis.*
            """)

        elif label == "Chicken_NewCastleDisease":
            st.markdown("""
            ### 🦠 Newcastle Disease
            Newcastle Disease (ND) adalah penyakit virus yang sangat menular dan mematikan, menyerang saluran pernapasan, sistem saraf, dan pencernaan ayam.

            **Gejala Umum:**
            - Bersin, batuk, dan sesak napas
            - Leher melintir atau lumpuh (gejala neurologis)
            - Penurunan produksi telur drastis
            - Kematian mendadak

            **Pencegahan & Pengobatan:**
            - Vaksinasi ND secara berkala
            - Pengawasan lalu lintas unggas
            - Isolasi dan pemusnahan ayam terinfeksi (jika perlu)

            🔥 *ND adalah penyakit yang wajib dilaporkan di banyak negara karena tingkat penyebarannya yang ekstrem.*
            """)

        elif label == "Chicken_Healthy":
            st.markdown("""
            ### ✅ Ayam Sehat
            Gambar kotoran ayam menunjukkan tidak adanya indikasi penyakit utama seperti Coccidiosis, Salmonella, atau Newcastle Disease.

            **Rekomendasi:**
            - Pertahankan pola makan seimbang
            - Jaga kebersihan kandang setiap hari
            - Lakukan vaksinasi dan biosekuriti sesuai jadwal

            🎉 *Tidak ditemukan kelainan – kondisi normal.*
            """)

    except Exception as e:
        st.error(f"Gagal melakukan prediksi: {e}")

# ---------------------------
# Halaman Beranda (Rebranding)
# ---------------------------
if mode == "🏠 Beranda":
    st.title("🐔 Certan")
    st.markdown("""
    ### Chicken Excreta Recognition & Analysis Tool
    _"Deteksi Dini, Produksi Terjaga"_ 🧪

    ---
    
    Selamat datang di **Certan**, aplikasi cerdas berbasis kecerdasan buatan (AI) untuk mendeteksi penyakit ayam dari gambar kotoran. 🚀

    Dengan teknologi **Deep Learning (ResNet-50)**, Certan mampu mengklasifikasikan gambar ke dalam 4 kondisi:
    
    - 🦠 **Coccidiosis**
    - 🧫 **Salmonella**
    - 🦠 **Newcastle Disease**
    - ✅ **Sehat**

    ---
    ### Kenapa Penting?
    - 🔍 Deteksi dini = Pencegahan cepat
    - 💸 Menghemat biaya pengobatan
    - 📈 Meningkatkan produktivitas ternak

    ---
    ### Fitur Aplikasi:
    - 📸 Deteksi otomatis dari gambar
    - 📚 Info lengkap tentang penyakit
    - 📷 Input gambar dari kamera atau galeri

    ---
    **Gunakan Certan untuk:**
    - Skrining cepat sebelum konsultasi dokter hewan
    - Edukasi peternak dan mahasiswa
    - Riset dan pengembangan peternakan digital

    ---
    """)
# ---------------------------
# Halaman Deteksi Gambar
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
    st.subheader("Tentang Aplikasi")
    st.markdown("""
**Certan** adalah aplikasi deteksi penyakit ayam berbasis gambar kotoran menggunakan **Deep Learning** (ResNet-50) dan **Streamlit**.

---

### 🧠 Tentang Model
- **Arsitektur**: ResNet-50
- **Dataset**: Kumpulan gambar kotoran ayam dari berbagai sumber daring & laboratorium lokal
- **Jumlah Data**: ±4.000 gambar (terdistribusi seimbang di 4 kelas)
- **Augmentasi**: Rotasi, flipping horizontal/vertikal, brightness shift
- **Optimasi**: Adam optimizer, learning rate 1e-4, batch size 32
- **Epoch Training**: 25
- **Akurasi Validasi**: ~91.3%

---

### 📂 Sumber Dataset
Dataset dikumpulkan dan dibersihkan dari:
- Kaggle & publikasi terbuka (Coccidiosis, Salmonella)
- Hasil dokumentasi lab peternakan lokal
- Dataset tambahan melalui crowdsourcing dari peternak mitra

---

### ⚠️ Catatan
Aplikasi ini bersifat **edukatif** dan **bukan** pengganti diagnosis dokter hewan profesional.

---

### 👨‍💻 Pengembang
**Kelompok 19 - D3 Teknologi Informasi**
- Aplikasi dibangun sebagai bagian dari proyek akhir semester.

📫 Untuk pertanyaan atau kolaborasi: [email@example.com]
""")

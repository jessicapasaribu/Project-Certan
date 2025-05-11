import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import urllib.request
import os

# --- Download model dari Google Drive jika belum ada ---
MODEL_URL = "https://drive.google.com/uc?export=download&id=1CX-jCa3Tz9LimSq729DLKVS97RwEXPwv"
MODEL_PATH = "Model-Certan-true.pt"

if not os.path.exists(MODEL_PATH):
    with st.spinner("📥 Mengunduh model..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("✅ Model berhasil diunduh!")

# --- Load Model ---
model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
model.eval()

# --- Kelas ---
classes = ['Chicken_Coccidiosis', 'Chicken_Healthy', 'Chicken_NewCastleDisease', 'Chicken_Salmonella']

# --- Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Deskripsi Penyakit ---
penjelasan = {
    "Chicken_Coccidiosis": "🔴 Penyakit parasit yang menyerang usus ayam. Gejala: diare berdarah, lesu, dan penurunan berat badan.",
    "Chicken_Healthy": "🟢 Tidak terdeteksi penyakit. Ayam kemungkinan dalam kondisi sehat.",
    "Chicken_NewCastleDisease": "🟡 Penyakit virus yang sangat menular. Gejala: sesak napas, leher terpuntir, dan kematian mendadak.",
    "Chicken_Salmonella": "🟠 Infeksi bakteri yang menyebabkan diare, kehilangan nafsu makan, dan bisa menular ke manusia.",
}

# --- Judul ---
st.title("🐔 Deteksi Penyakit Ayam dari Gambar Kotoran")
st.markdown("""
Aplikasi ini mendeteksi penyakit pada ayam berdasarkan **gambar kotoran** menggunakan model deep learning ResNet-50.

Silakan unggah atau ambil gambar untuk mendeteksi:
- Coccidiosis
- Salmonella
- Newcastle Disease
- Kondisi Sehat
""")

# --- Input Gambar ---
option = st.radio("Pilih metode input:", ["📸 Ambil Gambar", "🖼️ Upload Gambar"])

image = None
if option == "📸 Ambil Gambar":
    image = st.camera_input("Ambil gambar kotoran ayam")
else:
    image = st.file_uploader("Unggah gambar kotoran ayam", type=["jpg", "jpeg", "png"])

# --- Prediksi ---
if image:
    img = Image.open(image).convert('RGB')
    st.image(img, caption="Gambar yang dimasukkan", use_column_width=True)

    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    st.success(f"📌 **Hasil Prediksi: {label}**")
    st.info(penjelasan.get(label, "Tidak ada penjelasan tersedia."))

# --- Footer ---
st.markdown("""
---
🧪 Model: ResNet-50, dilatih untuk klasifikasi kotoran ayam  
📦 Framework: PyTorch + Streamlit  
👨‍💻 Oleh: [Nama Kamu]
""")

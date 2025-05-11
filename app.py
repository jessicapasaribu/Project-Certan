import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load model (disimpan dengan torch.save(model, ...))
@st.cache_resource
def load_model():
    model = torch.load("Model-Certan-true.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# Label prediksi sesuai urutan folder training
label_dict = {
    0: "Coccidiosis",
    1: "Healthy",
    2: "Newcastle Disease (ND)",
    3: "Salmonella"
}



# Transformasi sesuai pelatihan
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
st.markdown("""
### 🐔 Tentang Aplikasi
Aplikasi ini menggunakan model deep learning berbasis ResNet-50 untuk mendeteksi **penyakit ayam** melalui gambar kotoran.  
Cukup upload atau ambil gambar kotoran ayam, dan sistem akan memprediksi kemungkinan penyakit yang diderita ayam berdasarkan citra.

Model mengenali 4 kelas:
- **Coccidiosis**
- **Salmonella**
- **Newcastle Disease (ND)**
- **Healthy**

🧠 Teknologi yang digunakan: PyTorch, ResNet-50, Streamlit
""")


# Fungsi prediksi
def predict(image):
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 224, 224]
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
    return label_dict.get(pred_idx, "Tidak diketahui")

# Tampilan Streamlit
st.title("Prediksi Penyakit Ayam dari Gambar Kotoran")
st.write("Gunakan foto kotoran ayam untuk memprediksi kemungkinan penyakit.")

input_method = st.radio("Pilih metode input:", ["Upload Gambar", "Ambil dari Kamera"])

if input_method == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diupload", use_column_width=True)
        pred = predict(image)
        st.success(f"Hasil Prediksi: **{pred}**")

elif input_method == "Ambil dari Kamera":
    camera_image = st.camera_input("Ambil gambar dari kamera")
    if camera_image is not None:
        image = Image.open(camera_image)
        st.image(image, caption="Gambar dari kamera", use_column_width=True)
        pred = predict(image)
        st.success(f"Hasil Prediksi: **{pred}**")

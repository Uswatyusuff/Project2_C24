import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import os
import gdown
import cv2
from ultralytics import YOLO

from csrnet_model import CSRNet

# ---------- Constants ----------
CSRNET_MODEL_PATH = "csrnet_model.pth"
CSRNET_MODEL_URL = "https://drive.google.com/uc?id=1-698yvi-ZwsPrnRlm6EKE2TXZC_f7QZ7"

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# ---------- CSS ----------
st.set_page_config(page_title="PopulusAI - Crowd Counting", layout="wide")

# ---------- DARK MODE CSS ----------
st.markdown("""
<style>
body {
    background-color: #0f1117;
    color: #e0e0e0;
    font-family: 'Segoe UI', sans-serif;
}
h1 {
    font-size: 3rem !important;
    color: #ffffff;
    text-align: center;
    margin-top: 0;
}
.description-box {
    background-color: #1f2633;
    border-left: 6px solid #4fa8f6;
    padding: 18px;
    margin-bottom: 20px;
    border-radius: 10px;
    color: #dcdcdc;
}
.result-box {
    background-color: #1c1e24;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 0 10px rgba(255,255,255,0.05);
    text-align: center;
    margin-top: 20px;
}
.stTabs [data-baseweb="tab"] {
    font-weight: bold;
    background-color: #1c1e24;
    border-radius: 10px 10px 0 0;
    padding: 12px 20px;
    color: #a9b8ce;
}
.stTabs [aria-selected="true"] {
    background-color: #4fa8f6 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- Image Transform ----------
import torchvision.transforms as transforms
IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
])

# ---------- Downloader ----------
def download_model(model_name):
    if model_name == "CSRNet" and not os.path.exists(CSRNET_MODEL_PATH):
        with st.spinner("🔽 Downloading CSRNet model..."):
            gdown.download(CSRNET_MODEL_URL, CSRNET_MODEL_PATH, quiet=False)




@st.cache_resource
def load_csrnet_model():
    download_model("CSRNet")
    model = CSRNet()
    checkpoint = torch.load(CSRNET_MODEL_PATH, map_location=torch.device("cpu"))

    # ✅ Extract only the state_dict from the checkpoint
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


@st.cache_resource
def load_yolo_model():
    return YOLO('yolov8l.pt')

# ---------- Predict & Visualize ----------
def predict_and_visualize(image, model):
    img_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)

    count = float(output.sum().item())
    density_map = output.squeeze().cpu().numpy()

    fig, ax = plt.subplots()
    ax.imshow(density_map, cmap='jet')
    ax.axis('off')

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    return count, fig, buf

def predict_with_yolo(image):
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    model = load_yolo_model()
    results = model(image_bgr)
    annotated = results[0].plot()
    count = len(results[0].boxes)
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    return count, annotated_rgb

# ---------- Main App ----------
def main():
    # ---------- TITLE + INTRO ----------
    st.title("PopulusAI")

    st.markdown("""
    <p style="text-align:center; font-size: 18px; color:#cccccc;">
        A smart crowd counting tool using AI. Choose a model to estimate the number of people in an image 📸.<br>
        Powered by <strong>CSRNet</strong>,and <strong>YOLO</strong>.
    </p>
    """, unsafe_allow_html=True)


    tab1, tab2 = st.tabs(["Outdoor Crowds", "Indoor Spaces"])

    with tab1:
        st.markdown("""
        <div class="description-box">
            <h4>🏙️ Outdoor Crowd Estimation</h4>
            Ideal for wide, open spaces like streets, festivals, parks, or public events.<br><br>
            <strong>Powered by CSRNet</strong> — a deep learning model that creates heatmaps to estimate the number of people, especially in dense crowds.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("📤 Upload Image for Outdoor Analysis (CSRNet)", type=["png", "jpg", "jpeg"],
                                         key="CSRNet")
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Processing with CSRNet..."):
                model = load_csrnet_model()
                count, fig, buf = predict_and_visualize(image, model)
                st.markdown(f"<div class='result-box'><h3>🧮 Estimated Count: {count:.2f}</h3></div>",
                            unsafe_allow_html=True)
                st.pyplot(fig)
                st.download_button("📥 Download Density Map", data=buf, file_name="density_map.png", mime="image/png")

    with tab2:
        st.markdown("""
        <div class="description-box">
            <h4>🏢 Indoor Crowd Detection</h4>
            Best suited for enclosed environments like classrooms, hallways, or lobbies.<br><br>
            <strong>Powered by YOLO</strong> — a real-time object detection model that draws bounding boxes around individuals and counts them.
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("📤 Upload Image for Indoor Detection (YOLO)", type=["png", "jpg", "jpeg"],
                                         key="YOLO")
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner("Processing with YOLOv8..."):
                count, annotated_image = predict_with_yolo(image)
                st.image(annotated_image, caption="YOLO Prediction", use_container_width=True)
                st.markdown(f"<div class='result-box'><h3>🧮 Detected Count: {count}</h3></div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

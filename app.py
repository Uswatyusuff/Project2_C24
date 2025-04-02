import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from model import CSRNet
import torchvision.transforms as transforms
import io
import os
import gdown  # Make sure to install this: pip install gdown
import torch
# ---------- Streamlit Config ----------
st.set_page_config(page_title="Crowd Counter", layout="wide")

# ---------- Custom CSS Styling ----------
CUSTOM_CSS = """
<style>
.css-18e3th9 {background-color: #f0f2f6;}
.st-bx {background-color: #ffffff; border-radius: 10px; padding: 10px;}
.stButton>button {background-color: #4CAF50; color: white; border-radius: 8px;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------- Constants ----------
MODEL_PATH = "streamlit_model_initial_test.pth"
MODEL_URL = "https://drive.google.com/uc?id=1fnOrjFZnYdEmRCbPz_piTj3dqJRqoV_L"

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
])

# ---------- Model Downloader ----------
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ---------- Model Loading ----------
@st.cache_resource
def load_model():
    """
    Load and return the CSRNet model.
    Downloads model if not already available.
    """
    download_model()
    model = CSRNet()
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model

# ---------- Prediction Logic ----------
def predict_and_visualize(image: Image.Image, model: torch.nn.Module):
    """
    Predict crowd density and count using the model.
    Returns: predicted count, matplotlib figure of the density map.
    """
    img_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        predicted_count = float(output.sum().item())
        density_map = output.squeeze().cpu().numpy()

    # Create density map plot
    fig, ax = plt.subplots()
    ax.imshow(density_map, cmap='jet')
    ax.axis('off')

    # Save figure to BytesIO buffer
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', pad_inches=0)
    buf.seek(0)

    return predicted_count, fig, buf

# ---------- Main App ----------
def main():
    st.title("📊 Crowd Counting with CSRNet")

    # Layout Columns
    col_sidebar, col_main = st.columns([1, 2])

    # Sidebar - Results Panel
    with col_sidebar:
        st.subheader("🔎 Results Panel")
        st.markdown("Upload an image to see the predicted count and density map.")
        result_count = st.empty()
        result_image = st.empty()
        download_button = st.empty()

    # Main - Upload & Prediction
    with col_main:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Drag & Drop or Browse", type=["png", "jpg", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_column_width=True)

            with st.spinner('Processing...'):
                model = load_model()
                predicted_count, density_fig, fig_buffer = predict_and_visualize(image, model)

            # Show results
            result_count.markdown(f"### 🧮 Predicted Count: `{predicted_count:.2f}`")
            result_image.pyplot(density_fig)

            # Download Button
            download_button.download_button(
                label="📥 Download Density Map",
                data=fig_buffer,
                file_name="density_map.png",
                mime="image/png"
            )

# ---------- Run App ----------
if __name__ == "__main__":
    main()

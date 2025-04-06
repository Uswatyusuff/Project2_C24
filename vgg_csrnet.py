import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from csrnet_model import CSRNet  # Import CSRNet
from vgg_model import VGG, vgg19  # Import DM-Count model
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
import io
import os
import gdown  # Make sure to install this: pip install gdown

# ---------- Constants ----------
CSRNET_MODEL_PATH = "streamlit_model_initial_test.pth"
VGG_MODEL_PATH = "vgg19-dcbb9e9d.pth"

CSRNET_MODEL_URL = "https://drive.google.com/uc?id=1fnOrjFZnYdEmRCbPz_piTj3dqJRqoV_L"
VGG_MODEL_URL = "https://drive.google.com/uc?id=1DF-Cd_1NJ0jwbf99F1j6Xx0D9_mRJ5JE"


model_urls = {
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}

NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

IMAGE_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=NORMALIZATION_MEAN, std=NORMALIZATION_STD)
])


# ---------- Model Downloader ----------
def download_model(model_name):
    if model_name == "CSRNet" and not os.path.exists(CSRNET_MODEL_PATH):
        with st.spinner("Downloading CSRNet model..."):
            gdown.download(CSRNET_MODEL_URL, CSRNET_MODEL_PATH, quiet=False)

    if model_name == "VGG" and not os.path.exists(VGG_MODEL_PATH):
        with st.spinner("Downloading VGG model..."):
            gdown.download(VGG_MODEL_URL, VGG_MODEL_PATH, quiet=False)


# ---------- Model Loading ----------
@st.cache_resource
def load_csrnet_model():
    download_model("CSRNet")
    model = CSRNet()
    checkpoint = torch.load(CSRNET_MODEL_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.eval()
    return model


@st.cache_resource
def load_dmcount_model():
    download_model("DM-Count")  # Ensure model is downloaded

    model = vgg19()  # Pass 'VGG16' as argument
    #checkpoint = torch.load(VGG_MODEL_PATH, map_location=torch.device('cpu'))

    model.load_state_dict(model_zoo.load_url(model_urls['vgg19']), strict=False)
    model.eval()

    return model


# ---------- Prediction Logic ----------
def predict_and_visualize(image: Image.Image, model: torch.nn.Module, model_name: str):
    """
    Predicts crowd count and density map for a given model.
    """
    img_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        if model_name == "CSRNet":
            output = model(img_tensor)
        elif model_name == "VGG":
            output = model(img_tensor)[0]  # DM-Count returns tuple, take the first output

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
    st.title("PopulusAI - Smart Crowd Counting Tool 🌍")

    # **Model Selection**
    model_option = st.selectbox("🔍 Choose a Model:", ["CSRNet", "VGG"])

    # Upload Section
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)

        with st.spinner(f'Loading {model_option} model...'):
            if model_option == "CSRNet":
                model = load_csrnet_model()
            else:
                model = load_dmcount_model()

        with st.spinner('Processing...🌀'):
            predicted_count, density_fig, fig_buffer = predict_and_visualize(image, model, model_option)

        # Show results
        st.subheader(f"🧮 Predicted Count: `{predicted_count:.2f}`")
        st.pyplot(density_fig)

        # Download Button
        st.download_button(
            label="📥 Download Density Map",
            data=fig_buffer,
            file_name="density_map.png",
            mime="image/png"
        )


# ---------- Run App ----------
if __name__ == "__main__":
    main()
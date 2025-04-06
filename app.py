import streamlit as st
import torch
from PIL import Image
import matplotlib.pyplot as plt
from csrnet_model import CSRNet
import torchvision.transforms as transforms
import io
import os
import gdown  # Make sure to install this: pip install gdown

# ---------- Custom CSS Styling with Animations ----------
CUSTOM_CSS = """
<style>
/* Background color with a smooth transition */
body {
    background-color: #D0E9FF;
    font-family: 'Arial', sans-serif;
    color: #333333;
    transition: background-color 0.5s ease;
}

/* Title styling with emoji and animation */
.stTitle {
    font-size: 40px;
    font-weight: bold;
    text-align: center;
    color: #333333;
    margin-bottom: 20px;
    animation: fadeIn 2s ease-in-out;
}

/* Main container for layout */
.stContainer {
    background: #FFFFFF;
    padding: 30px;
    border-radius: 10px;
    box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    animation: slideUp 1s ease-out;
}

/* Text under the title */
.stText {
    font-size: 18px;
    color: #333333;
    text-align: center;
    margin-bottom: 20px;
}

/* Upload Section */
.stUpload {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
}

/* Button styles */
.stButton>button {
    background-color: #007BFF;
    color: white;
    font-size: 16px;
    padding: 10px 15px;
    border-radius: 5px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #0056b3;
}

/* Results Panel with emoji */
.stResult {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.1);
    text-align: center;
    margin-top: 20px;
    animation: zoomIn 1s ease-out;
}

/* Density Map */
.stImage {
    border-radius: 10px;
    box-shadow: 0px 2px 8px rgba(0, 0, 0, 0.2);
}

/* Fun fact pop-up animation */
@keyframes slideUp {
    0% { opacity: 0; transform: translateY(50px); }
    100% { opacity: 1; transform: translateY(0); }
}

@keyframes fadeIn {
    0% { opacity: 0; }
    100% { opacity: 1; }
}

@keyframes zoomIn {
    0% { opacity: 0; transform: scale(0.9); }
    100% { opacity: 1; transform: scale(1); }
}

</style>
"""

# ---------- Streamlit Configuration ----------
st.set_page_config(page_title="PopulusAI - Crowd Counting", layout="wide")
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

# ---------- Fun Fact About Crowd Counting ----------
def crowd_counting_fact():
    facts = [
        "Crowd counting has applications in crowd safety, event management, and urban planning.",
        "Did you know? It's estimated that the world population will reach 9 billion by 2050, making crowd counting even more important!",
        "AI-based crowd counting systems are much more accurate than traditional methods like human observation or counting sensors.",
        "Interesting fact: The first real-time crowd counting system was introduced for stadiums and public events in the 1990s.",
    ]
    return facts[st.session_state.get("fact_index", 0)]

# ---------- Main App ----------
def main():
    st.title("PopulusAI - Smart Crowd Counting Tool 🌍")  # Changed Heading Here

    # **Updated Text Below Title - Clear and Readable with Emoji**
    st.markdown('<p class="stText">PopulusAI uses state-of-the-art deep learning 🤖 to count crowds in images accurately. Upload an image 📷 to get started! </p>', unsafe_allow_html=True)

    # Layout Columns
    col_main, col_sidebar = st.columns([2, 1])

    # Main Section - Upload & Prediction
    with col_main:
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)

        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader("Drag & Drop or Browse", type=["png", "jpg", "jpeg"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)

            with st.spinner('Processing...🌀'):
                model = load_model()
                predicted_count, density_fig, fig_buffer = predict_and_visualize(image, model)

            # Show results
            st.markdown('<div class="stResult">', unsafe_allow_html=True)
            st.subheader(f"🧮 Predicted Count: `{predicted_count:.2f}`")
            st.pyplot(density_fig)
            st.markdown("</div>", unsafe_allow_html=True)

            # Download Button
            st.download_button(
                label="📥 Download Density Map",
                data=fig_buffer,
                file_name="density_map.png",
                mime="image/png"
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Sidebar - Information Panel with Fun Fact
    with col_sidebar:
        st.markdown('<div class="stContainer">', unsafe_allow_html=True)

        st.subheader("ℹ️ About PopulusAI")
        st.markdown("""        
        **PopulusAI** is an AI-powered crowd counting tool designed to accurately estimate the number of people in an image.  
        Perfect for public events, safety monitoring, and venue management.  
        - ✅ Uses advanced **deep learning (CSRNet)**  
        - ✅ Fast & Efficient  
        - ✅ Easy to use  
        """)

        st.subheader("💡 Fun Fact About Crowd Counting")
        fact = crowd_counting_fact()
        st.markdown(f"✨ {fact}")

        st.markdown("</div>", unsafe_allow_html=True)

# ---------- Run App ----------
if __name__ == "__main__":
    main()
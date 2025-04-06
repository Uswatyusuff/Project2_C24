import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from io import BytesIO
from PIL import Image

# Step 1: Load YOLOv8 Model
model = YOLO('yolov8l.pt')  # Load your pre-trained YOLOv8 model

# Step 2: Streamlit Interface (Upload Image)
st.title('Head Counting using YOLOv8')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert the uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    image = np.array(image)

    # Convert from RGB (PIL) to BGR (OpenCV)
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Step 3: Run YOLOv8 for head detection
    results = model(image_bgr)  # Run detection

    # Step 4: Annotate the image with detected heads
    annotated_image = results[0].plot()  # Annotate the image with bounding boxes

    # Convert the result back to RGB (for Streamlit to display it correctly)
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Step 5: Display the image with annotations
    st.image(annotated_image_rgb, caption="Detected Heads", use_column_width=True)

    # Step 6: Display the head count
    head_count = len(results[0].boxes)
    st.write(f"Detected Heads: {head_count}")


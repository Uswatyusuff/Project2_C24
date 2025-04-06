from io import BytesIO
from PIL import Image
import numpy as np


from streamlit_app import predict_and_visualize, load_csrnet_model

def test_predict_and_visualize_mock_image():
    # fake image (RGB, 224x224)
    dummy_image = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))

    # Load model
    model = load_csrnet_model()

    # Call  function
    count, fig, buf = predict_and_visualize(dummy_image, model)

    # Assertions
    assert isinstance(count, float)
    assert count >= 0
    assert hasattr(fig, "savefig")  # Matplotlib Figure
    assert isinstance(buf, BytesIO)

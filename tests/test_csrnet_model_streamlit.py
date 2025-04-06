from streamlit_app import load_csrnet_model, predict_and_visualize
from PIL import Image

def test_csrnet_model_loads():
    model = load_csrnet_model()
    assert model is not None

def test_csrnet_prediction_shape():
    image = Image.new("RGB", (512, 512))
    model = load_csrnet_model()
    count, fig, buf = predict_and_visualize(image, model)
    assert isinstance(count, float)
    assert buf.getbuffer().nbytes > 0

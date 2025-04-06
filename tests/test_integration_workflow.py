from PIL import Image

def test_full_prediction_flow():
    from streamlit_app import load_csrnet_model, predict_and_visualize
    model = load_csrnet_model()
    image = Image.new("RGB", (512, 512))
    count, fig, buf = predict_and_visualize(image, model)
    assert isinstance(count, float)
    assert fig is not None
    assert buf is not None

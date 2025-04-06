import subprocess
import time
import os

def test_streamlit_app_boots():
    script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'streamlit_app.py'))

    process = subprocess.Popen(
        ["streamlit", "run", script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(5)  # wait a bit for the app to launch
    process.terminate()

    stdout, stderr = process.communicate()
    assert b"Streamlit" in stdout or b"Streamlit" in stderr

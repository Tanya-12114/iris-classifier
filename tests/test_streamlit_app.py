"""System test: run the Streamlit app headlessly and click Predict."""
import os
from pathlib import Path

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parent.parent


def test_app_loads_and_predicts():
    os.chdir(ROOT)  # the app reads Iris.csv from the working directory
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Iris Flower Species Predictor"
    assert len(at.slider) == 4

    at.button[0].click().run()
    assert not at.exception
    assert any("predicted Iris species" in s.value for s in at.success)

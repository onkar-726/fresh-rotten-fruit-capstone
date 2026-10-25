"""Launch the real Gradio app and call it like the browser does (upload image -> result)."""
import socket
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

pytest.importorskip("tensorflow")
gr = pytest.importorskip("gradio")
gradio_client = pytest.importorskip("gradio_client")

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("use_cam", [False, True])
def test_app_end_to_end(tmp_path, use_cam):
    from fruitfresh.models import build_model

    model_path = tmp_path / "model.keras"
    build_model("baseline_cnn", image_size=(64, 64), pretrained=False).save(model_path)
    img = tmp_path / "fruit.jpg"
    Image.fromarray(np.random.randint(0, 255, (90, 120, 3), dtype="uint8")).save(img)

    sys.path.insert(0, str(ROOT / "app"))
    from app import build_demo

    demo = build_demo(model_path)
    port = free_port()
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
    try:
        client = gradio_client.Client(f"http://127.0.0.1:{port}", verbose=False)
        label, verdict, cam = client.predict(gradio_client.handle_file(str(img)), use_cam, api_name="/classify")
        assert "Predicted class" in verdict and "Confidence" in verdict and ("FRESH" in verdict or "ROTTEN" in verdict)
        assert label["label"] in {"Fresh Apple", "Fresh Banana", "Fresh Orange", "Rotten Apple", "Rotten Banana",
                                  "Rotten Orange"}
        assert (cam is not None) == use_cam
    finally:
        demo.close()

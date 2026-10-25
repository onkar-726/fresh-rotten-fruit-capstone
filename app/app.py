"""Final application: upload a fruit photo -> predicted class, fresh/rotten status and confidence.

    python app/app.py                                   # uses the final model chosen by scripts/compare.py
    python app/app.py --model workspace/runs/mobilenet_v2/seed42/model.keras
    python app/app.py --model results/export/mobilenet_v2.tflite --port 7860
    python app/app.py --host 0.0.0.0 --share            # reachable from other machines / public link

Works with a Keras (.keras) or TensorFlow Lite (.tflite) model. Grad-CAM ("where did the model look?")
is available for .keras models.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fruitfresh.paths import resolve_model  # noqa: E402
from fruitfresh.predict import CONFIDENCE_THRESHOLD, FruitPredictor, preprocess_pil  # noqa: E402


def format_result(r: dict) -> str:
    """Markdown shown to the user."""
    icon = "🟢" if r["status"] == "fresh" else "🔴"
    text = (f"## {icon} {r['fruit'].capitalize()} - {r['status'].upper()}\n\n"
            f"**Predicted class:** {r['label']}  \n"
            f"**Confidence:** {r['confidence']:.1%}  \n"
            f"**Fresh/rotten confidence:** {r['status_confidence']:.1%}")
    if r["uncertain"]:
        text += (f"\n\n⚠️ **Low confidence** (below {CONFIDENCE_THRESHOLD:.0%}). The photo may be unclear, or may not "
                 "show an apple, banana or orange. Try a closer, well-lit photo on a plain background.")
    return text


def build_demo(model_path):
    import gradio as gr

    predictor = FruitPredictor(model_path)

    def run(image, show_cam):
        if image is None:
            return None, "Please upload a photo first.", None
        r = predictor.predict(image)
        cam_img = None
        if show_cam and predictor.model is not None:
            from fruitfresh.gradcam import gradcam_heatmap, overlay_heatmap

            x = preprocess_pil(image, predictor.input_size)[0]
            cam, _, _ = gradcam_heatmap(predictor.model, x)
            cam_img = overlay_heatmap(x.astype("uint8"), cam)
        return r["probabilities"], format_result(r), cam_img

    with gr.Blocks(title="Fresh or Rotten? - Fruit Classifier") as demo:
        gr.Markdown("# 🍎🍌🍊 Fresh or Rotten?\nUpload a photo of an apple, banana or orange. "
                    "The model tells you the fruit, whether it is fresh or rotten, and how sure it is.")
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Fruit photo")
                cam = gr.Checkbox(label="Show where the model looked (Grad-CAM, .keras models only)")
                button = gr.Button("Classify", variant="primary")
            with gr.Column():
                verdict = gr.Markdown()
                label = gr.Label(num_top_classes=6, label="Probability of each class")
                cam_out = gr.Image(label="Grad-CAM heat-map")
        button.click(run, [image, cam], [label, verdict, cam_out], api_name="classify")
        gr.Markdown(f"Model file: `{Path(model_path).name}`")
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="auto", help="'auto', a run name, or a .keras/.tflite path")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    a = ap.parse_args()
    build_demo(resolve_model(a.model)).launch(server_name=a.host, server_port=a.port, share=a.share)


if __name__ == "__main__":
    main()

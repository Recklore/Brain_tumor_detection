from io import BytesIO
from pathlib import Path
import sys
import logging

from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from inference import BrainTumorInferenceEngine, DEVICE
from reporting import generate_structured_report


MODEL_PATH = BASE_DIR / "best_densenet121_scale224.pth"

logging.basicConfig(level=logging.INFO)


def create_app() -> Flask:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")

    engine = BrainTumorInferenceEngine(MODEL_PATH)

    app = Flask(__name__)

    # Allow typical local React dev servers.
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": [
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                ]
            }
        },
    )

    @app.get("/health")
    def health() -> tuple[dict, int]:
        return {
            "status": "ok",
            "device": str(DEVICE),
            "class_names": engine.class_names,
        }, 200

    @app.post("/api/predict")
    def predict() -> tuple[dict, int]:
        file = request.files.get("file")
        if file is None or file.filename == "":
            return {"error": "No image file provided. Use form-data key 'file'."}, 400

        try:
            image = Image.open(BytesIO(file.read())).convert("RGB")
        except Exception:
            return {"error": "Invalid image file."}, 400

        result = engine.predict(image)
        report, report_status = generate_structured_report(result)
        result["report"] = report.model_dump() if report is not None else None
        result["report_status"] = report_status
        return jsonify(result), 200

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)

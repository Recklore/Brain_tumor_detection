import base64
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import models, transforms


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = Path("./best_densenet121_scale224.pth")
IMAGE_PATH = Path("./Dataset_preprocessed/scale_224/pituitary/7.png")
ANNOTATED_OUTPUT_PATH = Path("./inference_annotated.png")

DEFAULT_CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ]
)

CLASS_COLOR_MAP = {
    "glioma": (0, 0, 255),
    "meningioma": (255, 0, 0),
    "pituitary": (0, 255, 255),
}


def build_model(num_classes: int) -> torch.nn.Module:
    model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, 128),
        nn.Linear(128, num_classes),
    )
    return model


def load_model_and_classes(checkpoint_path: Path) -> tuple[torch.nn.Module, list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    else:
        state_dict = checkpoint
        class_names = DEFAULT_CLASS_NAMES

    cleaned_state_dict = {
        k.replace("module.", ""): v for k, v in state_dict.items()
    }

    model = build_model(num_classes=len(class_names))
    model.load_state_dict(cleaned_state_dict)
    model = model.to(DEVICE)
    model.eval()
    return model, class_names


def _resolve_target_layer(model: torch.nn.Module, layer_path: str) -> torch.nn.Module:
    current = model
    for attr in layer_path.split("."):
        if not hasattr(current, attr):
            raise ValueError(f"Invalid Grad-CAM layer path: {layer_path}")
        current = getattr(current, attr)
    if not isinstance(current, torch.nn.Module):
        raise ValueError(f"Grad-CAM layer is not a module: {layer_path}")
    return current


@torch.no_grad()
def predict_probabilities(model: torch.nn.Module, input_tensor: torch.Tensor) -> np.ndarray:
    logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
    return probs


def generate_cam(
    cam_extractor: GradCAM, input_tensor: torch.Tensor, class_idx: int
) -> np.ndarray:
    targets = [ClassifierOutputTarget(class_idx)]
    grayscale_cam = cam_extractor(input_tensor=input_tensor, targets=targets)[0]
    return np.clip(grayscale_cam, 0.0, 1.0)


def cam_to_bboxes(
    cam_map: np.ndarray,
    image_width: int,
    image_height: int,
    binary_thresh: float,
    min_area_px: int,
    max_area_ratio: float,
    cam_peak_thresh: float,
) -> tuple[list[list[int]], float]:
    if cam_map.shape != (image_height, image_width):
        cam_map = cv2.resize(
            cam_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR
        )

    cam_peak = float(cam_map.max())
    if cam_peak < cam_peak_thresh:
        return [], cam_peak

    binary = (cam_map >= binary_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    image_area = image_width * image_height
    boxes: list[list[int]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < min_area_px:
            continue
        if area / float(image_area) > max_area_ratio:
            continue
        boxes.append([int(x), int(y), int(x + w), int(y + h)])

    return boxes, cam_peak


def fallback_box_from_cam(
    cam_map: np.ndarray,
    image_width: int,
    image_height: int,
    min_side_frac: float = 0.12,
) -> list[int] | None:
    if cam_map.shape != (image_height, image_width):
        cam_map = cv2.resize(
            cam_map, (image_width, image_height), interpolation=cv2.INTER_LINEAR
        )

    cam_peak = float(cam_map.max())
    if cam_peak <= 1e-6:
        return None

    adaptive_thresh = max(0.25, float(np.percentile(cam_map, 90.0)))
    binary = (cam_map >= adaptive_thresh).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)
        if w > 4 and h > 4:
            return [int(x), int(y), int(x + w), int(y + h)]

    max_pos = np.unravel_index(np.argmax(cam_map), cam_map.shape)
    y_center, x_center = int(max_pos[0]), int(max_pos[1])

    side = max(int(min(image_width, image_height) * min_side_frac), 20)
    half = side // 2
    x1 = max(0, x_center - half)
    y1 = max(0, y_center - half)
    x2 = min(image_width, x_center + half)
    y2 = min(image_height, y_center + half)

    if x2 <= x1 or y2 <= y1:
        return None
    return [int(x1), int(y1), int(x2), int(y2)]


def box_iou(box_a: list[int], box_b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-8
    return float(inter / union)


def cross_class_nms(
    candidates: list[dict], iou_thresh: float
) -> list[dict]:
    if not candidates:
        return []

    sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
    kept: list[dict] = []

    for candidate in sorted_candidates:
        should_keep = True
        for kept_candidate in kept:
            if box_iou(candidate["box"], kept_candidate["box"]) > iou_thresh:
                should_keep = False
                break
        if should_keep:
            kept.append(candidate)

    return kept


def annotate_image(image: Image.Image, detections: list[dict]) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for detection in detections:
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        class_name = str(detection["class_name"])
        score = float(detection["score"])
        color = CLASS_COLOR_MAP.get(class_name, (0, 255, 0))

        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            bgr,
            f"{class_name}:{score:.2f}",
            (x1, max(12, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    annotated_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb)


def pil_to_base64_png(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class BrainTumorInferenceEngine:
    def __init__(
        self,
        checkpoint_path: Path,
        cam_target_layer: str = "features.denseblock2",
        multilabel_thresh: float = 0.20,
        cam_binary_thresh: float = 0.55,
        min_area_px: int = 400,
        max_area_ratio: float = 0.70,
        cam_peak_thresh: float = 0.60,
        cross_class_iou_thresh: float = 0.70,
    ) -> None:
        self.model, self.class_names = load_model_and_classes(checkpoint_path)
        target_layer = _resolve_target_layer(self.model, cam_target_layer)
        self.cam_extractor = GradCAM(model=self.model, target_layers=[target_layer])

        self.multilabel_thresh = multilabel_thresh
        self.cam_binary_thresh = cam_binary_thresh
        self.min_area_px = min_area_px
        self.max_area_ratio = max_area_ratio
        self.cam_peak_thresh = cam_peak_thresh
        self.cross_class_iou_thresh = cross_class_iou_thresh

        self.detector_label_map = {
            name: idx + 1
            for idx, name in enumerate(self.class_names)
            if name != "notumor"
        }

    def predict(self, image: Image.Image) -> dict:
        image_rgb = image.convert("RGB")
        input_tensor = TRANSFORM(image_rgb).unsqueeze(0).to(DEVICE)

        probs = predict_probabilities(self.model, input_tensor)
        pred_idx = int(np.argmax(probs))
        pred_class = self.class_names[pred_idx]
        confidence = float(probs[pred_idx])

        if pred_class == "notumor":
            active_indices: list[int] = []
        else:
            active_indices = [
                idx
                for idx, class_name in enumerate(self.class_names)
                if class_name != "notumor" and probs[idx] >= self.multilabel_thresh
            ]

            if not active_indices:
                tumor_indices = [
                    idx for idx, class_name in enumerate(self.class_names) if class_name != "notumor"
                ]
                if tumor_indices:
                    fallback_idx = max(tumor_indices, key=lambda idx: probs[idx])
                    active_indices = [fallback_idx]

        image_width, image_height = image_rgb.size
        candidates: list[dict] = []
        for class_idx in active_indices:
            class_name = self.class_names[class_idx]
            cam_map = generate_cam(self.cam_extractor, input_tensor, class_idx)
            boxes, cam_peak = cam_to_bboxes(
                cam_map,
                image_width=image_width,
                image_height=image_height,
                binary_thresh=self.cam_binary_thresh,
                min_area_px=self.min_area_px,
                max_area_ratio=self.max_area_ratio,
                cam_peak_thresh=self.cam_peak_thresh,
            )

            used_fallback = False
            if not boxes:
                fallback_box = fallback_box_from_cam(
                    cam_map,
                    image_width=image_width,
                    image_height=image_height,
                )
                if fallback_box is not None:
                    boxes = [fallback_box]
                    used_fallback = True

            score = float(probs[class_idx])
            det_label = int(self.detector_label_map[class_name])
            for box in boxes:
                candidates.append(
                    {
                        "box": [int(v) for v in box],
                        "label": det_label,
                        "class_name": class_name,
                        "score": score,
                        "cam_peak": float(cam_peak),
                        "fallback": used_fallback,
                    }
                )

        detections = cross_class_nms(
            candidates, iou_thresh=self.cross_class_iou_thresh
        )
        annotated_image = annotate_image(image_rgb, detections)

        return {
            "prediction": pred_class,
            "confidence": confidence,
            "class_probabilities": {
                class_name: float(prob)
                for class_name, prob in zip(self.class_names, probs.tolist())
            },
            "active_classes": [self.class_names[idx] for idx in active_indices],
            "detector_label_map": self.detector_label_map,
            "detections": detections,
            "annotated_image_mime": "image/png",
            "annotated_image_base64": pil_to_base64_png(annotated_image),
        }


def predict_single_image(image_path: Path, engine: BrainTumorInferenceEngine) -> dict:
    image = Image.open(image_path).convert("RGB")
    return engine.predict(image)


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    engine = BrainTumorInferenceEngine(MODEL_PATH)
    result = predict_single_image(IMAGE_PATH, engine)

    print(f"Device: {DEVICE}")
    print(f"Image: {IMAGE_PATH}")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Class probabilities:")
    for class_name, prob in result["class_probabilities"].items():
        print(f"- {class_name}: {prob:.4f}")

    print(f"Detections: {len(result['detections'])}")
    for det in result["detections"]:
        print(
            f"- {det['class_name']}: score={det['score']:.4f}, box={det['box']}, cam_peak={det['cam_peak']:.4f}"
        )

    decoded_png = base64.b64decode(result["annotated_image_base64"])
    ANNOTATED_OUTPUT_PATH.write_bytes(decoded_png)
    print(f"Saved annotated image to: {ANNOTATED_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
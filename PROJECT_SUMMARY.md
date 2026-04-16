# GraBEST - Codebase Summary Report

**Grad-CAM Based Brain Tumor Estimation System**
_Full-stack medical imaging pipeline for brain tumor detection, classification, and localization_

**Team:** Aman Singh Rathour (Lead), Ashish Siyak, Shikhar Dixit, Pradeep Kumar
**Organization:** DuskerAi
**Generated:** 2026-04-16

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Directory Structure](#2-directory-structure)
3. [Dataset & Preprocessing](#3-dataset--preprocessing)
4. [ML Pipeline](#4-ml-pipeline)
5. [Inference Engine](#5-inference-engine)
6. [Backend API](#6-backend-api)
7. [Frontend Application](#7-frontend-application)
8. [Dependencies](#8-dependencies)
9. [Execution Workflow](#9-execution-workflow)
10. [Environment Setup](#10-environment-setup)

---

## 1. Project Overview

GraBEST is an end-to-end brain tumor detection system that combines deep learning classification with Grad-CAM-based localization. The project spans the full lifecycle from raw MRI preprocessing through model training, pseudo-label generation, multi-label fine-tuning, and a production-ready web application with AI-generated medical reports.

**Core capabilities:**
- 4-class brain tumor classification (glioma, meningioma, pituitary, notumor)
- Grad-CAM bounding box localization without manual annotation
- Multi-label detection (multiple tumor types per image)
- AI-generated structured medical reports via Google Gemini
- Full-stack web UI for image upload and result visualization

---

## 2. Directory Structure

```
grafest/
├── Dataset/                          # Raw MRI images (4 classes)
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
├── Dataset_preprocessed/             # Processed images at multiple scales
│   ├── scale_224/
│   ├── scale_112/
│   └── scale_56/
├── backend/                          # Flask REST API
│   ├── app.py                        # Main Flask app (77 lines)
│   ├── config.py                     # Gemini API configuration (62 lines)
│   ├── reporting.py                  # Structured report generation (323 lines)
│   ├── schemas.py                    # Pydantic validation schemas (45 lines)
│   └── requirements.txt
├── frontend/                         # React + Vite web UI
│   ├── src/
│   │   ├── App.jsx                   # Root component
│   │   ├── main.jsx                  # Entry point
│   │   ├── index.css                 # Global styles + animations
│   │   ├── App.css                   # App-level styles
│   │   ├── components/
│   │   │   ├── Upload.tsx            # Upload & results component (425 lines)
│   │   │   ├── Navbar.tsx            # Navigation header (44 lines)
│   │   │   └── UploadImage.tsx       # Upload wrapper
│   │   └── config/
│   │       └── config.ts             # Backend URL config
│   ├── package.json
│   ├── vite.config.js
│   └── dist/                         # Production build
├── inference.py                      # Standalone inference engine (399 lines)
├── multi_label_annotations.json      # Pseudo-label dataset
├── preprocess.ipynb                  # EDA & data preprocessing
├── finetune.ipynb                    # Single-label model training
├── gradcam.ipynb                     # Pseudo-label generation via Grad-CAM
├── multi_label_finetune.ipynb        # Multi-label model training
├── bench.ipynb                       # Benchmarking experiments
└── README.md
```

---

## 3. Dataset & Preprocessing

### Classes

| Class | Description |
|-------|-------------|
| glioma | Glioma tumor |
| meningioma | Meningioma tumor |
| pituitary | Pituitary tumor |
| notumor | Healthy (no tumor) |

### Preprocessing Pipeline (`preprocess.ipynb` - 9 cells)

**Exploratory Data Analysis:**
- Class distribution and imbalance analysis
- Visual inspection across classes
- File-type and corruption checks
- Per-class pixel intensity statistics
- Stratified train/validation split validation

**Image Processing Steps:**
1. **CLAHE** - Contrast Limited Adaptive Histogram Equalization for contrast enhancement
2. **ROI Masking** - Largest contour detection to isolate brain region
3. **Denoising** - Morphological operations + median blur for noise reduction
4. **Multi-scale Export** - Outputs at 224x224, 112x112, and 56x56 pixel resolutions

**Output:** Three preprocessed dataset variants under `Dataset_preprocessed/`

---

## 4. ML Pipeline

### 4.1 Single-Label Training (`finetune.ipynb` - 5 cells)

**Model:** DenseNet121 (ImageNet pretrained) with custom classifier head

```
DenseNet121 Backbone
  └── Dropout(p=0.3)
      └── Linear(in_features → 128)
          └── Linear(128 → 4 classes)
```

**Training Configuration:**
| Parameter | Value |
|-----------|-------|
| Input size | 224x224 (grayscale → 3-channel) |
| Loss | Weighted CrossEntropyLoss |
| Optimizer | Adam / SGD |
| LR Scheduler | CosineAnnealingLR |
| Normalization | ImageNet mean/std |

**Two-Stage Training:**
1. **Warmup** - Freeze backbone, train only classifier head
2. **Fine-tune** - Unfreeze late DenseNet blocks, train end-to-end

**Evaluation:** Macro F1, per-class precision/recall, confusion matrix

**Output:** `best_densenet121_scale224.pth`

---

### 4.2 Grad-CAM Pseudo-Label Generation (`gradcam.ipynb` - 6 cells)

Generates multi-label bounding box annotations from the single-label classifier using Grad-CAM activation maps, eliminating the need for manual annotation.

**Process:**
1. Load trained single-label checkpoint
2. Compute class probabilities per image
3. Generate Grad-CAM heatmaps (target layer: `features.denseblock2`)
4. Convert activation maps to bounding boxes via thresholding + contour analysis
5. Apply cross-class NMS (IoU threshold: 0.70) to suppress overlapping boxes

**Key Thresholds:**
| Parameter | Value |
|-----------|-------|
| CAM binary threshold | 0.55 |
| Min contour area | 400 px² |
| Max area ratio | 0.70 |
| CAM peak threshold | 0.60 |
| Cross-class IoU threshold | 0.70 |

**Fallback Mechanism:** If no boxes are found, uses 90th percentile threshold or largest connected component.

**Output:** `multi_label_annotations.json` containing per-image annotations with bounding boxes, class labels, confidence scores, and CAM peaks.

---

### 4.3 Multi-Label Fine-Tuning (`multi_label_finetune.ipynb` - 7 cells)

Trains the model to detect multiple tumor types per image using pseudo-labels.

**Key Differences from Single-Label:**
- **Labels:** Multi-hot binary targets from `active_classes`
- **Loss:** BCEWithLogitsLoss with per-class `pos_weight`
- **Thresholds:** Per-class decision thresholds optimized on validation set
- **Split:** Stratified by multi-hot signature

**Output:** `best_densenet121_multilabel_scale224.pth` (includes class names and per-class thresholds)

---

### 4.4 Benchmarking (`bench.ipynb` - 6 cells)

Experimental notebook for model performance analysis and hyperparameter tuning across training configurations.

---

## 5. Inference Engine (`inference.py` - 399 lines)

The `BrainTumorInferenceEngine` class encapsulates the full prediction pipeline.

### Prediction Flow

```
Input Image (PIL)
  → Resize 224x224, Grayscale→RGB, Normalize
  → DenseNet121 Forward Pass
  → Softmax Probabilities
  → Active Class Selection (threshold ≥ 0.20)
  → Per-Class Grad-CAM Generation (target: features.denseblock2)
  → CAM → Bounding Boxes (contour analysis + fallback)
  → Cross-Class NMS (IoU 0.70)
  → Annotated Image + Detection Results
```

### Output Format

```python
{
    "prediction": str,              # Top predicted class
    "confidence": float,            # Top class probability
    "class_probabilities": dict,    # All class probabilities
    "active_classes": list,         # Classes above threshold
    "detector_label_map": dict,     # Class → label ID mapping
    "detections": list,             # Bounding box detections
    "annotated_image_mime": str,    # "image/png"
    "annotated_image_base64": str   # Base64-encoded annotated image
}
```

### Visualization Colors

| Class | Color |
|-------|-------|
| glioma | Blue (0, 0, 255) |
| meningioma | Red (255, 0, 0) |
| pituitary | Cyan (0, 255, 255) |

---

## 6. Backend API

**Stack:** Flask 3.0+, flask-cors, Pydantic, Google Generative AI (Gemini)

### Endpoints

#### `GET /health`
Returns service status, device (CUDA/CPU), and loaded class names.

#### `POST /api/predict`
Accepts multipart/form-data image upload, returns prediction results with optional AI report.

**Response Fields:**
| Field | Type | Description |
|-------|------|-------------|
| prediction | string | Top predicted class |
| confidence | float | Prediction confidence |
| class_probabilities | object | All class probabilities |
| active_classes | array | Classes above threshold |
| detections | array | Bounding box results |
| annotated_image_base64 | string | Base64 annotated image |
| report | object/null | Structured AI report |
| report_status | string | Report generation status |

### AI Report Generation (`reporting.py` - 323 lines)

Uses Google Gemini to generate structured medical reports from model predictions.

**StructuredReport Schema:**
| Field | Type | Description |
|-------|------|-------------|
| diagnosis_summary | string | Summary of findings |
| risk_level | enum | "low", "moderate", "high" |
| key_findings | list (max 4) | Notable findings |
| recommendations | list (max 4) | Suggested next steps |
| confidence_note | string | Model confidence context |
| disclaimer | string | AI-generated content notice |

**Resilience Features:**
- Multi-model fallback chain (Gemini 2.0 Flash → 1.5 Flash variants)
- Quota detection and per-model cooldown tracking
- Instructor library integration for typed extraction
- Graceful degradation (returns null report on failure)

### Configuration (`config.py`)

| Env Variable | Default | Description |
|-------------|---------|-------------|
| `GEMINI_API_KEY` | None | API key (required for reports) |
| `GEMINI_MODEL` | gemini-2.0-flash | Preferred model |
| `GEMINI_FALLBACK_MODELS` | 4 flash variants | Fallback chain |
| `GEMINI_TIMEOUT_SECONDS` | 20 | Request timeout |
| `GEMINI_MAX_OUTPUT_TOKENS` | 400 | Max response tokens |
| `GEMINI_TEMPERATURE` | 0.2 | Generation temperature |

---

## 7. Frontend Application

**Stack:** React 19, Vite 8, Tailwind CSS 4, MUI Joy, TypeScript, Axios

### Components

#### `App.jsx` - Root Layout
- Navbar with DuskerAi branding
- Hero section with gradient background and blur decorations
- UploadImage component
- Footer

#### `Navbar.tsx` - Navigation
- Sticky header with red-to-rose gradient
- Links: All Work, Team, References
- Animated hover effects

#### `Upload.tsx` - Core Interaction (425 lines)
The primary component handling the entire user workflow:

**Upload State:**
- Drag-and-drop zone with file validation (jpg, jpeg, png)
- Image preview with medical scanning line animation
- Submit/Clear actions

**Results Display:**
- 3:2 column split layout (annotated image + sidebar)
- Class probability bar charts with per-class color coding
- AI Structured Report card (diagnosis, risk level, findings, recommendations)
- Detection detail listing (coordinates, dimensions, CAM peaks)

**UI Polish:**
- Custom scan-line animation (2.2s medical sweep effect)
- Fade-in-up transitions for results
- Pulse ring animation for tumor markers
- Toast notifications for errors/success
- Responsive layout (mobile to desktop)

### Color Scheme

| Element | Color |
|---------|-------|
| Primary gradient | Red-700 → Rose-700 |
| Background | Light pink (#fff5f5 → #ffe4e6) |
| Glioma bar | Red-500 |
| Meningioma bar | Amber-500 |
| Pituitary bar | Blue-500 |
| Notumor bar | Emerald-500 |

### Build Config
- **Dev server:** `http://localhost:5173` (Vite HMR)
- **Backend URL:** `http://localhost:8080` (configured in `config.ts`)
- **CORS:** Allowed from localhost:5173 and 127.0.0.1:5173

---

## 8. Dependencies

### Backend (`requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| flask | >=3.0.0 | Web framework |
| flask-cors | >=4.0.0 | CORS support |
| numpy | >=1.24.0 | Numerical operations |
| opencv-python | >=4.9.0 | Image processing |
| pillow | >=10.0.0 | Image I/O |
| grad-cam | >=1.5.4 | Grad-CAM visualization |
| google-generativeai | >=0.8.5 | Gemini API client |
| pydantic | >=2.8.0 | Data validation |
| instructor | >=1.6.3 | Structured LLM extraction |
| python-dotenv | >=1.0.1 | Environment variables |
| torch | latest | PyTorch ML framework |
| torchvision | latest | Vision models & transforms |

### Frontend (`package.json`)

| Package | Version | Purpose |
|---------|---------|---------|
| react | 19.2.4 | UI framework |
| vite | 8.0.4 | Build tool |
| tailwindcss | 4.2.2 | Utility CSS |
| @mui/joy | 5.0.0-beta.52 | UI components |
| axios | 1.15.0 | HTTP client |
| react-router-dom | 7.14.1 | Routing |
| react-toastify | 11.0.5 | Notifications |
| lucide-react | 1.8.0 | Icons |

---

## 9. Execution Workflow

The project follows a sequential pipeline from data to deployment:

```
Step 1: preprocess.ipynb
  └── Raw Dataset/ → Dataset_preprocessed/ (3 scales)

Step 2: finetune.ipynb
  └── Dataset_preprocessed/scale_224/ → best_densenet121_scale224.pth

Step 3: gradcam.ipynb
  └── best_densenet121_scale224.pth → multi_label_annotations.json

Step 4: multi_label_finetune.ipynb
  └── multi_label_annotations.json → best_densenet121_multilabel_scale224.pth

Step 5: Deploy
  ├── backend/app.py   (Flask API on :8080)
  └── frontend/        (React app on :5173)
```

---

## 10. Environment Setup

### Backend

```bash
cd backend
pip install -r requirements.txt

# Optional: enable AI report generation
export GEMINI_API_KEY=your_key_here

python app.py
# Server starts on http://0.0.0.0:8080
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# Dev server on http://localhost:5173
```

### Model Checkpoints Required

Place these in the project root (generated by the training notebooks):
- `best_densenet121_scale224.pth` - Single-label classifier
- `best_densenet121_multilabel_scale224.pth` - Multi-label classifier (if using multi-label mode)

---

## File Size Reference

| File | Size/Lines |
|------|-----------|
| `inference.py` | 399 lines |
| `backend/app.py` | 77 lines |
| `backend/config.py` | 62 lines |
| `backend/reporting.py` | 323 lines |
| `backend/schemas.py` | 45 lines |
| `frontend/src/components/Upload.tsx` | 425 lines |
| `frontend/src/components/Navbar.tsx` | 44 lines |
| `preprocess.ipynb` | ~1.3 MB (9 cells) |
| `finetune.ipynb` | ~57 KB (5 cells) |
| `gradcam.ipynb` | ~497 KB (6 cells) |
| `multi_label_finetune.ipynb` | ~32 KB (7 cells) |
| `bench.ipynb` | ~13 KB (6 cells) |

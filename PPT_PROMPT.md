# AI PPT Maker Prompt - GraBEST Project

> Copy everything below the line and paste it into your AI PPT maker (Gamma, SlidesAI, Beautiful.ai, Tome, etc.)

---

Create a professional, visually polished presentation (22-25 slides) for a college/hackathon project demo. The project is called **GraBEST** (Grad-CAM Based Brain Tumor Estimation System). It is a complete end-to-end AI-powered medical imaging platform that detects, classifies, and localizes brain tumors from MRI scans and generates AI medical reports. The presentation should look modern, clean, and medical/tech themed with a dark or deep-blue/teal color palette accented with red/rose highlights.

**Organization:** DuskerAi
**Team:**
- Aman Singh Rathour (Team Leader)
- Ashish Siyak (Team Member)
- Shikhar Dixit (Team Member)
- Pradeep Kumar (Team Member)

---

## SLIDE-BY-SLIDE CONTENT

### SLIDE 1 - Title Slide
- Title: **GraBEST - Grad-CAM Based Brain Tumor Estimation System**
- Subtitle: "AI-Powered Brain Tumor Detection, Classification & Localization from MRI Scans"
- Organization: DuskerAi
- Team: Aman Singh Rathour (Lead), Ashish Siyak, Shikhar Dixit, Pradeep Kumar
- Use a background with a subtle brain MRI scan or neural network visual

### SLIDE 2 - Problem Statement
- Brain tumors are among the most lethal cancers, requiring early and accurate detection for effective treatment
- Manual MRI analysis by radiologists is time-consuming, subjective, and prone to human error
- There is a critical shortage of trained neuroradiologists, especially in developing countries
- Delayed diagnosis leads to worse patient outcomes and higher mortality rates
- Key stat: Brain tumors account for 85-90% of all primary CNS tumors (WHO)
- Need: An automated, explainable, and accessible AI system to assist radiologists

### SLIDE 3 - Our Solution
- GraBEST: A full-stack AI platform that automates brain tumor detection from MRI images
- Four core capabilities:
  1. **4-Class Classification** - Glioma, Meningioma, Pituitary, No Tumor
  2. **Grad-CAM Localization** - Visual bounding boxes showing exactly where the tumor is, without needing manual bounding box annotations
  3. **Multi-Label Detection** - Can detect multiple tumor types in a single scan
  4. **AI Medical Reports** - Auto-generated structured diagnosis reports using Google Gemini
- End-to-end: from raw MRI preprocessing to a web application doctors can use

### SLIDE 4 - System Architecture (High-Level Overview)
- Show a flowchart/architecture diagram with these components connected:
  - Raw MRI Dataset (4 classes) → Preprocessing Pipeline → Preprocessed Data (3 scales)
  - Preprocessed Data → DenseNet121 Single-Label Training → Trained Model (.pth)
  - Trained Model → Grad-CAM Pseudo-Label Generator → Multi-Label Annotations (JSON)
  - Multi-Label Annotations → Multi-Label Fine-Tuning → Final Multi-Label Model
  - Final Model → Flask Backend API → React Frontend Web App
  - Backend also connects to Google Gemini API for report generation
- Label each connection with the artifact it produces

### SLIDE 5 - Dataset Overview
- Source: Brain Tumor MRI Dataset with 4 classes
- Classes and distribution:
  - **Glioma** - Aggressive, infiltrative brain tumor
  - **Meningioma** - Tumor arising from the meninges
  - **Pituitary** - Tumor in the pituitary gland
  - **No Tumor** - Healthy brain scans (control)
- Show a 2x2 grid of sample MRI images, one from each class
- Mention class imbalance challenges that were handled during training

### SLIDE 6 - Data Preprocessing Pipeline
- Title: "Preprocessing: From Raw MRI to Clean Multi-Scale Data"
- Step-by-step pipeline with icons/visuals for each:
  1. **Exploratory Data Analysis** - Class distribution, corruption checks, intensity histograms
  2. **CLAHE Enhancement** - Contrast Limited Adaptive Histogram Equalization to improve tissue visibility
  3. **ROI Masking** - Largest contour detection to isolate the brain region and remove background noise
  4. **Denoising** - Morphological operations + median blur to clean artifacts
  5. **Multi-Scale Export** - Output at 3 resolutions: 224x224, 112x112, 56x56 pixels
- Show before/after image comparison (raw vs preprocessed)
- Output: Clean, standardized datasets ready for model training

### SLIDE 7 - Why DenseNet121?
- Title: "Model Selection: DenseNet121"
- DenseNet121 architecture highlights:
  - **Dense connectivity** - Every layer receives input from all preceding layers (feature reuse)
  - **Parameter efficient** - Fewer parameters than ResNet while achieving better accuracy
  - **Strong gradient flow** - Alleviates vanishing gradient problem, ideal for medical imaging
  - **ImageNet pretrained** - Leverages transfer learning from 1.2M natural images
- Why it fits our use case:
  - Proven track record in medical image classification tasks
  - Good balance between model size and accuracy
  - Dense blocks capture fine-grained features important for tumor detection
- Custom classifier head: Dropout(0.3) → Linear(1024 → 128) → Linear(128 → 4)

### SLIDE 8 - Single-Label Training Strategy
- Title: "Stage 1: Single-Label Classification Training"
- Two-stage transfer learning approach:
  - **Stage 1 - Warmup (Head Only):** Freeze entire DenseNet121 backbone, train only the custom classifier head. This prevents destroying pretrained features early.
  - **Stage 2 - Fine-Tune:** Unfreeze late DenseNet dense blocks, train end-to-end with lower learning rate. This adapts deep features to brain MRI domain.
- Training details:
  - Input: 224x224 grayscale MRI converted to 3-channel RGB
  - Loss: Weighted CrossEntropyLoss (handles class imbalance by giving more weight to minority classes)
  - Optimizer: Adam with CosineAnnealingLR scheduler
  - Early stopping based on validation loss
  - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
- Output artifact: best_densenet121_scale224.pth

### SLIDE 9 - Single-Label Training Results
- Title: "Classification Performance"
- Show evaluation metrics:
  - Macro F1 Score
  - Per-class Precision, Recall, F1 (as a table)
  - Confusion Matrix heatmap (4x4 grid: glioma, meningioma, notumor, pituitary)
- Highlight key insights:
  - Which classes perform best/worst
  - Overall accuracy and reliability
- Note: These results validate the model before using it for Grad-CAM pseudo-label generation

### SLIDE 10 - What is Grad-CAM? (Explainability)
- Title: "Grad-CAM: Making AI Decisions Visible"
- Explain Grad-CAM (Gradient-weighted Class Activation Mapping):
  - Computes gradients of the target class score with respect to feature maps in a convolutional layer
  - Produces a heatmap highlighting which regions of the image were most important for the prediction
  - No additional training needed - uses the existing trained model
- Why it matters for medical AI:
  - **Explainability** - Doctors can see WHY the AI made its decision
  - **Trust** - Visual evidence builds confidence in AI predictions
  - **Localization** - Points to the exact tumor region without needing manually annotated bounding boxes
- Show a sample: Original MRI → Grad-CAM heatmap overlay → Extracted bounding box
- Target layer used: features.denseblock2

### SLIDE 11 - Pseudo-Label Generation Pipeline
- Title: "From Heatmaps to Bounding Boxes: Automated Annotation"
- This is our key innovation - generating bounding box annotations WITHOUT manual labeling
- Step-by-step process:
  1. Run trained classifier on every image → get class probabilities
  2. Identify "active classes" (probability ≥ 0.20 threshold)
  3. Generate Grad-CAM heatmap for each active class
  4. Convert heatmap to binary mask (threshold: 0.55)
  5. Find contours in mask → extract bounding boxes
  6. Filter boxes: minimum area 400px², max 70% of image, CAM peak ≥ 0.60
  7. Apply Cross-Class NMS (IoU threshold: 0.70) to remove duplicate overlapping boxes
- Fallback: If no box found, use 90th percentile threshold or largest connected component
- Output: multi_label_annotations.json with per-image bounding box annotations

### SLIDE 12 - Pseudo-Label Output Format
- Title: "Generated Annotation Structure"
- Show the JSON structure:
  - Meta configuration (all thresholds and settings)
  - Classifier class names: [glioma, meningioma, notumor, pituitary]
  - Detector label map: {glioma: 1, meningioma: 2, pituitary: 3}
  - Per-image entries containing:
    - Image path and ground truth class
    - Class probabilities from the classifier
    - Active classes detected
    - Bounding box annotations with coordinates [x1, y1, x2, y2], label, class name, confidence score, and CAM peak value
- Highlight: This eliminates the most expensive part of object detection - manual annotation
- Show a sample annotated image with Grad-CAM bounding boxes overlaid

### SLIDE 13 - Multi-Label Fine-Tuning
- Title: "Stage 2: Multi-Label Detection Training"
- Why multi-label? A patient could potentially have multiple abnormalities visible in one scan
- Training using the pseudo-labels generated in the previous step
- Key differences from single-label:
  - **Labels:** Multi-hot binary vectors instead of single class index
  - **Loss:** BCEWithLogitsLoss (Binary Cross Entropy) with per-class pos_weight for imbalance
  - **Stratification:** Split by multi-hot signature pattern
  - **Thresholds:** Per-class decision thresholds optimized on validation set (not just argmax)
- Same two-stage warmup + fine-tune approach
- Output: best_densenet121_multilabel_scale224.pth (saves class names + per-class thresholds)

### SLIDE 14 - Inference Engine Deep Dive
- Title: "Real-Time Prediction Pipeline"
- Show the complete inference flow as a visual pipeline:
  ```
  Upload MRI Image
    → Resize to 224x224, Convert Grayscale to 3-channel RGB
    → ImageNet Normalization
    → DenseNet121 Forward Pass
    → Softmax → Class Probabilities
    → Select Active Classes (threshold ≥ 0.20)
    → Generate Grad-CAM per Active Class (target: denseblock2)
    → Convert CAM Heatmaps to Bounding Boxes
    → Cross-Class NMS (IoU: 0.70)
    → Draw Color-Coded Boxes on Image
    → Return: Prediction + Confidence + Probabilities + Annotated Image + Detections
  ```
- Color coding: Glioma = Blue, Meningioma = Red, Pituitary = Cyan
- Runs on GPU (CUDA) if available, falls back to CPU
- Built as a reusable Python class: BrainTumorInferenceEngine

### SLIDE 15 - Backend Architecture
- Title: "Flask REST API Backend"
- Tech stack: Flask 3.0, flask-cors, Pydantic, PyTorch, Google Gemini AI
- API Endpoints:
  - **GET /health** - Service status, device info, loaded classes
  - **POST /api/predict** - Upload MRI image → Get full prediction results + AI report
- Request/Response flow:
  1. Client uploads image (multipart/form-data)
  2. Server loads image, converts to RGB
  3. Inference engine runs prediction
  4. Gemini API generates structured medical report
  5. Returns JSON: prediction, confidence, probabilities, annotated image (base64), detections, report
- CORS enabled for frontend dev server (localhost:5173)
- Server runs on port 8080

### SLIDE 16 - AI-Powered Medical Reports
- Title: "Structured Diagnosis Reports via Google Gemini"
- After model prediction, the system generates a human-readable medical report using Gemini AI
- Report schema (validated with Pydantic):
  - **Diagnosis Summary** - Concise summary of findings
  - **Risk Level** - Low / Moderate / High
  - **Key Findings** - Up to 4 notable observations
  - **Recommendations** - Up to 4 suggested next steps
  - **Confidence Note** - Context about model certainty
  - **Disclaimer** - "AI-generated for informational use only"
- Resilience features:
  - Multi-model fallback chain: Gemini 2.0 Flash → Gemini 1.5 Flash (4 variants)
  - Automatic quota detection with per-model cooldown
  - Graceful degradation: if Gemini is unavailable, prediction still works without report
  - Structured extraction via instructor library for type-safe responses
- Temperature: 0.2 (low = consistent, deterministic outputs)

### SLIDE 17 - Frontend Application
- Title: "React Web Interface"
- Tech stack: React 19, Vite 8, Tailwind CSS 4, MUI Joy, TypeScript, Axios
- User flow:
  1. **Upload** - Drag-and-drop or click to select MRI image (supports JPG, PNG)
  2. **Scan** - Medical scanning line animation while processing
  3. **Results** - Rich results display with multiple sections
- Results page layout:
  - Left (60%): Annotated MRI image with color-coded bounding boxes on dark background
  - Right (40%): Sidebar with class probability bar charts (color-coded per class)
  - Below: AI Structured Report card with diagnosis, risk level, findings, recommendations
  - Below: Detection details table with box coordinates, sizes, CAM peak values
- Design: Red-to-rose gradient theme, medical aesthetic, responsive (mobile to desktop)

### SLIDE 18 - UI/UX Highlights
- Title: "Polished User Experience"
- Show screenshots or mockups of the frontend
- Animations and micro-interactions:
  - Scan-line sweep animation (2.2s loop) during image analysis - mimics medical scanning
  - Fade-in-up transitions for results appearing
  - Pulse ring animation on tumor location markers
  - Toast notifications for errors, success, and loading states
- Color scheme:
  - Primary: Red-700 to Rose-700 gradient
  - Background: Soft pink (#fff5f5 to #ffe4e6)
  - Class bars: Glioma=Red, Meningioma=Amber, Pituitary=Blue, NoTumor=Emerald
- Responsive design with Tailwind breakpoints (mobile → tablet → desktop)
- Real-time feedback with loading states and error handling

### SLIDE 19 - Complete Tech Stack Summary
- Title: "Technology Stack"
- Show as a layered architecture or grid:
- **ML/AI Layer:**
  - PyTorch + TorchVision (model training & inference)
  - DenseNet121 (backbone architecture)
  - Grad-CAM library (explainability & localization)
  - Google Gemini AI (report generation)
  - Pydantic + Instructor (structured data validation)
- **Backend Layer:**
  - Python 3.x
  - Flask 3.0 (REST API framework)
  - OpenCV + Pillow (image processing)
  - NumPy (numerical computing)
- **Frontend Layer:**
  - React 19 (UI framework)
  - TypeScript (type safety)
  - Vite 8 (build tool with HMR)
  - Tailwind CSS 4 (utility-first styling)
  - MUI Joy (component library)
  - Axios (HTTP client)
- **Data Processing:**
  - CLAHE, morphological operations, contour analysis
  - Multi-scale image export (224, 112, 56)
  - Stratified splitting, class weighting

### SLIDE 20 - Project Workflow Summary
- Title: "End-to-End Pipeline"
- Show as a horizontal or vertical flowchart with 5 major stages:
  1. **PREPROCESS** → Raw MRI images cleaned via CLAHE, ROI masking, denoising, exported at 3 scales
  2. **TRAIN (Single-Label)** → DenseNet121 fine-tuned for 4-class classification with two-stage transfer learning
  3. **GENERATE (Pseudo-Labels)** → Grad-CAM heatmaps converted to bounding box annotations automatically
  4. **TRAIN (Multi-Label)** → Model fine-tuned for multi-label detection using pseudo-labels with per-class thresholds
  5. **DEPLOY** → Flask API + React frontend serving predictions with Grad-CAM visualization and AI reports
- Each stage shows its input artifact → process → output artifact

### SLIDE 21 - Key Innovations & Differentiators
- Title: "What Makes GraBEST Unique?"
- **Zero Manual Annotation** - Uses Grad-CAM to auto-generate bounding box labels, eliminating costly manual annotation by radiologists
- **Explainable AI** - Every prediction comes with a visual Grad-CAM heatmap showing exactly where the model is looking
- **Multi-Label Detection** - Can identify multiple tumor types in a single scan, unlike most binary/single-class systems
- **AI-Generated Reports** - Structured medical reports via Gemini with diagnosis summary, risk level, findings, and recommendations
- **Robust Fallbacks** - Multi-model fallback for report generation, fallback box extraction for Grad-CAM, graceful degradation throughout
- **Full-Stack Deployment** - Not just a notebook - a production-ready web application anyone can use

### SLIDE 22 - Challenges & How We Solved Them
- Title: "Challenges Faced"
- **Class Imbalance** → Weighted loss functions (CrossEntropyLoss with class weights, BCEWithLogitsLoss with pos_weight)
- **No Bounding Box Annotations** → Grad-CAM pseudo-label pipeline with contour analysis and fallback mechanisms
- **Low-Contrast MRI Scans** → CLAHE preprocessing + ROI masking to enhance tissue visibility
- **Noisy CAM Outputs** → Multi-threshold filtering, minimum area constraints, cross-class NMS
- **API Quota Limits** → Multi-model fallback chain with cooldown tracking for Gemini
- **Multi-Label Threshold Selection** → Per-class threshold optimization on validation set instead of fixed global threshold

### SLIDE 23 - Future Scope
- Title: "What's Next?"
- **3D Volumetric Analysis** - Extend from 2D slices to full 3D MRI volume analysis
- **More Tumor Types** - Expand beyond 4 classes to cover rare tumor subtypes
- **Object Detection Model** - Train a dedicated YOLO/Faster R-CNN using the pseudo-labels as initialization
- **DICOM Support** - Direct ingestion of hospital DICOM files
- **Federated Learning** - Train across hospitals without sharing patient data (privacy-preserving)
- **Mobile App** - Lightweight model deployment on mobile devices for field use
- **Clinical Validation** - Partner with hospitals for real-world validation studies
- **Segmentation** - Pixel-level tumor boundary segmentation for surgical planning

### SLIDE 24 - Live Demo / Screenshots
- Title: "GraBEST in Action"
- Show 2-3 screenshots of the web application:
  1. Upload screen with drag-and-drop interface
  2. Results screen showing annotated MRI with bounding boxes, probability bars, and AI report
  3. Close-up of the structured report card (diagnosis, risk, findings, recommendations)
- Or mention: "Live Demo Available" with the local setup instructions

### SLIDE 25 - Thank You / Q&A
- Title: "Thank You!"
- Team: Aman Singh Rathour (Lead), Ashish Siyak, Shikhar Dixit, Pradeep Kumar
- Organization: DuskerAi
- "Questions?"
- Optional: Add GitHub repo link or contact information

---

## DESIGN INSTRUCTIONS FOR THE AI PPT MAKER

- **Theme:** Modern, professional, medical-tech aesthetic
- **Color palette:** Deep navy/teal primary, with red/rose accents (#dc2626, #e11d48). White text on dark slides or dark text on light slides. Use the red-to-rose gradient for highlights and call-to-action elements.
- **Typography:** Clean sans-serif font (Inter, Poppins, or similar). Large bold headings, clean body text.
- **Visuals:** Use medical imaging icons, brain/neural network illustrations, flowchart diagrams, and code snippets where appropriate. Show architecture diagrams as connected boxes with arrows.
- **Layout:** Minimal text per slide, use bullet points, tables, and visual diagrams wherever possible. Avoid walls of text.
- **Slide transitions:** Subtle fade or slide transitions. No distracting animations.
- **Charts/Tables:** Use clean tables for training parameters, metrics, and tech stack comparisons. Use bar charts for class distributions and performance metrics.
- **Code blocks:** Show small code snippets in monospace font with syntax highlighting where relevant (model architecture, JSON structure).
- **Consistency:** Maintain consistent header placement, font sizes, and color usage across all slides.

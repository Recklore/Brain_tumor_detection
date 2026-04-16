# Brain Tumor Detection: End-to-End Research Journey

## Team Details
- Team Leader: Aman Singh Rathour
- Team Member 1: Ashish Siyak
- Team Member 2: Shikhar Dixit
- Team Member 3: Pradeep Kumar

## Preprocessed Data Drive Link
- Link: https://drive.google.com/file/d/10gTcJ4Op52Dvop_PcygV3kLTykdEtbvo/view?usp=sharing

## What We Built
This project started as a brain tumor classification problem and evolved into a richer clinical-assist workflow that combines:
- tumor class prediction,
- tumor localization with bounding boxes,
- and an AI-generated structured report for faster doctor-side review.

The complete development path was:
1. EDA on raw MRI data.
2. Image preprocessing and multi-scale dataset creation.
3. DenseNet121 finetuning for multi-class classification.
4. Custom CNN (from scratch) benchmarking to compare against DenseNet.
5. Transition planning from multi-class to multi-label modeling.
6. Grad-CAM based tumor localization and bounding-box generation.
7. Structured AI report integration for clinical convenience.
8. Multi-label finetuning on pseudo-labeled multi-label data.

## 1) EDA: Understanding the MRI Dataset
In the EDA stage, we focused on understanding data quality and class behavior before training:
- checked class distribution and imbalance,
- visualized representative MRI samples per class,
- inspected corruption and file-level quality,
- analyzed per-class pixel intensity statistics and histograms,
- validated stratified train/validation splitting behavior.

This stage made it clear where imbalance and data variation could impact training, and informed the loss weighting and augmentation decisions used later.

## 2) Preprocessing: Standardizing MRI Inputs
The preprocessing pipeline was designed to improve contrast, suppress noise, and keep useful anatomy:
- CLAHE for local contrast enhancement,
- ROI masking using the largest contour,
- denoising via morphological operations and median blur,
- multi-scale image generation (224, 112, 56).

These steps improved consistency across scans and produced the training-ready dataset used in later stages.

## 3) DenseNet121 Finetuning for Multi-Class Classification
We trained a DenseNet121-based classifier on four classes (glioma, meningioma, notumor, pituitary):
- used weighted cross-entropy to handle imbalance,
- applied staged transfer learning (warmup with frozen backbone, then selective unfreezing),
- used scheduler + early stopping to stabilize training.

At this point, prediction was multi-class with softmax, which gives one dominant class per image.

## 4) Custom CNN Benchmark (From Scratch)
To validate whether transfer learning was truly beneficial, we also built and trained a custom CNN from scratch in bench.ipynb and compared it against DenseNet121.

This comparison helped justify architectural choices and showed why a pretrained backbone was preferred for this MRI setting.

## 5) Multi-Class to Multi-Label: Why the Shift
A key challenge emerged: real tumor patterns may not always behave as strictly single-label from a modeling perspective, especially when we want richer clinical cues.

Initial plan:
- move from softmax head (single-class competition) to sigmoid-based multi-label outputs,
- train with BCEWithLogitsLoss so each class can be scored independently.

This planning led to a deeper question: can we move beyond only classifying and also localize where the tumor evidence is in the scan?

## 6) Grad-CAM: From Classification to Localization
Grad-CAM became the turning point.

Instead of only producing class probabilities, we used Grad-CAM heatmaps to:
- identify image regions driving predictions,
- convert heatmaps into tumor candidate boxes,
- refine boxes using filtering and cross-class NMS,
- draw bounding boxes directly on MRI images.

So the system progressed from "what class is this?" to "what class, and where is the tumor likely located?"

This localization output was also exported as pseudo-annotations, which supported the next multi-label training stage.

## 7) Doctor-Facing AI Report + Boxed MRI Output
After adding tumor localization, we added a structured AI report feature so the output is easier to interpret quickly:
- MRI image with proper bounding boxes,
- class probabilities and confidence,
- structured report fields such as summary, risk level, key findings, and recommendations.

The goal is convenience in clinical review workflows: visual evidence (boxes) plus concise textual interpretation in one response.

## 8) Multi-Label Finetuning on Pseudo-Labeled Data
Using the generated multi-label annotations, we finetuned a multi-label DenseNet variant:
- multi-hot target vectors built from active classes,
- BCEWithLogitsLoss with class-wise positive weighting,
- threshold tuning per class on validation outputs.

This stage operationalized the earlier sigmoid-head plan and aligned training with the multi-label objective.

## Final Outcome
The final system is not just a classifier.
It combines:
- multi-stage training and benchmarking,
- Grad-CAM-based tumor localization with bounding boxes,
- and AI-assisted reporting for practical MRI analysis support.

In short, the project evolved from basic multi-class prediction into a richer detection-and-interpretation pipeline designed for more actionable outputs.

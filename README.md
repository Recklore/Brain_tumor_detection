# Brain Tumor Dataset EDA and Preprocessing

## Team Details
- Team Leader: Aman Singh Rathour
- Team Member 1: Ashish Siyak
- Team Member 2: Shikhar Dixit
- Team Member 3: Pradeep Kumar

## Preprocessed Data Drive Link
- Link: (https://drive.google.com/file/d/1C_q2wwtPo_p1cjK5Up4pJ467wKoOkGqa/view?usp=sharing)

## Project Overview
This project prepares a brain tumor MRI image dataset for model training.
The workflow is implemented in preprocess.ipynb and has two major parts:
1. Exploratory Data Analysis (EDA) on the raw dataset.
2. Image preprocessing and multi-scale dataset generation.

Dataset classes used:
- glioma
- meningioma
- notumor
- pituitary

## File and Folder Structure
- Dataset/: raw class-wise MRI images
- Dataset_preprocessed/: generated preprocessed outputs
  - scale_224/
  - scale_112/
  - scale_56/
- preprocess.ipynb: EDA and preprocessing notebook

## EDA Implemented in preprocess.ipynb
The notebook performs the following EDA tasks on the raw dataset:

### 1) Class Distribution and Imbalance
- Counts samples per class.
- Computes total samples.
- Plots a bar chart with count and percentage labels.
- Reports class imbalance ratio (max/min).

### 2) Raw Sample Visualization
- Displays a gallery of 8 grayscale MRI samples per class.
- Helps visually inspect class diversity and image quality.

### 3) File and Data Quality Checks
- Reports file extension distribution.
- Scans for unreadable/corrupted images.
- Summarizes most common image dimensions.
- Summarizes file size statistics per class and plots class-wise file-size boxplots.

### 4) Pixel Intensity Analysis
- Computes per-class pixel statistics:
  - min, max, mean, std, p5, p50, p95
- Plots separate intensity histograms per class.

### 5) Train/Validation Split Sanity Check
- Performs stratified train/validation split.
- Shows per-class train and validation counts.
- Displays a summary table of split percentages.
- Plots:
  - grouped bar chart (train vs validation counts)
  - heatmap of split percentages

## Preprocessing Implemented in preprocess.ipynb
The notebook defines and applies the following preprocessing pipeline:

### 1) CLAHE Contrast Enhancement
- Improves local contrast on grayscale MRI images.

### 2) ROI Masking
- Applies thresholding and contour detection.
- Keeps the largest contour region as the area of interest.

### 3) Denoising
- Applies morphological operations and median blur to reduce noise.

### 4) Multi-Scale Image Pyramid
- Generates resized outputs at:
  - 224 x 224
  - 112 x 112
  - 56 x 56

### 5) Dataset Export
- Saves processed images as PNG files to:
  - Dataset_preprocessed/scale_224/<class_name>/
  - Dataset_preprocessed/scale_112/<class_name>/
  - Dataset_preprocessed/scale_56/<class_name>/

## Notebook Execution Order
Run preprocess.ipynb from top to bottom in sequence:
1. Imports
2. Dataset class definition
3. Transform setup
4. Dataset loading
5. EDA analysis cells
6. Train/validation split and class-weight computation
7. Preprocessing function definitions
8. Final preprocessing execution cell

## Main Dependencies
- torch
- torchvision
- opencv-python (cv2)
- scikit-learn
- numpy
- pandas
- matplotlib
- pillow

## Expected Output
After running the preprocessing cell, the notebook should:
- Process all images in Dataset/
- Save generated outputs into Dataset_preprocessed/ for all three scales
- Print per-class progress and total processed image count

## Notes
- EDA is currently focused on raw-data analysis and split sanity checks.
- The generated class weights can be used in training loss functions to handle class imbalance.

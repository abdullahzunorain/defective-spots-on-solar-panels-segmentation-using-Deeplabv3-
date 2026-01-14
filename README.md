
---

# Defective Spots on Solar Panels Segmentation Using DeepLabV3+

**Author:** Abdullah Zunorain
**Date:** January 2026
**Project Type:** Semantic Segmentation / Image Processing
**GitHub Repository:** [defective-spots-on-solar-panels-segmentation-using-Deeplabv3-](https://github.com/abdullahzunorain/defective-spots-on-solar-panels-segmentation-using-Deeplabv3-)

---

## 1. Project Overview

This project focuses on **detecting defective spots (hotspots) on solar panels** using drone imagery. The pipeline is based on **DeepLabV3+ with a ResNet101 backbone**, with an optional **Conditional Random Field (CRF) post-processing** step to refine segmentation results.

The main objective is to **compare the performance of DeepLabV3+ with and without CRF** to determine whether CRF improves segmentation accuracy and quality.

**Key Goals:**

* Develop a clean, reproducible pipeline for semantic segmentation.
* Compare baseline DeepLabV3+ predictions with CRF-refined outputs.
* Evaluate results using standard metrics and visualizations.

---

## 2. Dataset

* **Source:** Drone images of solar panels
* **Training set:** 518 images + corresponding masks
* **Validation set:** 130 images + corresponding masks
* **Classes:**

  * `0` → Background
  * `1` → Defective spot

**Notes:**

* Masks indicate the defective areas to be segmented.
* The dataset is slightly imbalanced; the pipeline handles this using weighted loss functions or oversampling techniques.

> **Important:** The raw dataset is **not included** in the repository. Please refer to your local dataset folder `processed_dataset_ver_04/`.

---

## 3. Methodology

### 3.1 Data Preprocessing

* Images and masks resized and normalized.
* Data augmentation applied (rotation, flipping, brightness/contrast).
* Masks converted to tensors suitable for training.

### 3.2 Model Architecture

* **DeepLabV3+** with **ResNet101 backbone**.
* Pretrained weights used for faster convergence.
* Weighted Cross-Entropy Loss for imbalanced classes.
* Optimizer: Adam or SGD with learning rate scheduler.

### 3.3 Post-Processing (Optional)

* **Conditional Random Fields (CRF)** applied to DeepLabV3+ predictions.
* Hyperparameters tuned to refine mask boundaries and improve segmentation quality.

### 3.4 Evaluation

* Metrics: Accuracy, Precision, Recall, F1-score, mIoU.
* Visual comparison: Ground Truth | DeepLabV3+ | DeepLabV3+ + CRF.
* Training monitoring: Loss curves and learning curves.
* ROC curve generation for further validation.

---

## 4. Project Structure

```
project-root/
├── final/
│   ├── eda_outputs/                  # Exploratory Data Analysis outputs (CSV, JSON, plots)
│   ├── model_training_results/       # Training outputs, metrics, visualizations
│   ├── preparing dataset.ipynb       # Data preprocessing notebook
│   ├── semantic-segmentation-deeplabv3-plus.ipynb  # Training + Evaluation notebook
├── .gitignore                         # Git ignore file for large data/weights
├── solar pannels semantic segmentation Deeplab3+.txt  # Project notes
└── README.md
```

> **Note:** The dataset folder `processed_dataset_ver_04/` is excluded from Git to avoid uploading large files.

---

## 5. Results

* CRF post-processing improves segmentation smoothness and accuracy.
* Side-by-side visualizations clearly show differences between baseline and refined predictions.
* Quantitative metrics indicate F1-score and mIoU improvements with CRF.

---

## 6. Usage Instructions

1. **Clone the repository:**

```bash
git clone git@github.com:abdullahzunorain/defective-spots-on-solar-panels-segmentation-using-Deeplabv3-.git
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Run Notebooks:**

   * `preparing dataset.ipynb` → Data preprocessing and augmentation
   * `semantic-segmentation-deeplabv3-plus.ipynb` → Model training, CRF post-processing, evaluation, and visualizations

4. **Outputs:**

   * Metrics CSV files
   * Comparison visualizations
   * Loss and learning curves

---


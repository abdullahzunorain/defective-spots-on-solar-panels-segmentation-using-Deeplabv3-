
# Defective Spots on Solar Panels Segmentation Using DeepLabV3+

**Author:** Abdullah Zunorain

**Date:** January 2026

**Project Type:** Semantic Segmentation / Computer Vision

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Motivation](#motivation)
3. [Dataset](#dataset)
4. [Methodology](#methodology)

   * [Data Preprocessing](#data-preprocessing)
   * [Model Architecture](#model-architecture)
   * [Post-Processing](#post-processing-optional)
   * [Evaluation Metrics](#evaluation-metrics)
5. [Project Structure](#project-structure)
6. [Results](#results)
7. [Usage Instructions](#usage-instructions)
8. [References](#references)
9. [Contact](#contact)

---

## Project Overview

This project focuses on **detecting defective spots (hotspots) on solar panels** using high-resolution drone imagery. The solution leverages **DeepLabV3+** with a **ResNet101 backbone** for semantic segmentation. To refine segmentation boundaries, an optional **Conditional Random Field (CRF)** post-processing step is applied.

The core objective is to develop a reproducible, robust pipeline and compare the segmentation performance with and without CRF, using **quantitative metrics** and **visual analysis**.

**Key Goals:**

* Build a clean and reproducible semantic segmentation pipeline.
* Evaluate and compare DeepLabV3+ baseline predictions with CRF-enhanced outputs.
* Analyze segmentation quality through metrics, loss curves, and visualizations.

---

## Motivation

Solar panels are prone to defects like hotspots, which reduce efficiency and lifespan. Manual inspection is **time-consuming, expensive, and error-prone**. Using **drone imagery** and **AI-based segmentation**, it is possible to **automate defect detection**, enabling:

* Faster inspection cycles.
* Precise localization of defective areas.
* Objective, repeatable evaluation for maintenance planning.

---

## Dataset

* **Source:** Drone-captured images of solar panels
* **Training set:** 518 images + masks
* **Validation set:** 130 images + masks

**Classes:**

| Class | Label          | Description            |
| ----- | -------------- | ---------------------- |
| 0     | Background     | Non-defective areas    |
| 1     | Defective spot | Hotspot/defect regions |

**Notes:**

* Masks highlight defective areas.
* Dataset may be slightly imbalanced; handled using **weighted loss** or **oversampling**.
* The raw dataset is **excluded** from the repository due to size constraints. Use local folder `processed_dataset_ver_04/`.

---

## Methodology

### Data Preprocessing

* **Resizing and normalization:** Images and masks converted to uniform dimensions and normalized for model input.
* **Augmentation:**

  * Random rotations and flips
  * Brightness and contrast adjustments
* **Mask conversion:** Segmentation masks converted to tensor format suitable for PyTorch training.

### Model Architecture

* **Model:** DeepLabV3+
* **Backbone:** ResNet101 (pretrained weights for faster convergence)
* **Loss Function:** Weighted Cross-Entropy to address class imbalance
* **Optimizers:** Adam or SGD with learning rate scheduler

### Post-Processing (Optional)

* **Conditional Random Fields (CRF)** applied to DeepLabV3+ predictions to refine boundaries.
* **Hyperparameter tuning** ensures better mask smoothness and segmentation accuracy.

### Evaluation Metrics

* **Pixel-level metrics:** Accuracy, Precision, Recall, F1-score, mIoU
* **Visual assessment:** Side-by-side comparison:

  * Ground Truth | DeepLabV3+ | DeepLabV3+ + CRF
* **Training monitoring:** Loss and learning curves
* **ROC/AUC analysis:** For further validation of defective class detection

---

## Project Structure

```
project-root/
├── final/
│   ├── eda_outputs/                     # Exploratory Data Analysis outputs (plots, CSV, JSON)
│   ├── model_training_results/          # Training outputs, metrics, visualizations
│   ├── preparing dataset.ipynb          # Notebook for preprocessing and augmentation
│   └── semantic-segmentation-deeplabv3-plus.ipynb  # Training, evaluation, CRF, visualization
├── .gitignore                            # Ignore large data, dataset, model checkpoints
├── solar pannels semantic segmentation Deeplab3+.txt  # Notes and references
└── README.md                             # This file
```

> **Note:** The dataset folder `processed_dataset_ver_04/` is **excluded** from Git to avoid large file uploads.

---

## Results

* **CRF post-processing** improves segmentation boundary accuracy and overall mask quality.
* **Visual results:** Side-by-side comparisons clearly show improvement from baseline to CRF-refined masks.
* **Quantitative improvements:** Slight but consistent increase in F1-score and mIoU.

**Example Metrics Comparison:**

| Metric            | Baseline | CRF    |
| ----------------- | -------- | ------ |
| Accuracy          | 0.9832   | 0.9834 |
| Precision (macro) | 0.9014   | 0.9023 |
| Recall (macro)    | 0.9903   | 0.9903 |
| F1-score (macro)  | 0.9405   | 0.9411 |
| mIoU              | 0.8919   | 0.8929 |

**Class-wise IoU:**

| Class | Baseline | CRF    |
| ----- | -------- | ------ |
| 0     | 0.9820   | 0.9822 |
| 1     | 0.8019   | 0.8037 |

---

## Usage Instructions

### 1. Clone Repository

```bash
git clone git@github.com:abdullahzunorain/defective-spots-on-solar-panels-segmentation-using-Deeplabv3-.git
cd defective-spots-on-solar-panels-segmentation-using-Deeplabv3-
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Key libraries:** PyTorch, torchvision, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, tqdm, pydensecrf

### 3. Run Notebooks

* `preparing dataset.ipynb` → Data preprocessing and augmentation
* `semantic-segmentation-deeplabv3-plus.ipynb` → Model training, CRF post-processing, evaluation, visualization

### 4. Outputs

* CSV metrics (`per_image_metrics.csv`, `metrics_summary.csv`)
* Random comparison visualizations (`comparison_samples_random.png`)
* Loss and learning curves (`loss_curve.png`)
* ROC/AUC plots (`roc_auc.png`)
* Model checkpoints (`best_deeplab_resnet101.pth`)

---

## Contact

**Abdullah Zunorain**
Email: [abdullahzunorain2@gmail.com](mailto:abdullahzunorain2@gmail.com)
GitHub: [https://github.com/abdullahzunorain](https://github.com/abdullahzunorain)

---



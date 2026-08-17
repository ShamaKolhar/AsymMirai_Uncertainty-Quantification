# AsymMirai Uncertainty Quantification

This project extends the [AsymMirai](https://github.com/jdonnelly36/AsymMirai) framework for **breast cancer risk prediction from mammograms** by adding **Uncertainty Quantification (UQ)** during inference.

AsymMirai is an interpretable extension of the MIT Mirai framework that uses bilateral mammographic asymmetry across four standard views:

* L-CC — Left Cranio-Caudal
* R-CC — Right Cranio-Caudal
* L-MLO — Left Mediolateral Oblique
* R-MLO — Right Mediolateral Oblique

The original model uses bilateral differences between mammographic views to identify regions of asymmetry and predict breast cancer risk.

This project extends the inference pipeline with:

* **Monte Carlo Dropout (MC Dropout)** — to estimate epistemic/model uncertainty
* **Test-Time Augmentation (TTA)** — to assess prediction sensitivity to input perturbations
* **Prediction variance analysis** — to identify potentially ambiguous cases
* **Uncertainty visualisation** — to analyse prediction confidence and variability

The overall goal is to investigate whether uncertainty estimation can make an interpretable mammography model more **trust-aware**, particularly by identifying cases that may require additional human review.

---

## Quick Start

The recommended way to run this project is through **Google Colab**.

Open:

```text
run_quantification.ipynb
```

and select:

**Runtime → Run all**

> **Important:** The notebook expects a specific Google Drive folder structure. To run without modifying the notebook paths, reproduce the folder structure described below.

---

## Project Structure

The project expects the following structure inside Google Drive:

```text
My Drive/
└── dissertation/
    │
    ├── AsymMirai-master/
    │   └── AsymMirai-master/
    │       └── run_quantification.ipynb
    │
    ├── datathon_tables/
    │   └── ...
    │
    ├── pngimages/
    │   └── ...
    │
    ├── training_preds/
    │   ├── validation_mc_dropout_predictions.csv
    │   ├── Basic_Statistics.csv
    │   ├── hist_uncertainty.png
    │   ├── confidence_vs_uncertainty.png
    │   └── ambiguous_barplot.png
    │
    └── images/
        └── ...
```

### Existing output files

The CSV/PNG output files included with the project were originally located under:

```text
AsymMirai_UQ/training_preds/training_preds/
```

Move these files into:

```text
MyDrive/dissertation/training_preds/
```

so that the notebook can find them using its existing paths.

---

# Running the Project in Google Colab

### 1. Upload the project folders

Upload the required folders to:

```text
My Drive/dissertation/
```

while maintaining the directory structure described above.

### 2. Open the notebook

In Google Drive, navigate to:

```text
MyDrive/dissertation/AsymMirai-master/AsymMirai-master/run_quantification.ipynb
```

Right-click the notebook and select:

**Open with → Google Colaboratory**

### 3. Run the notebook

In Google Colab, select:

**Runtime → Run all**

The notebook will:

1. Mount Google Drive
2. Load the required metadata from `datathon_tables/`
3. Load the AsymMirai model
4. Prepare the mammography inputs
5. Run Monte Carlo Dropout
6. Run Test-Time Augmentation
7. Calculate prediction statistics and uncertainty
8. Identify potentially ambiguous predictions
9. Generate uncertainty visualisations
10. Save the resulting files under:

```text
MyDrive/dissertation/training_preds/
```

---

# Uncertainty Quantification

## Monte Carlo Dropout

Monte Carlo Dropout is used to estimate **epistemic uncertainty**, representing uncertainty associated with the model.

Instead of producing a single deterministic prediction, the model performs multiple stochastic forward passes with dropout enabled during inference.

The resulting predictions are used to calculate statistics such as:

* Mean prediction
* Prediction variance
* Prediction standard deviation

Higher variation across stochastic predictions indicates greater model uncertainty.

In this implementation, dropout was incorporated into the ResNet-18 backbone used by AsymMirai without retraining the original model.

---

## Test-Time Augmentation

Test-Time Augmentation (TTA) is used to investigate **input-level uncertainty** by applying controlled perturbations to mammography images during inference.

The implemented augmentations include:

* Horizontal flipping
* Small-angle rotations
* Gaussian blur
* Brightness/contrast variation
* Noise perturbations

The model produces predictions for the augmented inputs, and the variation between these predictions is used as a measure of prediction sensitivity.

MC Dropout and TTA can also be analysed together to provide a broader view of predictive uncertainty.

---

# DICOM Images

The project uses mammography images from the **EMory BrEast Imaging Dataset (EMBED)**.

The preprocessing pipeline works with DICOM images and extracts the required metadata and image information before converting suitable images into PNG format for model inference.

The original dataset contains a large number of DICOM images. The preprocessing pipeline filters the data to retain complete four-view mammography exams:

```text
L-CC
R-CC
L-MLO
R-MLO
```

The project initially processed **145,482 DICOM images** and produced **31,686 PNG images** after filtering and preprocessing.

The final uncertainty analysis was performed on **1,422 complete mammography exams**.

---

## EMBED Dataset Access

The EMBED dataset is not included in this repository.

Information about the dataset and its access process can be found here:

[EMBED Open Data](https://github.com/Emory-HITI/EMBED_Open_Data)

The original implementation uses Amazon S3 for accessing the DICOM data.

### S3 configuration

The DICOM access code is already included in:

```text
run_quantification.ipynb
```

If you need to access the S3 data, configure your AWS credentials through a secure method such as environment variables or Google Colab Secrets.

**Do not commit AWS access keys or secret keys to the repository.**

The relevant configuration follows this general structure:

```python
import os
import boto3

s3 = boto3.client("s3")

bucket_name = "embed-dataset-open"
prefix = "images/cohort_1/"
drive_root = "/content/drive/MyDrive/dissertation/images"
```

The notebook then uses the S3 paginator to locate and download the required DICOM files.

---

# Data Preprocessing

The preprocessing pipeline performs the following steps:

```text
DICOM Images
     │
     ▼
Metadata Extraction
     │
     ▼
View & Laterality Matching
     │
     ▼
Complete 4-View Exam Filtering
     │
     ▼
DICOM → PNG Conversion
     │
     ▼
Metadata Matching
     │
     ▼
AsymMirai Inference
     │
     ▼
Uncertainty Quantification
```

DICOM files are processed using `pydicom`, while OpenCV is used for image processing and augmentation.

Metadata is matched using information including:

* Exam ID
* View type
* Laterality

Only exams containing all four standard views are retained for bilateral asymmetry analysis.

---

# Outputs

The uncertainty quantification pipeline generates the following outputs:

### Prediction results

```text
training_preds/
└── validation_mc_dropout_predictions.csv
```

Contains prediction-level information and uncertainty measurements.

### Summary statistics

```text
training_preds/
└── Basic_Statistics.csv
```

Contains summary statistics for predictions and uncertainty.

### Visualisations

```text
training_preds/
├── hist_uncertainty.png
├── confidence_vs_uncertainty.png
└── ambiguous_barplot.png
```

These visualisations are used to analyse:

* Distribution of uncertainty
* Prediction confidence versus uncertainty
* Ambiguous versus non-ambiguous predictions

---

# Results

The uncertainty-aware AsymMirai pipeline was evaluated on **1,422 mammography exams**.

The analysis focused on uncertainty and prediction stability rather than retraining the underlying AsymMirai model.

Using the predefined uncertainty threshold:

```text
Total evaluated exams:       1,422
Ambiguous predictions:          72
Ambiguity rate:               5.06%
```

The flagged cases represent predictions exhibiting higher estimated uncertainty and provide a potential mechanism for **selective review of ambiguous cases**.

The purpose of this analysis is not to claim clinical diagnostic performance, but to investigate whether uncertainty estimates can identify cases where the model's prediction is less stable.

---

# Technologies

| Category         | Technology                     |
| ---------------- | ------------------------------ |
| Programming      | Python                         |
| Deep Learning    | PyTorch                        |
| CNN Backbone     | ResNet-18                      |
| Medical Imaging  | pydicom                        |
| Image Processing | OpenCV                         |
| Data Processing  | Pandas, NumPy                  |
| Visualisation    | Matplotlib, Seaborn            |
| UQ               | Monte Carlo Dropout, TTA       |
| Compute          | Google Colab / NVIDIA Tesla T4 |
| Storage          | Google Drive / Amazon S3       |

The project uses PyTorch for model execution and modification, Torchvision for image transformations and pretrained components, pydicom for DICOM handling, Pandas/NumPy for data processing, and Matplotlib/Seaborn/OpenCV for analysis and visualisation.

---

# Additional Files

The repository/archive may also contain:

```text
EMBED_Open_Data-main/
```

This contains reference notebooks and documentation related to the EMBED dataset.

It is **not required for the uncertainty quantification pipeline itself**, but can be useful for understanding the dataset and its organisation.

A top-level:

```text
requirements.txt
```

is also provided for local execution.

For Google Colab, the notebook installs the required dependencies through its setup cells.

---

# Local Execution

The project was primarily developed and tested using Google Colab.

For local execution:

```bash
pip install -r requirements.txt
```

You will also need to update the Google Drive paths used in the notebook.

For example, paths such as:

```text
/content/drive/MyDrive/dissertation/...
```

must be replaced with paths corresponding to your local environment.

The same directory structure should be maintained wherever possible.

---

# Research Context

This project was developed as an extension of AsymMirai to investigate the integration of uncertainty estimation into an interpretable mammography risk prediction pipeline.

The baseline AsymMirai model uses bilateral dissimilarity between left and right mammographic views and provides localized visual explanations through prediction windows and asymmetry maps.

This project adds uncertainty estimation on top of that existing inference pipeline rather than replacing the underlying AsymMirai approach.

The broader objective is to explore **trustworthy AI for medical imaging**, where a model can provide not only a prediction but also an indication of how stable or uncertain that prediction may be.

---

# Future Work

Potential extensions include:

* Per-view uncertainty estimation for L-CC, R-CC, L-MLO and R-MLO
* Comparison with deep ensembles and Bayesian approaches
* Calibration of uncertainty estimates
* Testing under image corruption and distribution shifts
* Out-of-distribution evaluation
* Human-in-the-loop evaluation with radiologists
* Integration of clinical metadata
* Development of a clinical decision-support prototype

These extensions would help determine whether computational uncertainty estimates correspond to clinically meaningful case difficulty.

---

# Disclaimer

This repository is intended for **research and educational purposes only**.

The predictions and uncertainty estimates produced by this project have not been validated for clinical diagnosis or patient care and should not be used as a substitute for professional medical judgement.

---

# References

### AsymMirai

Donnelly et al.

[AsymMirai GitHub Repository](https://github.com/jdonnelly36/AsymMirai)

### EMBED

[EMBED Open Data](https://github.com/Emory-HITI/EMBED_Open_Data)

### Monte Carlo Dropout

Gal, Y. & Ghahramani, Z.
*Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.*

### Mirai

Yala et al.
*A deep learning model to assess cancer risk from mammograms.*

---

## Author

**Shama Kolhar**

MSc Data Science
University of Exeter

[GitHub](https://github.com/ShamaKolhar)

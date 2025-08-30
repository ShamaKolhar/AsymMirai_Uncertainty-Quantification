This project builds upon the AsymMirai [https://github.com/jdonnelly36/AsymMirai] framework for breast cancer risk prediction using mammographic asymmetry.
Originally developed as an extension of MIT Mirai model, AsymMirai leverages bilateral views (L-CC, R-CC, L-MLO, R-MLO) to compute asymmetry and predict breast cancer risk in the next 1–5 years.
This fork adds Uncertainty Quantification (UQ) capability using:
•	Monte Carlo Dropout (MC-Dropout)
•	Test-Time Augmentation (TTA)
The purpose of this extension is to assess model trustworthiness, detect ambiguous cases, and visualize epistemic uncertainty.

Quick start: Open `run_quantification.ipynb` in Google Colab and **Run all**.  
> To run without editing any code, place the folders exactly as shown below.

## Folder placement (Google Drive)

Place these at **My Drive / dissertation** in Google Drive

File Structure :

MyDrive/
└─ dissertation/
├─ AsymMirai-master/
│ └─ AsymMirai-master/
│ ├─ run_quantification.ipynb <- open this in Colab
│ ├─ asymmetry_model/
│ ├─ configs/
│ ├─ onconet/
│ └─ ...
├─ datathon_tables/
│ ├─ EMBED_metadata_with_dicom_paths.csv
│ ├─ dicom_dataset.csv
│ ├─ EMBED_OpenData_NUS_Datathon_metadata.csv
│ ├─ EMBED_OpenData_NUS_Datathon_metadata_reduced.csv
│ ├─ EMBED_OpenData_NUS_Datathon_clinical.csv
│ └─ EMBED_OpenData_NUS_Datathon_clinical_reduced.csv
├─ training_preds/
│ ├─ validation_mc_dropout_predictions.csv
│ ├─ temp_mc_dropout_predictions.csv
│ ├─ Basic_Statistics.csv
│ ├─ hist_uncertainty.png
│ ├─ confidence_vs_uncertainty.png
│ └─ ambiguous_barplot.png
└─ (optional) pngimages/  #converted DICOM→PNG

**Note:** The CSV/PNG output files bundled in your archive were under  
> `AsymMirai_UQ/training_preds/training_preds/`. Move those files into  
> `MyDrive/dissertation/training_preds/` (exact path) so the notebook finds them automatically.

The archive also includes:
- `EMBED_Open_Data-main/...` (reference notebooks/docs; not required to run UQ)
- Top-level `requirements.txt` (only needed for local runs; Colab cells install deps automatically)

## Run (Colab)

1. Upload the folders as shown above to **My Drive / dissertation**.
2. In Google Drive, right-click  
   `MyDrive/dissertation/AsymMirai-master/AsymMirai-master/run_quantification.ipynb` → **Open with → Colab**.
3. In Colab: **Runtime → Run all**.
   - The notebook:
     - Mounts Google Drive
     - Loads metadata from `datathon_tables/`
     - Uses/creates `pngimages/` if you perform DICOM→PNG
     - Runs MC Dropout & TTA
     - Saves outputs to `MyDrive/dissertation/training_preds/`

# DICOM images

This code is already in the run_quantification.ipynb :

#Access DICOM images from Amazon s3 bucket
import os
import boto3
os.environ['AWS_ACCESS_KEY_ID'] = ''
os.environ['AWS_SECRET_ACCESS_KEY'] = ''
os.environ['AWS_DEFAULT_REGION'] = 'us-west-2'

s3 = boto3.client('s3')
bucket_name = 'embed-dataset-open'
prefix = 'images/cohort_1/'
drive_root = '/content/drive/MyDrive/dissertation/images'


paginator = s3.get_paginator('list_objects_v2')

for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
    for obj in page.get('Contents', []):
        s3_key = obj['Key']

        if s3_key.endswith('/'):
            continue

        # Strip prefix from key to build relative local path
        relative_path = s3_key[len(prefix):]
        local_path = os.path.join(drive_root, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        print(f"Downloading: {s3_key} → {local_path}")
        s3.download_file(bucket_name, s3_key, local_path)

**Downloads DICOM images from the Amazon S3 Bucket**

Information on dataset & accessibility to dataset an also be found here: https://github.com/Emory-HITI/EMBED_Open_Data

## Outputs

- `training_preds/validation_mc_dropout_predictions.csv` (per-image predictions + UQ)
- Plots: `hist_uncertainty.png`, `confidence_vs_uncertainty.png`, `ambiguous_barplot.png`
- `Basic_Statistics.csv` (summary stats)

Mirror the same folder structure under a local path and update any /content/drive/MyDrive/dissertation/... paths in the first cells of the notebook.

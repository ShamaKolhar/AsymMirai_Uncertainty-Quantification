# Trust-Aware AsymMirai: Uncertainty Quantification for Mammography Risk Prediction

MSc Data Science dissertation project, University of Exeter.

Deep-learning breast cancer risk models output a single number, with no indication of when that number shouldn't be trusted. This project adds **uncertainty quantification** to [AsymMirai](https://github.com/jdonnelly36/AsymMirai), an interpretable 1–5-year breast cancer risk model that compares the left and right breast. The aim is for the model to report *how sure it is*, and to flag the exams it is least sure about for radiologist review.

> Research prototype. Not a medical device and not for clinical use.

### At a glance

| | |
|---|---|
| **Problem** | A risk model that is confidently wrong is dangerous. Clinicians need to know *which* predictions to double-check. |
| **What I built** | An uncertainty layer for AsymMirai (MC Dropout, paired test-time augmentation, deep ensembles, conformal prediction) plus a full evaluation pipeline, CLI and Streamlit demo |
| **Key finding** | My dissertation (v1) showed MC Dropout on the released model gave near-zero variance and near-identical predictions for all 1,422 validation exams. I traced the cause and redesigned the method in v2 ([details below](#how-this-project-evolved-the-dissertation-finding)). |
| **Data** | EMBED (Emory Breast Imaging Dataset): 145K+ DICOM mammograms pulled from Amazon S3, curated to complete four-view screening exams |
| **Stack** | Python · PyTorch · torchvision · pydicom · OpenCV · pandas · scikit-learn · lifelines · Streamlit · pytest · Google Colab (T4) |
| **Status** | Pipeline complete and tested on synthetic data. Real-data evaluation on EMBED in progress. |

---

## Approach

```mermaid
flowchart LR
    A[EMBED DICOMs] --> B[16-bit PNGs<br/>+ exam table<br/>+ cancer labels]
    B --> C[AsymMirai<br/>Mirai encoder → L/R asymmetry → risk]
    C --> D1[MC Dropout]
    C --> D2[Paired TTA]
    C --> D3[MC Dropout + TTA]
    C --> D4[Deep ensemble]
    D1 & D2 & D3 & D4 --> E[Risk + uncertainty<br/>per exam]
    E --> F[Evaluation<br/>AUC · calibration · risk–coverage<br/>conformal · robustness · fairness]
```

**Uncertainty methods**
- **MC Dropout**: stochastic forward passes with dropout on the image embeddings.
- **Paired test-time augmentation**: small rotations, shifts and contrast changes.
- **MC Dropout + TTA**: both combined.
- **Deep ensemble**: several dropout-trained heads with different seeds and bootstrap samples.
- **Conformal prediction**: class-conditional sets with a coverage guarantee. A set containing both labels means "refer".
- **Localisation uncertainty**: how often the model's "most asymmetric region" moves between stochastic passes.

## How this project evolved: the dissertation finding

**v1 (MSc dissertation, 2025).** I built a pipeline that pulled 145,482 DICOM mammograms from Amazon S3 and kept only complete four-view exams (31,686 images). I then added MC Dropout and test-time augmentation to the released AsymMirai model **without retraining it**, and ran it on 1,422 validation exams.

![v1 diagnostic](docs/figures/v1_diagnostic.png)

**What the results showed:**
- Every exam received almost the same predicted probability (0.005160 ± 0.00000001).
- MC Dropout variance was between 1e-16 and 1e-14, which is numerical noise. The 72 exams above the v1 "ambiguous" threshold were not genuinely uncertain.
- **Diagnosis:** the released weights were trained without the dropout I switched on at inference, so the stochastic passes barely changed the output. MC Dropout only approximates Bayesian uncertainty if the model was *trained* with that dropout. The v1 inference script also flipped each view independently as its augmentation, and for a model whose score is a left/right difference, that adds artificial asymmetry.

**v2 (this version)** is a redesign based on those lessons: dropout-trained heads, paired dropout and paired augmentation for the left and right breast, norm-preserving dropout scaling, a patient-level calibration/test split, and an evaluation that asks whether the uncertainty is actually *useful* (risk–coverage, misclassification detection, robustness, fairness). The v1 code is preserved in git history at commit [`4f868aa`](https://github.com/ShamaKolhar/AsymMirai_Uncertainty-Quantification/tree/4f868aa/legacy).

## Key design decisions

| Decision | Why |
|---|---|
| **Paired dropout and TTA**: the same dropout mask and the same augmentation for the left and right breast | AsymMirai's score *is* the left/right difference. Independent noise on each side would create artificial asymmetry, so the "uncertainty" would be an artefact. |
| **Norm-preserving dropout scaling** (1/√(1−p) instead of 1/(1−p)) | Keeps the expected squared L2 asymmetry unchanged, so MC-averaged risks are not biased upwards. |
| **Dropout only in the head; backbone features cached** | The backbone runs once per exam and the K dropout samples reuse its output: about K× cheaper. |
| **Head re-fitted with dropout on** (backbone frozen) | MC Dropout approximates Bayesian uncertainty only if the model was trained with that dropout. |
| **Exam = accession number; 2D CC/MLO screening views only** | A patient has many exams over the years, and synthetic C-view and spot/mag images are not standard screening views. |
| **Patient-level calib/test split** | Referral thresholds and conformal quantiles are fitted on `calib` and reported on `test`, with no patient in both. |
| **Class-conditional (Mondrian) conformal** | With ~1–2% cancer prevalence, marginal 90% coverage could be met while missing most cancers. |
| **Re-implemented, device-agnostic forward pass** | Allows MC sampling on cached features and runs on CPU or GPU. Unit-tested equal to the original AsymMirai model to 1e-6. |

## Evaluation

Everything is reported on the held-out `test` split:

- **Discrimination**: 1–5-year AUC with stratified-bootstrap 95% CIs, plus Harrell's C-index
- **Calibration**: ECE (equal-frequency bins), Brier score, reliability diagrams
- **Is the uncertainty useful?**
  - Risk–coverage curves and AURC: does performance improve as the most uncertain exams are referred?
  - AUROC of uncertainty for detecting misclassifications
- **Robustness**: does uncertainty rise as Gaussian noise, blur or salt-and-pepper corruption gets more severe?
- **Fairness**: AUC, calibration and referral rate by race, breast density and age band
- **Conformal prediction**: empirical coverage overall and for cancer cases, and the fraction of exams referred

`python -m uq.evaluate` writes `report.md`, `metrics.json` and 11 figures.

## Status

- ✅ **Pipeline complete and tested.** 9 tests pass on CPU, including a full end-to-end run on synthetic 16-bit DICOMs: conversion → labels → head fine-tuning → all four UQ methods → evaluation and figures.
- ⏳ **Real-data evaluation on EMBED in progress.** Results will be added here.

---

## Quick start (Google Colab)

1. Clone this repo into Google Drive, e.g. `MyDrive/dissertation/AsymMirai_Uncertainty-Quantification`.
2. Download the **trained AsymMirai weights** (link in [`AsymMirai/README.md`](AsymMirai/README.md)) to `MyDrive/dissertation/weights/asymmirai_full_model.pt`.
3. Put the EMBED tables (`EMBED_OpenData_NUS_Datathon_clinical_reduced.csv`, `EMBED_OpenData_NUS_Datathon_metadata_reduced.csv`) in `MyDrive/dissertation/data/tables/`. EMBED is available on request from [Emory HITI](https://github.com/Emory-HITI/EMBED_Open_Data) and is **not** redistributed here.
4. Add `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` as **Colab Secrets**.
5. Open [`notebooks/run_quantification.ipynb`](notebooks/run_quantification.ipynb), select a GPU runtime, and run the cells in order. Run the 20-exam sanity check first.

## Command line

```bash
pip install -r requirements.txt
export ASYMMIRAI_UQ_ROOT=/path/to/folder/with/data_weights_outputs   # optional
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...                # never commit these

python -m uq.data.download_embed --max-files 400      # 1. DICOMs from S3 (resumable)
python -m uq.data.dicom_to_png --workers 8            # 2. 16-bit PNGs + index
python -m uq.data.build_metadata                      # 3. exam table, labels, patient split
python -m uq.finetune_head --seeds 0 1 2 3 4          # 4. optional: dropout-trained heads / ensemble
python -m uq.inference --limit 20 --run-name sanity   # 5a. sanity check: predictions must vary
python -m uq.inference --run-name full                # 5b. all exams, all conditions
python -m uq.evaluate --run outputs/full              # 6. metrics, figures, report.md
streamlit run app/streamlit_app.py                    # 7. demo
pytest tests/                                         # tests (synthetic data, CPU)
```

All settings are in [`configs/uq_config.yaml`](configs/uq_config.yaml): paths, sample counts, dropout, corruption conditions, horizons and conformal α.

## Repository layout

```
├── uq/                        # this project's code
│   ├── data/
│   │   ├── download_embed.py     # S3 download (credentials from env / Colab Secrets)
│   │   ├── dicom_to_png.py       # 16-bit PNG conversion, MONOCHROME1 handling
│   │   └── build_metadata.py     # exam grouping, labels, patient-level split
│   ├── preprocessing.py       # 16-bit loading, corruptions, paired TTA
│   ├── dataset.py             # 4-view exam dataset
│   ├── models/
│   │   ├── uq_asymmirai.py    # UQ-ready AsymMirai: paired dropout, Mirai/ResNet/ConvNeXt backbones
│   │   └── loading.py         # loads the released AsymMirai weights
│   ├── inference.py           # MC Dropout / TTA / MC+TTA / ensemble × clean + corrupted inputs
│   ├── finetune_head.py       # dropout-trained heads, ensemble members, new-backbone heads
│   ├── metrics.py             # AUC (CI), C-index, ECE, Brier, risk–coverage, subgroups
│   ├── conformal.py           # class-conditional split conformal prediction
│   ├── evaluate.py            # report, metrics, figures
│   └── plotting.py            # figure style
├── AsymMirai/                 # third-party: original AsymMirai code (Donnelly et al., MIT), vendored unmodified
├── configs/uq_config.yaml
├── notebooks/run_quantification.ipynb
├── app/streamlit_app.py       # demo: per-exam risk, uncertainty, referral flag
└── tests/                     # unit tests + synthetic end-to-end test
```

`AsymMirai/` is vendored unchanged because the released weights are a pickled model that needs the original class definitions to load. All of this project's code is in `uq/`, `app/`, `tests/` and `notebooks/`.

## Extensions

- **ConvNeXt-Tiny backbone**: set `model.backbone: convnext_tiny` and fit its head with `uq.finetune_head`. The asymmetry statistics are initialised from data.
- **Streamlit demo**: browse a run's results, or run live inference on four uploaded views.

## Credits

- **AsymMirai**: Donnelly et al., *AsymMirai: Interpretable Mammography-based Deep Learning Model for 1–5-year Breast Cancer Risk Prediction*, Radiology 2024. [Code](https://github.com/jdonnelly36/AsymMirai) (MIT).
- **Mirai**: Yala et al., 2021.
- **EMBED**: Jeong et al., 2023.

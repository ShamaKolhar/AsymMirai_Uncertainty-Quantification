# Trust-Aware AsymMirai: Uncertainty Quantification for Mammography Risk Prediction

MSc Data Science dissertation project (University of Exeter). This project extends
**AsymMirai** (an interpretable breast-cancer risk model built on Mirai) with **uncertainty
quantification**, so the model can say *how sure it is* and flag exams for radiologist review.

Uncertainty methods:
- **MC Dropout** with *paired* dropout masks, so left and right breasts share the same mask and no fake asymmetry is created
- **Paired test-time augmentation** (the same small rotation, shift and contrast change for both breasts)
- **MC Dropout + TTA** combined
- **Deep ensembles** of fine-tuned heads
- **Conformal prediction** with class-conditional coverage guarantees
- **Localisation uncertainty**: how much the "most asymmetric region" moves across stochastic passes

> Research prototype. Not a medical device and not for clinical use.

---

## Quick start (Colab)

1. Copy this folder to `MyDrive/dissertation/AsymMirai_UQ`.
2. Download the **trained AsymMirai weights** (link in `AsymMirai/README.md`, "Our trained AsymMirai weights are available here") to `MyDrive/dissertation/weights/asymmirai_full_model.pt`.
3. Put the EMBED tables (`EMBED_OpenData_NUS_Datathon_clinical_reduced.csv`, `EMBED_OpenData_NUS_Datathon_metadata_reduced.csv`) in `MyDrive/dissertation/data/tables/`. Request access to EMBED at https://github.com/Emory-HITI/EMBED_Open_Data. The data is **not** redistributed with this repo.
4. Add `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` as **Colab Secrets**.
5. Open `notebooks/run_quantification.ipynb`, choose a GPU runtime, and run the cells in order.

## Command line (local / any GPU machine)

```bash
pip install -r requirements.txt
export ASYMMIRAI_UQ_ROOT=/path/to/folder/with/data_weights_outputs   # optional
export AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...                # never commit these

python -m uq.data.download_embed --max-files 400      # 1. DICOMs from S3 (resumable)
python -m uq.data.dicom_to_png --workers 8            # 2. 16-bit PNGs + index
python -m uq.data.build_metadata                      # 3. exam table, labels, patient split
python -m uq.finetune_head --seeds 0 1 2 3 4          # 4. optional: dropout-trained heads / ensemble
python -m uq.inference --limit 20 --run-name sanity   # 5a. sanity check (predictions must vary!)
python -m uq.inference --run-name full                # 5b. all exams, all conditions
python -m uq.evaluate --run outputs/full              # 6. metrics, figures, report.md
streamlit run app/streamlit_app.py                    # 7. demo
pytest tests/                                         # unit + end-to-end tests (synthetic data, CPU)
```

All settings live in `configs/uq_config.yaml`: paths, sample counts, dropout, corruption conditions, horizons and conformal α.

## Repository layout

```
AsymMirai_UQ/
├── uq/                      # the UQ extension (new code)
│   ├── config.py            # YAML config, env-var root, seeding, device
│   ├── data/
│   │   ├── download_embed.py   # S3 download, credentials from env / Colab Secrets
│   │   ├── dicom_to_png.py     # 16-bit PNG, MONOCHROME1, Mirai windowing
│   │   └── build_metadata.py   # exam grouping by accession, labels, patient split
│   ├── preprocessing.py     # 16-bit loading, corruptions, paired TTA
│   ├── dataset.py           # 4-view exam dataset
│   ├── models/
│   │   ├── uq_asymmirai.py  # device-agnostic AsymMirai + paired dropout + Mirai/ResNet/ConvNeXt backbones
│   │   └── loading.py       # loads the released pickled AsymMirai weights
│   ├── inference.py         # MC dropout / TTA / MC+TTA / ensemble x clean + corrupted conditions
│   ├── finetune_head.py     # head training with dropout, ensemble members, new-backbone heads
│   ├── metrics.py           # AUC (CI), C-index, ECE, Brier, risk-coverage, unc-error, subgroups
│   ├── conformal.py         # split + class-conditional conformal prediction
│   ├── evaluate.py          # report.md, metrics.json, figures
│   └── plotting.py          # consistent, accessible figure style
├── AsymMirai/               # ORIGINAL AsymMirai code, restored unmodified (needed to load its weights)
├── configs/uq_config.yaml
├── notebooks/run_quantification.ipynb
├── app/streamlit_app.py
├── tests/                   # 9 tests incl. a full synthetic end-to-end run
└── legacy/                  # v1 code (GitHub + Colab versions), kept for reference only (do not use)
```

---

## What changed in v2, and why

The v1 results did not measure uncertainty. Every exam received the same prediction,
P(cancer) = 0.005159 (std 2.6e-9 across 1,422 exams). That value is exactly
`sigmoid(-8,000,000 / 1,520,381)`, the model's fixed offset. The reported "uncertainties"
(~1e-15) were floating-point rounding noise. v2 fixes the causes and rebuilds the evaluation.

### Phase 1: correctness

| # | Problem in v1 | Fix in v2 | Where |
|---|---|---|---|
| 1 | Trained weights never loaded: `embedding_model=None` produced an ImageNet ResNet-18 with an identity stretch matrix and an 8,000,000 offset. The original `mirai_localized_dif_head.py` had been overwritten, so the released pickled weights couldn't even be unpickled. | Original AsymMirai code restored, and the released full model is loaded into a UQ-ready re-implementation. Tested equal to the original forward pass to 1e-6. | `uq/models/loading.py`, `AsymMirai/`, `tests/test_model.py` |
| 2 | 8-bit PNGs normalised with 16-bit statistics, so every pixel was about −0.65 (a flat grey image). | 16-bit PNGs; the loader refuses 8-bit files (optional 8-bit mode with recomputed statistics). | `uq/preprocessing.py` |
| 3 | DICOM→PNG: per-image 8-bit min-max scaling, MONOCHROME1 not handled. | 16-bit min-max window (Mirai's DCMTK convention, or real `dcmj2pnm` via `--use-dcmtk`); MONOCHROME1 inverted. | `uq/data/dicom_to_png.py` |
| 4 | `exam_id = PatientID`, so different exams were merged (L-CC from one year, R-CC from another); C-view and spot/mag views mixed in. | Grouped by accession number (`acc_anon`) via an EMBED metadata merge; only 2D CC/MLO screening views kept. | `uq/data/build_metadata.py` |
| 5 | No labels (every label = 0), so no AUC was possible. | `years_to_cancer` / `years_to_last_followup` from the clinical table (path_severity 0/1); post-diagnosis exams dropped. | `uq/data/build_metadata.py` |
| 6 | TTA flipped each view independently, creating artificial asymmetry. | Paired TTA: identical small affine + contrast change for both breasts; missing views stay missing. | `PairedTTA` |
| 7 | Gaussian noise applied to the whole "validation" run. | Clean is always evaluated; each corruption (Gaussian, blur, salt-and-pepper × severity) is a separate, seeded condition. | `uq/inference.py` |
| 8 | Dropout p=0.7 inside a pretrained backbone never trained with dropout; independent masks for left and right. | Paired, norm-preserving dropout on the embeddings (backbone dropout optional, off by default); head fine-tuning *with* dropout; deep ensembles as a comparison. | `uq/models/uq_asymmirai.py`, `uq/finetune_head.py` |

### Phase 2: meaningful UQ evaluation

| # | v2 |
|---|---|
| 9 | The 95th-percentile "ambiguous" rule (always 5% by construction) is replaced with risk–coverage curves, AURC, uncertainty-vs-error AUROC and Spearman, ECE, Brier, reliability diagrams, and uncertainty vs. corruption severity. Thresholds are fitted on `calib` and reported on `test` (patient-disjoint). |
| 10 | 1–5-year AUCs with stratified bootstrap 95% CIs, plus Harrell's C-index, comparable to the AsymMirai paper. |
| 11 | Subgroup analysis (race, BI-RADS density, age band): AUC, ECE, mean uncertainty, referral rate. |

### Phase 3: engineering

| # | v2 |
|---|---|
| 12 | No credentials in code: AWS keys are read from environment variables or Colab Secrets (`.env.example`). |
| 13 | EMBED data no longer shipped (`.gitignore`); access instructions instead. |
| 14 | Figures saved before closing (v1 called `show()` before `savefig()`, giving blank PNGs); consistent, colour-blind-checked style. |
| 15 | `train.py` (which only did inference) split into `inference.py`, `finetune_head.py` and `evaluate.py`; YAML config instead of hard-coded Drive paths; seeds everywhere; merge instead of an O(n²) `str.contains` loop; device-agnostic (CPU/GPU). |
| 16 | `requirements.txt` fixed (v1 pinned `torch==1.2.0` but used an API from torchvision 0.13+). |

### Phase 4: extensions

| # | v2 |
|---|---|
| 17 | **ConvNeXt-Tiny backbone** (`model.backbone: convnext_tiny`). Its head is fitted with `uq.finetune_head`, and the asymmetry statistics are initialised from data so the sigmoid isn't saturated. |
| 18 | **Localisation uncertainty**: the fraction of stochastic passes whose most-asymmetric window differs from the modal one, per exam, correlated with risk uncertainty. |
| 19 | **Conformal prediction**: class-conditional (Mondrian) split conformal. `{cancer, no cancer}` sets mean "refer". |
| 20 | **Streamlit demo**: browse results (risk, uncertainty, referral flag, figures) or run live on 4 uploaded views. |

## Verification status

- ✅ `pytest tests/` passes on CPU: model equivalence with the original AsymMirai, loading of pickled weights, paired dropout creating no fake asymmetry, missing views, ConvNeXt, and a full synthetic run (DICOM → PNG → labels → fine-tuning → all UQ methods → evaluation → non-blank figures).
- ⏳ **Not yet run on real EMBED images with the released weights.** That needs a GPU, the weights and the images. First run `--limit 20` and check that `run_info.json → conditions.clean.pred_std` is clearly above 0.

## Citation / credits

AsymMirai: Donnelly et al., *AsymMirai: Interpretable Mammography-based Deep Learning Model for 1–5-year Breast Cancer Risk Prediction*, Radiology 2024. Code: https://github.com/jdonnelly36/AsymMirai (MIT). Mirai: Yala et al., 2021. EMBED: Jeong et al., 2023.

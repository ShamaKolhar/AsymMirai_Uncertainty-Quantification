# Legacy (v1) code: reference only, do not use

v1 of this project had two variants. Both are kept so the dissertation can describe what changed in v2 and why.

## `v1_github/`: the version previously on this repo's `main` branch
- `dissertation.ipynb`: Colab preprocessing notebook
- `run_utils.py`, `run_model_in_merged_dataset.py`, `testing.py`: MC Dropout + TTA inference
- `mirai_localized_dif_head.py`, `train.py`: modified copies of the AsymMirai files
- `README_v1.md`: the previous README

Known problems:
- The Mirai backbone was replaced by `torch.nn.Identity()`, so no image features were extracted.
- Trained weights were never loaded.
- `torch.sigmoid` was applied to outputs that were already probabilities (a double sigmoid).
- `model.train()` put BatchNorm into training mode during inference.
- TTA was applied to each view independently, creating artificial left/right asymmetry.
- "Ambiguous" was a fixed std > 0.25 threshold, with no labels to evaluate it against.

## `v1_local/`: the Drive/Colab version used for the reported results
- `old_mc_dropout_inference.py` (was `asymmetry_model/train.py`): ran an untrained head on an ImageNet ResNet-18 and produced a constant prediction of 0.00516 for every exam.
- `old_mirai_localized_dif_head.py`: the modified model file that replaced the original AsymMirai class.

The main README's "What changed in v2" section lists every fix.

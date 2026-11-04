from asymmetry_metrics import hybrid_asymmetry
from mirai_localized_dif_head import LocalizedDifModel
from mirai_metadataset import MiraiMetadataset
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import partial 
import torchvision.transforms as transforms
import time
import os
import cv2

def restructure_mirai_metadata(df):
    # Combine 'laterality' and 'view' to form side-view labels (e.g. L-CC, R-MLO)
    df['side_view'] = df['laterality'] + '-' + df['view']

    # Check for duplicates and keep only one path per exam-side-view
    df = df.drop_duplicates(subset=['exam_id', 'side_view'])

    # Pivot table to restructure image paths as columns: L-CC, R-CC, etc.
    pivot = df.pivot(index='exam_id', columns='side_view', values='file_path').reset_index()

    # Extract metadata
    meta_cols = ['exam_id']
    if 'years_to_cancer' in df.columns:
        meta_cols.append('years_to_cancer')

    meta_df = df[meta_cols].drop_duplicates()
    
    # Merge pivoted data with metadata
    final_df = pd.merge(pivot, meta_df, on='exam_id', how='left')
    return final_df


def resize_and_normalize(img_path, use_crop=False, augment=False, noise_type=None):
    import numpy as np
    img_mean = 7699.5
    img_std = 11765.06
    target_size = (1664, 2048)

    # Load the grayscale image
    if isinstance(img_path, str) and os.path.exists(img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    else:
        raise ValueError(f"Invalid image path: {img_path}")

    # Noise before normalization
    if noise_type == "gaussian":
        noise = np.random.normal(loc=0, scale=20, size=img.shape)
        img = np.clip(img + noise, 0, 255)
    
    elif noise_type == "salt_pepper":
        prob = 0.01
        rnd = np.random.rand(*img.shape)
        img[rnd < prob / 2] = 0
        img[rnd > 1 - prob / 2] = 255
    
    elif noise_type == "blur":
        img = cv2.GaussianBlur(img, (5, 5), 0)

    # Handle blank (zero) images
    if np.sum(img) == 0:
        img = torch.tensor(img).expand(1, 3, *img.shape).float()
        return F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)[0]

    # Normalize and resize
    img = torch.tensor((img - img_mean) / img_std)
    img = img.expand(1, 3, *img.shape).float()
    img_resized = F.interpolate(img, size=target_size, mode='bilinear', align_corners=False)

    return img_resized[0]



def main():
    device = 0
    torch.cuda.set_device(device)

    model = LocalizedDifModel(
        asymmetry_metric=hybrid_asymmetry,
        embedding_channel=512,
        embedding_model=None,
        train_backbone=False,
        use_stretch=True,
        flexible_asymmetry=True,
        use_stretch_matrix=True,
        device_ids=[device],
        dropout_prob=0.7
    ).cuda()
    model.eval()

    # --- Load and restructure metadata ---
    batch_size = 2
    mc_iterations = 100
    val_file_name = '/content/drive/MyDrive/dissertation/AsymMirai-master/AsymMirai-master/asymmetry_model/mirai_metadata.csv'
    val_df = pd.read_csv(val_file_name)
    val_df = restructure_mirai_metadata(val_df)
    required_views = ['L-CC', 'R-CC', 'L-MLO', 'R-MLO']
    val_df = val_df.dropna(subset=required_views)

# Inject noise into the inputs here
    noisy_resizer = partial(resize_and_normalize, noise_type="gaussian")  # Try "salt_pepper" or "blur" too

    val_dataset = MiraiMetadataset(
    val_df,
    resizer=noisy_resizer,
    mode='val',
    align_images=False,
    multiple_pairs_per_exam=False
)

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(10, batch_size)
    )

    os.makedirs('/content/drive/MyDrive/dissertation/training_preds', exist_ok=True)

    def enable_dropout(model):
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
    def mc_dropout_predict(model, inputs, mc_iterations=30, tta=True):
      enable_dropout(model)
      predictions = []

      with torch.no_grad():
          for _ in range(mc_iterations):
              aug_inputs = [x.clone() for x in inputs]
              if tta:
                # Apply horizontal flip as TTA
                  aug_inputs = [torch.flip(x, dims=[-1]) if torch.rand(1).item() > 0.5 else x for x in aug_inputs]

              output, _ = model(*aug_inputs)
              predictions.append(output.unsqueeze(0))

      predictions = torch.cat(predictions, dim=0)
      return predictions.mean(dim=0), predictions.var(dim=0)


    results = []
    thresh = 0.001

    for index, sample in enumerate(val_dataloader):
        eid, label, l_cc_img, l_cc_path, r_cc_img, r_cc_path, l_mlo_img, l_mlo_path, r_mlo_img, r_mlo_path = sample
        inputs = [l_cc_img.cuda(), r_cc_img.cuda(), l_mlo_img.cuda(), r_mlo_img.cuda()]
        label = label.cuda()

        mean_pred, var_pred = mc_dropout_predict(model, inputs, mc_iterations)
        preds = torch.argmax(mean_pred, dim=1)
        uncertainty = var_pred.max(dim=1).values
        ambiguous = uncertainty > thresh


        for i in range(mean_pred.shape[0]):
            results.append({
                'exam_id': eid[i].item(),
                'prediction_neg_mean': mean_pred[i, 0].item(),
                'prediction_pos_mean': mean_pred[i, 1].item(),
                'uncertainty_neg_var': var_pred[i, 0].item(),
                'uncertainty_pos_var': var_pred[i, 1].item(),
                'uncertainty_max': uncertainty[i].item(),
                'ambiguous': ambiguous[i].item()
            })

        if index % 5 == 0:
            pd.DataFrame(results).to_csv(
                '/content/drive/MyDrive/dissertation/training_preds/temp_mc_dropout_predictions.csv',
                index=False
            )

        # Convert results to DataFrame
    df = pd.DataFrame(results)

    # Apply quantile-based ambiguity threshold
    threshold = df['uncertainty_max'].quantile(0.95)
    df['ambiguous'] = df['uncertainty_max'] > threshold

    # Save final results
    df.to_csv(
        '/content/drive/MyDrive/dissertation/training_preds/validation_mc_dropout_predictions.csv',
        index=False
    )

    print(f"✅ Saved final predictions with ambiguity threshold: {threshold:.2e}")

if __name__ == '__main__':
    main()

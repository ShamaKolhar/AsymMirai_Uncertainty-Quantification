# run_model_on_merged_dataset.py

import pandas as pd
import torch
import os
import boto3

from mirai_localized_dif_head import LocalizedDifModel
from asymmetry_metrics import hybrid_asymmetry
from run_utils import get_exam_views, predict_with_uncertainty

def auto_log_predictions(df, model, output_path="/content/predictions_output.csv"):
    results = []
    seen = set()
    for eid in df['exam_id'].unique():
        if eid in seen:
            continue
        print(f"\n🔬 Testing exam_id={eid}")
        l_cc, r_cc, l_mlo, r_mlo = get_exam_views(df, eid)

        if all(v is None for v in [l_cc, r_cc, l_mlo, r_mlo]):
            print("❌ No usable views.")
            continue

        try:
            mean, std = predict_with_uncertainty(model, l_cc, r_cc, l_mlo, r_mlo, T=30)
            results.append({
                "exam_id": eid,
                "mean_pred": mean.tolist(),
                "std_pred": std.tolist(),
                "ambiguous": std.max() > 0.25
            })
        except Exception as e:
            print(f"❌ Error with exam {eid}: {e}")

        torch.cuda.empty_cache()
        seen.add(eid)

    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"📁 Results saved to {output_path}")

def main():
    data_path = "/content/AsymMirai/AsymMirai-master/asymmetry_model/final_exam_image_map.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"🚫 CSV not found: {data_path}")

    # Step 1: Load original data
    raw_df = pd.read_csv(data_path)

    # Step 2: Fetch valid DICOM keys from S3
    s3 = boto3.client("s3")
    bucket_name = "embed-dataset-open"

    valid_keys = set()
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix="images/")
    while True:
        for obj in response.get('Contents', []):
            valid_keys.add(obj['Key'])

        if response.get("IsTruncated"):
            token = response["NextContinuationToken"]
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix="images/", ContinuationToken=token)
        else:
            break

    # Step 3: Filter DataFrame to only include valid paths
    df = raw_df[raw_df['file_path'].str.lstrip("/").isin(valid_keys)].copy()

    print(f"✅ Filtered merged dataset: {df.shape}")
    print(f"📊 Unique exams: {df['exam_id'].nunique()}")

    # Step 4: Initialize and run model
    model = LocalizedDifModel(
        asymmetry_metric=hybrid_asymmetry,
        embedding_channel=512,
        embedding_model=None,
        use_stretch=False,
        train_backbone=False
    )

    auto_log_predictions(df, model, output_path="/content/predictions_output.csv")

if __name__ == "__main__":
    main()

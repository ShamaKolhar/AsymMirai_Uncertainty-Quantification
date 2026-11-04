# testing.py

import pandas as pd
import torch
from mirai_localized_dif_head import LocalizedDifModel
from asymmetry_metrics import hybrid_asymmetry
from run_utils import get_exam_views, predict_with_uncertainty

def main():
    # Load cleaned merged dataset
    df = pd.read_csv("/content/AsymMirai/AsymMirai-master/asymmetry_model/final_exam_image_map.csv")
    print(f"✅ Loaded {df.shape[0]} rows | Unique exams: {df['exam_id'].nunique()}")

    # Pick one known-good exam_id (manually inspected or from earlier run)
    test_eid =5156106114290000  # replace with a valid one you know works
    print(f"\n🔬 Testing exam_id={test_eid}")

    l_cc, r_cc, l_mlo, r_mlo = get_exam_views(df, test_eid)

    if any(x is None for x in [l_cc, r_cc, l_mlo, r_mlo]):
        print("❌ One or more required views are missing.")
        return

    # Load model
    model = LocalizedDifModel(
        asymmetry_metric=hybrid_asymmetry,
        embedding_channel=512,
        embedding_model=None,
        use_stretch=False,
        train_backbone=False
    )

    # Predict
    mean, std = predict_with_uncertainty(model, l_cc, r_cc, l_mlo, r_mlo, T=30)
    print(f"📊 Prediction: {mean}")
    print(f"📉 Uncertainty: {std}")
    print("⚠️ Ambiguous" if std.max() > 0.25 else "✅ Confident")

if __name__ == "__main__":
    main()

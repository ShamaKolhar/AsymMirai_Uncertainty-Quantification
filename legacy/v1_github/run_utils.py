# run_utils.py

import torch
import pydicom
import boto3
import numpy as np
import pandas as pd
import json
from io import BytesIO
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

s3 = boto3.client("s3")
bucket = "embed-dataset-open"

def read_dicom_from_s3(s3_path):
    key = s3_path[len('images/'):] if s3_path.startswith('images/') else s3_path
    try:
        obj = s3.get_object(Bucket=bucket, Key=key)
        dcm = pydicom.dcmread(BytesIO(obj['Body'].read()))
        return dcm.pixel_array
    except Exception as e:
        print(f"❌ Failed to load {s3_path}: {e}")
        return None

def load_tensor(s3_path, mean=7699.5, std=11765.06):
    if s3_path is None:
        return None
    img = read_dicom_from_s3(s3_path)
    if img is None:
        return None
    return torch.tensor((img - mean) / std).expand(1, 3, *img.shape).float().cuda()

def get_exam_views(df, eid, mean=7699.5, std=11765.06):
    exam = df[df['exam_id'] == eid]

    def get_path(side, view):
        match = exam[(exam['view'] == view) & (exam['laterality'] == side)]
        return match['file_path'].values[-1] if len(match) > 0 else None

    def load_tensor_or_dummy(path):
        if path is None:
            return torch.zeros(1, 3, 1024, 1024).float().cuda()  # Dummy image
        img = read_dicom_from_s3(path)
        if img is None:
            return torch.zeros(1, 3, 1024, 1024).float().cuda()
        return torch.tensor((img - mean) / std).expand(1, 3, *img.shape).float().cuda()

    return (
        load_tensor_or_dummy(get_path('L', 'CC')),
        load_tensor_or_dummy(get_path('R', 'CC')),
        load_tensor_or_dummy(get_path('L', 'MLO')),
        load_tensor_or_dummy(get_path('R', 'MLO')),
    )


def apply_tta(tensor_img):
    if tensor_img is None:
        return None
    img_np = tensor_img.squeeze(0).cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    img_pil = T.ToPILImage()(img_np)
    aug_pil = tta(img_pil)
    return TF.to_tensor(aug_pil).unsqueeze(0).cuda()

def predict_with_uncertainty(model, l_cc, r_cc, l_mlo, r_mlo, T=20, augment_fn=None):
    model.train()
    preds = []
    with torch.no_grad():
        for _ in range(T):
            li_cc = augment_fn(l_cc) if augment_fn and l_cc is not None else l_cc
            ri_cc = augment_fn(r_cc) if augment_fn and r_cc is not None else r_cc
            li_mlo = augment_fn(l_mlo) if augment_fn and l_mlo is not None else l_mlo
            ri_mlo = augment_fn(r_mlo) if augment_fn and r_mlo is not None else r_mlo
            out = model(li_cc, ri_cc, li_mlo, ri_mlo)
            preds.append(torch.sigmoid(out).cpu().numpy())
    preds = np.vstack(preds)
    return preds.mean(axis=0), preds.std(axis=0)

def auto_log_predictions(model, df, exam_ids, augment_fn=apply_tta, T=30, out_path="/content/predictions_output.jsonl"):
    results = []
    success = 0

    with open(out_path, "w") as fout:
        for eid in tqdm(exam_ids, desc="🔍 Processing Exams"):
            try:
                l_cc, r_cc, l_mlo, r_mlo = get_exam_views(df, eid)

                mean_pred, std_pred = predict_with_uncertainty(
                    model, l_cc, r_cc, l_mlo, r_mlo, T=T, augment_fn=augment_fn
                )

                result = {
                    "exam_id": int(eid),
                    "mean_pred": mean_pred.tolist(),
                    "std_pred": std_pred.tolist(),
                    "ambiguous": float(std_pred.max()) > 0.25
                }

                fout.write(f"{result}\n")
                results.append(result)
                success += 1

            except Exception as e:
                print(f"❌ Error on exam_id={eid}: {e}")
            finally:
                torch.cuda.empty_cache()

    print(f"\n✅ Completed {success} / {len(exam_ids)} exams")
    return results


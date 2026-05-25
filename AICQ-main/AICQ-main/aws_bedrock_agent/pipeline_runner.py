import os
import json
from typing import List

import pandas as pd
import boto3

from config import get_config
from data_interface import ensure_json_from_csv, load_dataset_json
from helper_hybrid import load_gpr_model

from .data_pipeline import (
    classify_and_map_with_log,
)
from .xai_logger import XAILogger


def inject_labels_into_json(json_path: str, internal_labels: List[str]) -> str:
    payload = load_dataset_json(json_path)
    data = payload.get('data', [])
    if len(data) != len(internal_labels):
        raise ValueError("Length mismatch between dataset and labels")
    for i, row in enumerate(data):
        row['moisture_label'] = internal_labels[i]
    with open(json_path, 'w') as f:
        json.dump(payload, f)
    return json_path


def upload_artifacts_s3(option_name: str, artifacts: list[str]):
    bucket = os.getenv('S3_BUCKET')
    if not bucket:
        return
    s3 = boto3.client('s3')
    for path in artifacts:
        key = f"{option_name}/{os.path.basename(path)}"
        try:
            s3.upload_file(path, bucket, key)
        except Exception:
            pass


def run_with_human_inputs(option_name: str, csv_path: str, json_path: str, human_inputs: List[str]):
    cfg = get_config()
    xai = XAILogger(option_name, out_dir=os.path.dirname(json_path) or '.')
    xai.add_step('Start', {'csv_path': csv_path, 'json_path': json_path})
    json_ready = ensure_json_from_csv(csv_path, json_path)
    xai.add_step('CSV->JSON', {'json_ready': json_ready})
    log_path = os.path.join(os.path.dirname(json_path), f"moisture_classification_{option_name}.csv")
    labels, internal, confs = classify_and_map_with_log(human_inputs, log_path)
    xai.add_step('LLM classification', {'count': len(labels), 'log': log_path})
    json_ready = inject_labels_into_json(json_ready, internal)
    xai.add_step('Injected moisture_label', {'json_ready': json_ready})
    backend = 'sagemaker' if os.getenv('SAGEMAKER_ENDPOINT_NAME') else 'local_joblib'
    xai.add_step('Model backend', {'backend': backend, 'endpoint': os.getenv('SAGEMAKER_ENDPOINT_NAME', '')})
    gpr_model = load_gpr_model(cfg['model']['path'])
    from hybrid_gpr_by_mask import run_option as run
    run(option_name, csv_path, json_ready, cfg, gpr_model)
    pred_csv = os.path.join(os.path.dirname(__file__), f"../prediction_vs_ndg_{option_name}.csv")
    res_txt = os.path.join(os.path.dirname(__file__), f"../final_results_{option_name}.txt")
    xai.add_step('Pipeline finished', {'predictions': pred_csv, 'summary': res_txt})
    # Upload artifacts if requested
    artifacts = [
        log_path,
        pred_csv,
        res_txt,
    ]
    upload_artifacts_s3(option_name, artifacts)
    xai.add_step('Artifacts', {'uploaded_to_s3': bool(os.getenv('S3_BUCKET')), 'paths': artifacts})
    xai.write()


def run_from_csv_with_text(option_name: str, csv_in_path: str, csv_copy_out_path: str, json_out_path: str):
    cfg = get_config()
    xai = XAILogger(option_name, out_dir=os.path.dirname(json_out_path) or '.')
    xai.add_step('Start', {'csv_in_path': csv_in_path, 'csv_copy_out_path': csv_copy_out_path, 'json_out_path': json_out_path})
    os.makedirs(os.path.dirname(csv_copy_out_path), exist_ok=True)
    df = pd.read_csv(csv_in_path)
    if 'moisture_label' in df.columns:
        df = df.drop(columns=['moisture_label'])
    if 'moisture_text' not in df.columns:
        raise ValueError("CSV must contain a 'moisture_text' column of human descriptions")
    df.to_csv(csv_copy_out_path, index=False)
    xai.add_step('Prepared CSV copy', {'csv_copy_out_path': csv_copy_out_path})
    from .data_pipeline import csv_to_json_without_label
    payload = csv_to_json_without_label(csv_copy_out_path, json_out_path)
    xai.add_step('CSV->JSON', {'json_out_path': json_out_path, 'rows': len(payload.get('data', []))})
    texts = [str(row.get('moisture_text') or '') for row in payload['data']]
    log_path = os.path.join(os.path.dirname(json_out_path), f"moisture_classification_{option_name}.csv")
    labels, internal, confs = classify_and_map_with_log(texts, log_path)
    xai.add_step('LLM classification', {'count': len(labels), 'log': log_path})
    json_ready = inject_labels_into_json(json_out_path, internal)
    xai.add_step('Injected moisture_label', {'json_ready': json_ready})
    backend = 'sagemaker' if os.getenv('SAGEMAKER_ENDPOINT_NAME') else 'local_joblib'
    xai.add_step('Model backend', {'backend': backend, 'endpoint': os.getenv('SAGEMAKER_ENDPOINT_NAME', '')})
    gpr_model = load_gpr_model(cfg['model']['path'])
    from hybrid_gpr_by_mask import run_option as run
    run(option_name, csv_copy_out_path, json_ready, cfg, gpr_model)
    pred_csv = os.path.join(os.path.dirname(__file__), f"../prediction_vs_ndg_{option_name}.csv")
    res_txt = os.path.join(os.path.dirname(__file__), f"../final_results_{option_name}.txt")
    xai.add_step('Pipeline finished', {'predictions': pred_csv, 'summary': res_txt})
    artifacts = [
        log_path,
        pred_csv,
        res_txt,
    ]
    upload_artifacts_s3(option_name, artifacts)
    xai.add_step('Artifacts', {'uploaded_to_s3': bool(os.getenv('S3_BUCKET')), 'paths': artifacts})
    xai.write()

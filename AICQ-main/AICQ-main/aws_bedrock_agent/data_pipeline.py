import os
import json
from typing import List, Dict, Any

import pandas as pd

from .bedrock_client import BedrockMoistureClassifier


REQUIRED_BASE_COLUMNS = [
    'NDG-density', 'rel_X (m)', 'rel_Y (m)', 'usedfortraining', 'OMC'
]


def classify_and_map_with_log(human_inputs: List[str], log_path: str):
    """Classify free-text moisture to dry|medium|wet, map to low|medium|high, and log.

    Returns (labels, internal_labels, confidences).
    """
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    clf = BedrockMoistureClassifier()
    labels: List[str] = []
    internals: List[str] = []
    confidences: List[float] = []
    with open(log_path, 'w') as f:  # overwrite each run for clarity
        f.write("index,input,llm_label,internal_label,confidence\n")
        for i, txt in enumerate(human_inputs):
            lb, conf = clf.classify(txt or "medium")
            if lb == "dry":
                internal = "low"
            elif lb == "wet":
                internal = "high"
            else:
                internal = "medium"
            labels.append(lb)
            internals.append(internal)
            confidences.append(conf)
            safe_txt = (txt or "").replace('\n',' ').replace('\r',' ')
            f.write(f"{i},{safe_txt},{lb},{internal},{conf:.3f}\n")
    return labels, internals, confidences


def csv_to_json_without_label(csv_path: str, json_path: str) -> Dict[str, Any]:
    """Convert a Book2 CSV copy (with moisture_text, no moisture_label) to JSON.

    - Validates required base columns and presence of moisture_text
    - Ensures LWD_av is present (computes from LWD2/3/4 if missing)
    - Writes a dataset JSON consumed downstream
    """
    df = pd.read_csv(csv_path)

    # Validate required structure
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} missing required columns: {missing}")
    if 'moisture_text' not in df.columns:
        raise ValueError("CSV must contain a 'moisture_text' column of human descriptions")

    # Do not allow pre-labeled moisture_label in these copies
    if 'moisture_label' in df.columns:
        df = df.drop(columns=['moisture_label'])

    # Ensure LWD_av exists; compute if missing
    if 'LWD_av' not in df.columns:
        if {'LWD2','LWD3','LWD4'}.issubset(df.columns):
            df['LWD_av'] = (df['LWD2'] + df['LWD3'] + df['LWD4']) / 3.0
        else:
            df['LWD_av'] = 0.0

    records = df.to_dict(orient='records')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    payload = {
        'schema_version': 1,
        'source_csv': os.path.abspath(csv_path),
        'num_rows': len(records),
        'data': records,
    }
    with open(json_path, 'w') as f:
        json.dump(payload, f)
    return payload

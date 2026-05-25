import os
import json
import pandas as pd
from typing import Dict, Any


# LWD2/3/4 are no longer hard-required. We require either LWD_av, or
# the presence of LWD2+LWD3+LWD4 to compute LWD_av.
REQUIRED_COLUMNS_BASE = [
    'NDG-density', 'rel_X (m)', 'rel_Y (m)', 'usedfortraining', 'OMC'
]


def csv_to_json(csv_path: str, json_path: str) -> None:
    """Convert a CSV dataset to JSON, ensuring required fields exist.

    Ensures `LWD_av` is present (computed from LWD2/3/4 if missing) and writes a
    structured JSON containing metadata and row records. Raises if required CSV
    columns are missing.

    Args:
        csv_path: Source CSV path.
        json_path: Destination JSON path.
    """
    df = pd.read_csv(csv_path)
    missing = [c for c in REQUIRED_COLUMNS_BASE if c not in df.columns]
    if missing:
        raise ValueError(f"CSV {csv_path} missing required columns: {missing}")

    # Ensure LWD_av exists; compute if possible
    if 'LWD_av' not in df.columns:
        if all(c in df.columns for c in ('LWD2','LWD3','LWD4')):
            df['LWD_av'] = (df['LWD2'] + df['LWD3'] + df['LWD4']) / 3.0
        else:
            # If neither LWD_av nor the components exist, default to 0.0
            df['LWD_av'] = 0.0

    records = df.to_dict(orient='records')
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump({
            'schema_version': 1,
            'source_csv': os.path.abspath(csv_path),
            'num_rows': len(records),
            'data': records,
        }, f)


def load_dataset_json(json_path: str) -> Dict[str, Any]:
    """Load a dataset JSON produced by csv_to_json.

    Also ensures `LWD_av` is available for older files by reconstructing it in
    memory if needed.

    Args:
        json_path: Path to JSON file.

    Returns:
        Dict[str, Any]: Parsed dataset with 'data' records.
    """
    with open(json_path, 'r') as f:
        payload = json.load(f)
    # Backward compatibility: if LWD_av missing in data, reconstruct in-memory
    if payload.get('data') and isinstance(payload['data'], list):
        any_missing = any('LWD_av' not in row for row in payload['data'])
        if any_missing:
            for row in payload['data']:
                row['LWD_av'] = (row['LWD2'] + row['LWD3'] + row['LWD4']) / 3.0
    return payload


def ensure_json_from_csv(csv_path: str, json_path: str) -> str:
    """Convert CSV to JSON if JSON is missing or older than CSV.

    Args:
        csv_path: Source CSV path.
        json_path: Destination JSON path.

    Returns:
        str: Path to the up-to-date JSON file.
    """
    if not os.path.exists(json_path) or os.path.getmtime(json_path) < os.path.getmtime(csv_path):
        csv_to_json(csv_path, json_path)
    return json_path



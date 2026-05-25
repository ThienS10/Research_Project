import os
from typing import List

from fastapi import FastAPI
from pydantic import BaseModel

from .bedrock_client import BedrockMoistureClassifier
from .pipeline_runner import run_with_human_inputs, run_from_csv_with_text
from config import get_config

app = FastAPI(title="AWS Agent: Moisture Classifier + Hybrid GPR")


class ClassifyRequest(BaseModel):
    input: str


class ClassifyManyRequest(BaseModel):
    inputs: List[str]


class ProcessDatasetRequest(BaseModel):
    option_name: str
    csv_path: str
    json_path: str
    human_inputs: List[str]


class ProcessCsvTextRequest(BaseModel):
    option_name: str
    csv_in_path: str
    csv_copy_out_path: str
    json_out_path: str


@app.post("/classify")
def classify(req: ClassifyRequest):
    clf = BedrockMoistureClassifier()
    label, confidence = clf.classify(req.input)
    return {"label": label, "confidence": confidence}


@app.post("/classify_many")
def classify_many(req: ClassifyManyRequest):
    clf = BedrockMoistureClassifier()
    labels = []
    confidences = []
    for t in req.inputs:
        lb, conf = clf.classify(t)
        labels.append(lb)
        confidences.append(conf)
    return {"labels": labels, "confidences": confidences}


@app.post("/process_dataset")
def process_dataset(req: ProcessDatasetRequest):
    get_config()  # ensure config loads
    run_with_human_inputs(req.option_name, req.csv_path, req.json_path, req.human_inputs)
    log_path = os.path.join(os.path.dirname(req.json_path), f"moisture_classification_{req.option_name}.csv")
    return {"status": "ok", "message": "Pipeline executed", "option": req.option_name, "log": log_path}


@app.post("/process_csv_text")
def process_csv_text(req: ProcessCsvTextRequest):
    get_config()  # ensure config loads
    run_from_csv_with_text(req.option_name, req.csv_in_path, req.csv_copy_out_path, req.json_out_path)
    log_path = os.path.join(os.path.dirname(req.json_out_path), f"moisture_classification_{req.option_name}.csv")
    return {"status": "ok", "message": "Pipeline executed from CSV with moisture_text", "option": req.option_name, "log": log_path, "csv_copy": req.csv_copy_out_path}


# For Lambda via API Gateway
try:
    from mangum import Mangum
    handler = Mangum(app)
except Exception:
    handler = None

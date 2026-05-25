## AIQA (AWS AI Agent Hackathon)

AIQA is an augmented intelligence system for quality assurance of soil compaction. It blends physics-based ground modeling and a GPR surrogate with an AWS-hosted reasoning LLM that interprets human moisture descriptions, enabling real-time assessment under significant uncertainty. In practice, real-time moisture content measurement during compaction is rare due to significant technical and operational challenges; this gap often leads to miscompaction and premature road failures. AIQA addresses it by leveraging human contextual observations (e.g., field notes, photos, crew feedback) and using Bedrock to convert them into rigorous moisture states that feed directly into the physical model and GPR pipeline, restoring the missing variable without intrusive sensors.

### What problem it solves
- Field data often lacks a reliable per-row “moisture_label” needed by the physics/GPR pipeline.
- This agent uses an LLM to interpret human text (moisture_text) into domain labels (dry|medium|wet → low|medium|high), injects them into the dataset used by the pipeline, and runs the full computation.

### How the LLM is used (key gap filled)
- Input: free-text `moisture_text` per row (e.g., “slightly damp”, “waterlogged”).
- LLM decision (Amazon Bedrock, Claude): classify text to `dry|medium|wet`.
- Mapping: map to internal labels `low|medium|high` required by physics/GPR.
- Impact: labels drive moisture numeric (via OMC ± offset) and suction, directly affecting physics and GPR predictions.
- Fuzzy interface (roadmap): classified labels will be converted via singleton fuzzification for use in the physical model, enabling smooth integration of human linguistic inputs with suction/void-ratio computations.

### Tech used (aligned to award requirements)
- Amazon Bedrock (Claude via bedrock-runtime): reasoning LLM for decision-making.
- Amazon Bedrock AgentCore (optional but included): agent + Lambda action group to route tasks to the pipeline.
- AWS SAM (API Gateway + Lambda): deployment of the FastAPI microservice.
- Amazon SageMaker (optional, supported): host the GPR surrogate model; pipeline calls it via sagemaker-runtime InvokeEndpoint when configured.
- Amazon S3 (optional): artifact uploads (predictions, logs, reports) after each run.
- FastAPI + Uvicorn: local API for development and demos.
- XAI logging: structured JSON + Markdown reports describing each step, parameters, and outputs.

### Folder structure
- `data/`: Book2 CSV copies with `moisture_text` only (no `moisture_label`), and generated JSON + logs/reports.
- `aws_bedrock_agent/`: service, Bedrock client, pipeline runner, AgentCore helpers, SAM template.
- `helper_hybrid.py`, `hybrid_gpr_by_mask.py`, `data_interface.py`, `hybrid_initial_settings.py`, `config.py`: core physics + GPR pipeline, unchanged and reusable.
- `gpr_model_constrained3.joblib`: local pre-trained surrogate.
- `config.json`: points to `data/*` and local model path.
- `run.py`: runs both options end-to-end.
- `requirements.txt`: dependencies.

### Local run
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install -r requirements.txt`
3. `export AWS_REGION=us-east-1`
4. Optional (require real Bedrock, no fallback): `export BEDROCK_STRICT=1`
5. Optional (use SageMaker endpoint instead of local joblib): `export SAGEMAKER_ENDPOINT_NAME=<your-endpoint>`
6. Optional (upload artifacts to S3): `export S3_BUCKET=<your-bucket>`
7. `python run.py`

Generated artifacts
- XAI reports (explainability):
  - `data/xai_report_<option>.json`, `data/xai_report_<option>.md`
- LLM classification log (audit):
  - `data/moisture_classification_<option>.csv`
- Predictions and summaries:
  - `prediction_vs_ndg_<option>.csv`, `final_results_<option>.txt`
- Visual map (if produced):
  - `Hybrid_Lane_map_best_kernel_<option>.tiff`

### Local API (FastAPI)
- Start: `uvicorn aws_bedrock_agent.app:app --reload`
- Process CSV with `moisture_text`:
```
POST /process_csv_text
{
  "option_name": "option1_llm",
  "csv_in_path": "data/Book2_option1_llm.csv",
  "csv_copy_out_path": "data/Book2_option1_llm.csv",
  "json_out_path": "data/Book2_option1_llm.json"
}
```
- Returns JSON status and writes artifacts as above.

### Deploy (SAM → API Gateway + Lambda)
1. `cd aws_bedrock_agent`
2. `sam build`
3. `sam deploy --guided`
4. In Lambda env vars, set as needed:
   - `AWS_REGION`, `BEDROCK_STRICT=1`, `S3_BUCKET` (optional), and if you mount/host inputs elsewhere, adjust paths accordingly.

### Minimal Lambda bundle (CloudShell friendly)
- Purpose: expose only `/classify` and `/classify_many` without heavyweight scientific dependencies, keeping the unzipped size well below the 250 MB CloudShell limit.
- Build:
  1. `./deployment/lambda_minimal/build.sh` (runs from repo root; outputs `dist/lambda-minimal.zip`).
  2. Upload the ZIP to Lambda (or through CloudShell → S3) and set the handler to `lambda_handler.handler`.
- Included files: `lambda_handler.py` and the lightweight `aws_bedrock_agent/bedrock_client.py` + config.
- Not included: the physics/GPR pipeline, pandas/numpy/scikit-learn, and the local `.joblib` model. The handler returns `501` if you call `/process_dataset` or `/process_csv_text`.
- Dependencies: none—AWS Lambda’s built-in `boto3`/`botocore` is sufficient.
- To re-enable the full pipeline, deploy with SAM and install from `requirements.txt` (or add a SageMaker endpoint for the surrogate model).

### AgentCore (optional, recommended)
- Provision an Agent and Lambda action group:
```
cd aws_bedrock_agent
export AWS_REGION=us-east-1
export AGENT_FOUNDATION_MODEL="anthropic.claude-3-5-sonnet-20240620-v1:0"
export LAMBDA_ARN="arn:aws:lambda:<region>:<acct>:function:<your-ApiFunction>"
python agentcore_setup.py
# Note printed agent_id and alias_id
```
- Invoke the agent:
```
export AGENT_ID=<agent_id>
export AGENT_ALIAS_ID=<alias_id>
python agent_invoke.py
```

### SageMaker integration (optional, supported)
- Host the GPR surrogate on SageMaker (sklearn-compatible container).
- Set `SAGEMAKER_ENDPOINT_NAME` and `AWS_REGION` before running.
- The pipeline will call `sagemaker-runtime:InvokeEndpoint` via an adapter; otherwise it falls back to the local `joblib` model.

### Why this is an “agent”
- Uses a reasoning LLM (Bedrock Claude) to decide per-row moisture labels from human text.
- Demonstrates autonomy: classification → label injection → physics + GPR → metrics and artifacts.
- Integrates AWS services: Bedrock (LLM), AgentCore (optional), API Gateway + Lambda (SAM), SageMaker (optional), and S3 (optional).

### Notes and configuration
- Strict mode: `BEDROCK_STRICT=1` forces real Bedrock calls (no heuristic fallback). Use for production/judging.
- Classification transparency: every run writes `moisture_classification_<option>.csv` (input, LLM label, mapped label, confidence).
- Explainability: `xai_report_<option>.json/.md` capture step-by-step actions, parameters, and outputs.

### Minimal reproducible demo
- Use the provided CSV copies in `data/` with `moisture_text` already present.
- Run `python run.py` to generate labels with LLM, inject into JSON, and produce predictions and reports.

### Automated demo script (for submission video)
- Ensure dependencies are installed (`pip install -r requirements.txt`). If you hit NumPy warnings, pin with `pip install "numpy<2" bottleneck numexpr`.
- Run `python demo_showcase.py` to execute both pipeline options and print:
  - holdout accuracy metrics from `final_results_<option>.txt`
  - the first few rows of `prediction_vs_ndg_<option>.csv`
  - the moisture classification audit log
  - a snapshot of the explainability steps recorded in `xai_report_<option>.json`
- The script sets `AWS_REGION=us-east-1` if unset and relies on the Bedrock heuristic fallback, so it works offline.
- Capture this scripted terminal run in your video, then cut to the generated artifacts (e.g., `Hybrid_Lane_map_best_kernel_<option>.tiff`) to stay within the 3-minute limit.

### Conclusion

This paper has highlighted the intricate challenges of real-time assessment of unsaturated soils using deflection tests. The augmented intelligence framework introduced in this study (AIQA) integrates human insight with the analytical and computational prowess of AI, aiming to overcome the limitations presented by each approach when used independently. By incorporating fuzzy human inputs and sophisticated AI algorithms, the framework addresses critical gaps in traditional ground assessment methods, notably in scenarios where key data such as gravimetric water content and suction are missing.

The validation of this framework through recent field trials has demonstrated its potential to significantly improve the accuracy and reliability of geotechnical ground assessments. Specifically, in the conducted field trials, the framework achieved an overall accuracy of 89% in correctly identifying acceptable grid points, a roughly 45% point improvement compared to traditional empirical methods. These results illustrate the framework’s ability to enhance the qualitative and quantitative assessment of soil properties during compaction, even when key data are unavailable. Moreover, the framework proved robust against errors in gravimetric water content assumptions, maintaining accuracy despite deviations in human-derived inputs.

Reference: Ghorbani, J., Aghdasi, S., Nazem, M., McCartney, J.S. and Kodikara, J., 2025. Augmented intelligence framework for real-time ground assessment under significant uncertainty. Engineering with Computers, pp.1-22.

# End-to-End Deployment Guide

Step-by-step instructions to deploy the Medical Vision Pipeline from scratch, run the API, and process real clinical documents. This guide assumes you are starting from a clean machine with an NVIDIA GPU.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Environment Setup](#3-environment-setup)
4. [Download the Model](#4-download-the-model)
5. [Quantize the Model (NF4)](#5-quantize-the-model-nf4)
6. [Start the API Server](#6-start-the-api-server)
7. [Start the Gradio Demo UI](#7-start-the-gradio-demo-ui)
8. [Deploy with Docker](#8-deploy-with-docker)
9. [Testing the API](#9-testing-the-api)
10. [Sample Clinical Use Cases](#10-sample-clinical-use-cases)
11. [Using the HuggingFace Adapter Directly](#11-using-the-huggingface-adapter-directly)
12. [Production Considerations](#12-production-considerations)
13. [Troubleshooting](#13-troubleshooting)

---

## 1. Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU VRAM | 8 GB | 16+ GB |
| System RAM | 16 GB | 32+ GB |
| Disk Space | 30 GB | 50 GB |
| CUDA Compute | sm_70+ | sm_80+ |

**Tested on:** NVIDIA DGX Spark with GB10 GPU (Blackwell sm_121, 119.7 GB unified VRAM).

The NF4-quantized model uses only **5.7 GB VRAM** at inference time, making it accessible on consumer GPUs (RTX 3060 12GB, RTX 4070, etc.).

### Software Requirements

- **Operating System:** Linux (Ubuntu 20.04+ recommended), or WSL2 on Windows
- **NVIDIA Driver:** 535+ (for CUDA 12.x)
- **CUDA Toolkit:** 12.x
- **Python:** 3.10+
- **conda** or **venv** for virtual environments
- **Docker + NVIDIA Container Toolkit** (for containerized deployment only)
- **Git** and **git-lfs** (for cloning)

### Verify GPU and CUDA

```bash
# Check NVIDIA driver and GPU
nvidia-smi

# Expected output should show your GPU name, driver version, and CUDA version
# Example:
#   NVIDIA GB10        | Driver: 575.51.03  | CUDA: 12.8
```

---

## 2. Clone the Repository

```bash
git clone https://github.com/sarathi-aiml/quantization-pipe.git
cd quantization-pipe
```

The repository contains scripts, API code, benchmarks, and documentation. Model weights are NOT included (they are 15+ GB) and must be downloaded separately.

---

## 3. Environment Setup

### Option A: Conda (Recommended)

```bash
# Create environment
conda create -n med-vision python=3.10 -y
conda activate med-vision

# Install PyTorch with CUDA support
# For standard GPUs (sm_70 to sm_90):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For Blackwell GPUs (sm_121) -- requires nightly:
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128

# Install project dependencies
pip install -r requirements.txt
```

### Option B: venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Validate the Environment

```bash
python scripts/validate_environment.py
```

This checks that PyTorch can see the GPU, CUDA is functional, and all required packages are installed. Expected output:

```
GPU: NVIDIA GB10 (or your GPU)
CUDA: 12.x
PyTorch: 2.x.x+cuXXX
All checks passed.
```

> **Blackwell GPU note:** The `scripts/blackwell_compat.py` module patches integer reduction ops (`prod`, `cumprod`) that fail via NVRTC JIT on sm_121. This is automatically applied when running the API server or any pipeline script.

---

## 4. Download the Model

```bash
# Download Qwen2.5-VL-7B-Instruct from HuggingFace (~15.5 GB, takes ~6 minutes)
python scripts/download_model.py
```

This downloads to `models/qwen25-vl-7b-base/`. You'll see a progress bar and download time is logged.

**Alternative:** If you already have the model cached, symlink it:
```bash
ln -s /path/to/your/Qwen2.5-VL-7B-Instruct models/qwen25-vl-7b-base
```

### Download the Fine-Tuned LoRA Adapter

The LoRA adapter is hosted on HuggingFace and is much smaller (~727 MB):

```bash
# Download adapter to the expected location
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'sarathi-balakrishnan/Qwen2.5-VL-7B-Medical-LoRA',
    local_dir='models/qwen25-vl-7b-medical-lora'
)
print('LoRA adapter downloaded successfully')
"
```

---

## 5. Quantize the Model (NF4)

```bash
# This creates the NF4 quantization config at models/qwen25-vl-7b-bnb-nf4/
python scripts/quantize_base.py

# Verify quantization with a benchmark run
python scripts/benchmark_quantized.py
```

The quantization step configures bitsandbytes NF4 (4-bit NormalFloat with double quantization). The actual quantization happens on-the-fly during model loading -- there's no separate quantized model file. The config is saved so the API server knows which quantization settings to use.

**Expected results:**
- VRAM: **5,664 MB** (down from 15,820 MB in FP16)
- Avg latency: **15.24s** per image (down from 31.10s)

---

## 6. Start the API Server

### Method 1: Direct Launch

```bash
# Activate your environment
conda activate med-vision  # or source .venv/bin/activate

# Start the FastAPI server
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1
```

**What happens on startup:**
1. Blackwell compatibility patches are applied (if needed)
2. Base model loads with NF4 quantization (~90 seconds)
3. LoRA adapter loads and merges into the base model
4. Server becomes ready at `http://localhost:8000`

**Wait for this log message before sending requests:**
```
INFO - Model ready on cuda:0  |  VRAM allocated: 11335.8 MB
INFO - Uvicorn running on http://0.0.0.0:8000
```

### Method 2: Run as Background Process

```bash
nohup python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 --workers 1 > api.log 2>&1 &
echo $! > api.pid

# Check the log
tail -f api.log

# Later, to stop:
kill $(cat api.pid)
```

### Verify the Server is Running

```bash
curl http://localhost:8000/health | python -m json.tool
```

Expected response:
```json
{
    "model_loaded": true,
    "gpu_available": true,
    "gpu_name": "NVIDIA GB10",
    "current_vram_usage_mb": 11335.83,
    "total_vram_mb": 122597.38,
    "timestamp": "2026-02-08T21:00:00.000000+00:00"
}
```

### API Documentation (Swagger)

Open `http://localhost:8000/docs` in your browser for the interactive Swagger UI where you can test all endpoints directly.

---

## 7. Start the Gradio Demo UI

In a **separate terminal** (while the API server is running):

```bash
conda activate med-vision
python -m api.demo_ui
```

Open `http://localhost:7860` in your browser. The Gradio UI has three tabs:

1. **Medical Term Extraction** -- Upload an image, click "Extract", get structured medical data
2. **Custom Analysis** -- Upload an image with a custom prompt for specific analysis
3. **Benchmarks** -- View all pipeline benchmark results

---

## 8. Deploy with Docker

Docker deployment packages everything into a single container that runs both the API server and Gradio demo.

### Build the Image

```bash
docker build -t medical-vision-pipeline .
```

### Run the Container

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html):

```bash
docker run --gpus all \
  -p 8000:8000 \
  -p 7860:7860 \
  medical-vision-pipeline
```

The container:
1. Starts the FastAPI server on port 8000
2. Waits for the model to load (health check polling)
3. Starts the Gradio demo on port 7860
4. Runs a health check every 60 seconds with a 5-minute startup grace period

### Persistent Model Cache (Avoid Re-downloading)

Mount your local models directory to avoid re-downloading on each container start:

```bash
docker run --gpus all \
  -p 8000:8000 \
  -p 7860:7860 \
  -v $(pwd)/models:/app/models \
  medical-vision-pipeline
```

### Access Points

| Service | URL |
|---------|-----|
| FastAPI Server | http://localhost:8000 |
| Swagger API Docs | http://localhost:8000/docs |
| Gradio Demo | http://localhost:7860 |
| Health Check | http://localhost:8000/health |

---

## 9. Testing the API

### Run Automated Tests

The test suite starts its own server instance, creates a synthetic medical document, tests all endpoints, and reports results:

```bash
python tests/test_api.py
```

Expected output:
```
========================================================================
  MEDICAL VISION API -- END-TO-END TEST SUITE
========================================================================

  [PASS]  GET /health
         OK -- GPU: NVIDIA GB10, VRAM: 5673 MB

  [PASS]  POST /extract
         OK -- 7/7 terms found, latency=27.69s, response=1352 chars

  [PASS]  POST /analyze
         OK -- 5/5 med terms, latency=11.09s, response=... chars

  [PASS]  GET /benchmark
         OK -- 8 files: baseline_fp16.json, ...

  [PASS]  POST /extract (bad input)
         OK -- correctly returned 400

------------------------------------------------------------------------
  Total: 5  |  Passed: 5  |  Failed: 0
========================================================================
```

### Manual Testing with curl

**Extract medical information (default prompt):**
```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@your_medical_document.png" | python -m json.tool
```

**Analyze with a custom prompt:**
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@your_medical_document.png" \
  -F "prompt=List all medications with their dosages and routes" | python -m json.tool
```

**Check server health:**
```bash
curl http://localhost:8000/health | python -m json.tool
```

**Get all benchmark data:**
```bash
curl http://localhost:8000/benchmark | python -m json.tool
```

### Testing with Python requests

```python
import requests

API_URL = "http://localhost:8000"

# Health check
health = requests.get(f"{API_URL}/health").json()
print(f"Model loaded: {health['model_loaded']}")
print(f"GPU: {health['gpu_name']}")
print(f"VRAM: {health['current_vram_usage_mb']:.0f} MB")

# Extract from an image
with open("medical_document.png", "rb") as f:
    response = requests.post(
        f"{API_URL}/extract",
        files={"file": ("doc.png", f, "image/png")}
    )

result = response.json()
print(f"\nLatency: {result['inference_latency_seconds']:.2f}s")
print(f"Extracted content:\n{result['extracted_content']}")
```

---

## 10. Sample Clinical Use Cases

Below are detailed, ready-to-run examples for common clinical document processing scenarios. Each example includes the curl command, the expected API response structure, and what to look for in the output.

### Use Case 1: Emergency Department Triage

**Scenario:** A patient arrives at the ED with a referral letter. The triage nurse needs to quickly extract diagnoses, medications, allergies, and vital signs.

**Step 1:** Photograph or scan the referral letter and save as `ed_referral.png`.

**Step 2:** Send to the API:
```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@ed_referral.png" \
  -F "prompt=Extract all medical diagnoses, current medications with dosages, known allergies, and vital signs from this referral document. Flag any critical or abnormal values." \
  | python -m json.tool
```

**Expected output structure:**
```json
{
    "analysis": "**Diagnoses:**\n1. Acute Myocardial Infarction (STEMI)\n2. Hypertension Stage 2\n...\n\n**Medications:**\n- Aspirin 325 mg PO STAT\n- Heparin 60 units/kg IV\n...\n\n**Allergies:**\n- Penicillin (anaphylaxis)\n...\n\n**Vital Signs:**\n- BP: 172/98 mmHg (HIGH)\n- HR: 108 bpm (TACHYCARDIA)\n...",
    "prompt_used": "Extract all medical diagnoses...",
    "inference_latency_seconds": 15.2,
    "model_info": {
        "model_name": "Qwen2.5-VL-7B-Instruct",
        "precision": "NF4 (bitsandbytes)",
        "lora_adapter": "qwen25-vl-7b-medical-lora",
        "device": "cuda:0"
    },
    "timestamp": "2026-02-08T21:00:00+00:00"
}
```

### Use Case 2: Prescription Processing

**Scenario:** A pharmacy needs to digitize handwritten/printed prescriptions into structured data for the dispensing system.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@prescription.png" \
  -F "prompt=Extract all prescribed medications with their exact dosages, directions for use, quantity dispensed, and number of refills. Format as a numbered list." \
  | python -m json.tool
```

**What to look for in the output:**
- Each medication as a separate entry
- Dosage with units (mg, mcg, units)
- Route of administration (PO, IV, SL, IM, topical)
- Frequency (QD, BID, TID, QID, PRN, Q4H, QHS)
- Quantity and refills

### Use Case 3: Laboratory Results Parsing

**Scenario:** A clinician receives lab results as a scanned PDF page and needs them in a structured format for the EHR.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@lab_results.png" \
  -F "prompt=Extract all laboratory test results. For each test, provide: test name, result value, units, reference range, and flag (H for high, L for low, or normal). Format as a table." \
  | python -m json.tool
```

**Expected accuracy:** Our benchmark shows 83-84% accuracy on complex lab panels with the base NF4 model, correctly identifying test names, values, units, and abnormal flags.

### Use Case 4: Discharge Summary Processing

**Scenario:** A case manager needs to extract follow-up instructions and medication changes from a discharge summary for care coordination.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@discharge_summary.png" \
  -F "prompt=Extract: 1) All discharge diagnoses, 2) Complete discharge medication list with dosages and frequencies, 3) All follow-up appointments with dates and specialties, 4) Activity restrictions, 5) Warning signs to watch for." \
  | python -m json.tool
```

### Use Case 5: ICU Flowsheet Digitization

**Scenario:** An ICU nurse needs to digitize a paper flowsheet for quality review.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@icu_flowsheet.png" \
  -F "prompt=Extract all ventilator settings, hemodynamic parameters (HR, MAP, CVP, ScvO2), active medication drips with rates, ABG values, intake/output totals, and current diagnoses from this ICU flowsheet." \
  | python -m json.tool
```

**Benchmark result:** The base NF4 model achieved **100% accuracy** on our ICU flowsheet test case, correctly extracting all 19 medical terms and all 30 values including ventilator settings, drip rates, and hemodynamic parameters.

### Use Case 6: Surgical Operative Note Parsing

**Scenario:** A surgical department needs to extract procedure details for coding and billing.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@operative_note.png" \
  -F "prompt=Extract the following from this operative note: preoperative diagnosis, postoperative diagnosis, procedure name and details, operative findings with measurements, estimated blood loss, specimens sent to pathology, and all postoperative medication orders with dosages." \
  | python -m json.tool
```

**Benchmark result:** 90.2% combined accuracy on surgical operative notes.

### Use Case 7: Batch Processing Multiple Documents

**Scenario:** A research team needs to process hundreds of clinical documents for a retrospective study.

```python
#!/usr/bin/env python3
"""Batch process clinical documents through the Medical Vision API."""

import json
import time
from pathlib import Path
import requests

API_URL = "http://localhost:8000"
INPUT_DIR = Path("clinical_documents")
OUTPUT_DIR = Path("extracted_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PROMPT = """Extract all medical diagnoses, medications with dosages,
laboratory values with reference ranges, and procedures from this document.
Format the output as structured JSON."""

# Verify API is ready
health = requests.get(f"{API_URL}/health").json()
assert health["model_loaded"], "Model not loaded!"
print(f"API ready | GPU: {health['gpu_name']} | VRAM: {health['current_vram_usage_mb']:.0f} MB")

# Process all images
results = []
image_files = sorted(INPUT_DIR.glob("*.png")) + sorted(INPUT_DIR.glob("*.jpg"))
print(f"\nProcessing {len(image_files)} documents...\n")

for i, img_path in enumerate(image_files, 1):
    t0 = time.time()
    with open(img_path, "rb") as f:
        response = requests.post(
            f"{API_URL}/analyze",
            files={"file": (img_path.name, f, "image/png")},
            data={"prompt": PROMPT},
            timeout=300,
        )

    if response.status_code != 200:
        print(f"  [{i}/{len(image_files)}] FAILED: {img_path.name} -> HTTP {response.status_code}")
        continue

    data = response.json()
    elapsed = time.time() - t0

    # Save individual result
    out_path = OUTPUT_DIR / f"{img_path.stem}_extracted.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    results.append({
        "file": img_path.name,
        "latency": data["inference_latency_seconds"],
        "response_length": len(data["analysis"]),
    })

    print(f"  [{i}/{len(image_files)}] {img_path.name} -> {data['inference_latency_seconds']:.1f}s, {len(data['analysis'])} chars")

# Summary
print(f"\n{'='*60}")
print(f"Processed: {len(results)}/{len(image_files)} documents")
if results:
    avg_latency = sum(r["latency"] for r in results) / len(results)
    print(f"Average latency: {avg_latency:.1f}s per document")
    print(f"Total time: {sum(r['latency'] for r in results):.0f}s")
    print(f"Results saved to: {OUTPUT_DIR}/")
```

Save as `batch_process.py` and run:
```bash
mkdir -p clinical_documents
# Place your .png or .jpg clinical documents in the directory
python batch_process.py
```

### Use Case 8: Cardiology Consultation Note

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@cardiology_note.png" \
  -F "prompt=Extract all echocardiogram measurements (EF, chamber dimensions, valve assessments), cardiac catheterization findings, ECG interpretation, and the recommended medication plan including specific drug names and dosages." \
  | python -m json.tool
```

### Use Case 9: Medication Reconciliation

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@med_reconciliation.png" \
  -F "prompt=Create a complete medication reconciliation list. For each medication: drug name, dose, route, frequency, indication. Also extract all comorbidities and any drug allergies." \
  | python -m json.tool
```

### Use Case 10: Radiology Report Parsing

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@radiology_report.png" \
  -F "prompt=Extract all findings from this radiology report including: anatomical locations, measurements in cm/mm, comparison with prior studies, and the radiologist's impression. Organize by body region." \
  | python -m json.tool
```

---

## 11. Using the HuggingFace Adapter Directly

If you don't want to clone the full repo, you can use the model directly from HuggingFace:

```python
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from PIL import Image

# Step 1: Configure NF4 quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Step 2: Load base model (downloads ~15.5 GB on first run)
print("Loading base model...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)

# Step 3: Load and merge LoRA adapter (downloads ~727 MB on first run)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(
    model, "sarathi-balakrishnan/Qwen2.5-VL-7B-Medical-LoRA"
)
model = model.merge_and_unload()
model.eval()

# Step 4: Load processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# Step 5: Run inference
image = Image.open("your_medical_document.png")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract all diagnoses, medications with dosages, and vital signs."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text], images=image_inputs, videos=video_inputs,
    padding=True, return_tensors="pt",
).to(model.device)

with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=1024)

generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
response = processor.decode(generated_ids, skip_special_tokens=True)
print(response)
```

> **Note:** For production use, we recommend using the base model **without** the LoRA adapter. The base Qwen2.5-VL-7B with NF4 quantization scores 83.5% on medical extraction accuracy vs 73.7% with the fine-tuned adapter. Simply skip the `PeftModel.from_pretrained()` step.

---

## 12. Production Considerations

### Performance Expectations

| Metric | Value |
|--------|-------|
| Model load time | ~90 seconds |
| Average inference latency | 15-25 seconds per image |
| Throughput | ~0.066 images/second (sequential) |
| VRAM usage (NF4 base) | 5,664 MB |
| VRAM usage (NF4 + LoRA merged) | 11,336 MB |

### Scaling

The model serves one request at a time (single worker). For higher throughput:

- **Horizontal scaling:** Run multiple container instances behind a load balancer, each with its own GPU
- **Request queueing:** Add a message queue (Redis, RabbitMQ) in front of the API for burst handling
- **Batch endpoint:** Modify `api/server.py` to accept multiple images per request

### Security

- The API has **CORS allow-all** enabled by default. Restrict `allow_origins` in `api/server.py` for production.
- No authentication is built in. Add OAuth2/API key middleware for external-facing deployments.
- **Never expose port 8000 directly to the internet** without a reverse proxy (nginx, Traefik) and TLS.
- Medical documents may contain PHI (Protected Health Information). Ensure HIPAA compliance if processing real patient data.

### Monitoring

- The `/health` endpoint returns real-time GPU status and VRAM usage
- The `/benchmark` endpoint returns all collected performance metrics
- Server logs include request-level latency for every inference call
- Docker health check runs every 60 seconds

### Recommended Architecture for Production

```
Internet -> Nginx (TLS) -> Load Balancer -> [GPU Server 1: Docker Container]
                                         -> [GPU Server 2: Docker Container]
                                         -> [GPU Server N: Docker Container]
```

---

## 13. Troubleshooting

### "CUDA out of memory"

The NF4-quantized model needs ~5.7 GB VRAM (base only) or ~11.3 GB (with LoRA merged). If you're running out of memory:

```bash
# Check what's using GPU memory
nvidia-smi

# Kill any stale Python processes using the GPU
kill $(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
```

To use less VRAM, load the base model without the LoRA adapter (5.7 GB vs 11.3 GB).

### "Model not loading" / Server hangs on startup

Model loading takes ~90 seconds. Check the server logs:
```bash
# If running directly
# Watch for "Model ready on cuda:0" message

# If running in Docker
docker logs -f <container_id>
```

### "Cannot connect to API" from Gradio

The Gradio demo connects to `http://localhost:8000`. Ensure the FastAPI server is running and healthy:
```bash
curl http://localhost:8000/health
```

### Blackwell GPU (sm_121) issues

If you see errors like `NVRTC compilation failed` or `RuntimeError: prod`:
- Ensure you're using PyTorch nightly with cu128: `pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128`
- The `blackwell_compat.py` patches should handle this automatically, but verify it's being imported

### AWQ quantization fails

AWQ is incompatible with transformers 4.57+ (`PytorchGELUTanh` was removed). Use BitsAndBytes NF4 instead, which is the default in this pipeline.

### Docker: "no NVIDIA GPU device is present"

Install the NVIDIA Container Toolkit:
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Slow inference (>30 seconds per image)

- Ensure you're using NF4 quantization (not FP16). FP16 is 2x slower.
- Large images increase latency. Resize input images to max 1024px on the longest side.
- First inference after model load is slower due to CUDA kernel compilation. Subsequent requests are faster.

---

## Quick Reference Commands

```bash
# Start everything (API + Gradio)
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000 &
python -m api.demo_ui &

# Quick health check
curl -s http://localhost:8000/health | python -m json.tool

# Extract from a document
curl -X POST http://localhost:8000/extract -F "file=@doc.png" | python -m json.tool

# Analyze with custom prompt
curl -X POST http://localhost:8000/analyze -F "file=@doc.png" -F "prompt=List all medications" | python -m json.tool

# Run automated tests
python tests/test_api.py

# Docker one-liner
docker run --gpus all -p 8000:8000 -p 7860:7860 -v $(pwd)/models:/app/models medical-vision-pipeline
```

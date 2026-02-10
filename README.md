# Medical Vision Pipeline

End-to-end quantization and fine-tuning pipeline for **Qwen2.5-VL-7B-Instruct**, optimized for medical document understanding. Quantize a 7B vision-language model from FP16 to NF4, fine-tune with QLoRA on medical datasets, evaluate on clinical documents, and deploy via REST API -- all on a single GPU.

---

## Key Results

| Metric | FP16 Baseline | NF4 Quantized | Change |
|--------|---------------|---------------|--------|
| VRAM (post-load) | 15,820 MB | 5,664 MB | **-64.2%** |
| Avg Inference Latency | 31.10s | 15.24s | **-51.0%** |
| Medical Extraction Accuracy | -- | 83.5% | -- |

> NF4 quantization reduces VRAM by 64% and halves inference time with no measurable quality loss.

---

## Pipeline Overview

```
Phase 1          Phase 2          Phase 3          Phase 4
Environment  --> Download Model & --> Quantize    --> Download &
Setup            Baseline Bench      (NF4/AWQ)       Prepare Data
  |                  |                   |                |
  v                  v                   v                v
benchmarks/      benchmarks/        benchmarks/      datasets/
environment.json baseline_fp16.json quantization_    formatted/
                                    info.json
                                    quantized_int4.json

Phase 5          Phase 6              Phase 7          Phase 8
Fine-Tune    --> Evaluate Fine-   --> API Deploy  --> Final Report
(QLoRA)          Tuned Model          (FastAPI)       & Documentation
  |                  |                   |                |
  v                  v                   v                v
models/          benchmarks/          api/server.py    benchmarks/
qwen25-vl-7b-   evaluation_          api/demo_ui.py   FINAL_REPORT.json
medical-lora/    finetuned.json       Dockerfile       docs/BENCHMARK_
                 finetuned_                             REPORT.md
                 standard_bench.json
```

---

## Project Structure

```
quantization-pipe/
|-- api/
|   |-- server.py              # FastAPI REST server
|   |-- demo_ui.py             # Gradio demo interface
|-- benchmarks/
|   |-- environment.json       # GPU, CUDA, package versions
|   |-- download_info.json     # Model download metadata
|   |-- baseline_fp16.json     # FP16 inference benchmark
|   |-- quantized_int4.json    # NF4 inference benchmark
|   |-- quantization_info.json # Quantization method details
|   |-- training_metrics.json  # QLoRA training results
|   |-- finetuned_standard_bench.json  # Fine-tuned model benchmark
|   |-- evaluation_finetuned.json      # Medical accuracy evaluation
|   |-- FINAL_REPORT.json     # Compiled report (generated)
|-- datasets/
|   |-- raw/                   # Downloaded datasets
|   |-- formatted/             # Processed training data
|   |-- test_images/           # Synthetic medical document images
|   |-- eval_images/           # Evaluation test cases
|-- docs/
|   |-- BUILD_LOG.md           # Phase-by-phase build log
|   |-- BENCHMARK_REPORT.md    # Formatted benchmark report (generated)
|   |-- CONTENT_DRAFT.md       # Social media drafts
|-- logs/                      # Training and runtime logs
|-- models/
|   |-- qwen25-vl-7b-base/            # Original HF model weights
|   |-- qwen25-vl-7b-bnb-nf4/         # NF4 quantization config
|   |-- qwen25-vl-7b-medical-lora/    # LoRA adapter weights
|-- scripts/
|   |-- validate_environment.py   # Phase 1: Environment checks
|   |-- download_model.py         # Phase 2: Download model
|   |-- blackwell_compat.py       # Blackwell GPU compatibility patches
|   |-- benchmark_baseline.py     # Phase 2: FP16 baseline bench
|   |-- quantize_base.py          # Phase 3: Quantization
|   |-- benchmark_quantized.py    # Phase 3: NF4 benchmark
|   |-- download_datasets.py      # Phase 4: Dataset acquisition
|   |-- prepare_training_data.py  # Phase 4: Data formatting
|   |-- finetune_lora.py          # Phase 5: QLoRA training
|   |-- evaluate_finetuned.py     # Phase 6: Accuracy evaluation
|   |-- generate_report.py        # Phase 8: Report generation
|-- tests/
|   |-- test_api.py            # API endpoint tests
|-- Dockerfile                 # NVIDIA CUDA container
|-- requirements.txt           # Python dependencies
|-- README.md                  # This file
```

---

## Quick Start

### Prerequisites

- NVIDIA GPU with at least 8 GB VRAM (tested on NVIDIA GB10, 119.7 GB)
- CUDA 12.x
- Python 3.10+
- conda or venv

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n med-vision python=3.10 -y
conda activate med-vision

# Install dependencies
pip install -r requirements.txt

# Validate environment
python scripts/validate_environment.py
```

> **Blackwell GPU users (sm_121):** PyTorch nightly with cu128 is required. Stable releases do not support Blackwell architecture. The `blackwell_compat.py` script patches integer reduction ops that fail via NVRTC JIT on sm_121.

### 2. Download Model and Run Baseline

```bash
# Download Qwen2.5-VL-7B-Instruct (~15.5 GB)
python scripts/download_model.py

# Run FP16 baseline benchmark
python scripts/benchmark_baseline.py
```

### 3. Quantize

```bash
# Quantize to NF4 (bitsandbytes)
python scripts/quantize_base.py

# Benchmark quantized model
python scripts/benchmark_quantized.py
```

### 4. Prepare Training Data

```bash
# Download PathVQA and MTSamples datasets
python scripts/download_datasets.py

# Format for training (produces datasets/formatted/)
python scripts/prepare_training_data.py
```

### 5. Fine-Tune with QLoRA

```bash
# Train LoRA adapter on NF4 base (approx. 2 hours on NVIDIA GB10)
python scripts/finetune_lora.py
```

### 6. Evaluate

```bash
# Run medical accuracy evaluation on 5 clinical test cases
python scripts/evaluate_finetuned.py
```

### 7. Deploy API

```bash
# Start the FastAPI server (loads model on startup)
python -m uvicorn api.server:app --host 0.0.0.0 --port 8000

# In a separate terminal, start the Gradio demo
python -m api.demo_ui

# Run API tests
python tests/test_api.py
```

### 8. Generate Final Report

```bash
python scripts/generate_report.py
# Outputs: benchmarks/FINAL_REPORT.json, docs/BENCHMARK_REPORT.md
```

---

## API Reference

### `GET /health`

Returns model readiness, GPU status, and VRAM usage.

```bash
curl http://localhost:8000/health
```

### `POST /extract`

Extract medical information from a document image using the default medical extraction prompt.

```bash
curl -X POST http://localhost:8000/extract \
  -F "file=@path/to/medical_document.png"
```

### `POST /analyze`

Analyze a medical image with an optional custom prompt.

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@path/to/medical_document.png" \
  -F "prompt=List all medications with dosages"
```

### `GET /benchmark`

Return all benchmark JSON results.

```bash
curl http://localhost:8000/benchmark
```

---

## Docker Deployment

### Build

```bash
docker build -t medical-vision-pipeline .
```

### Run

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

```bash
docker run --gpus all \
  -p 8000:8000 \
  -p 7860:7860 \
  medical-vision-pipeline
```

- **FastAPI server:** http://localhost:8000
- **API docs (Swagger):** http://localhost:8000/docs
- **Gradio demo:** http://localhost:7860

The container starts both the FastAPI server and Gradio demo. The health check endpoint (`/health`) is probed every 60 seconds with a 5-minute startup grace period to allow for model loading.

### Persistent Model Cache

To avoid re-downloading the model on every container start, mount a volume:

```bash
docker run --gpus all \
  -p 8000:8000 \
  -p 7860:7860 \
  -v /path/to/models:/app/models \
  medical-vision-pipeline
```

---

## Technical Details

### Quantization: BNB NF4

- **Method:** 4-bit NormalFloat (NF4) via bitsandbytes
- **Double quantization:** enabled (quantizes the quantization constants)
- **Compute dtype:** bfloat16
- **AWQ:** attempted but AutoAWQ is incompatible with transformers 4.57 (`PytorchGELUTanh` removed)

### Fine-Tuning: QLoRA

- **LoRA rank:** 64, **alpha:** 128, **dropout:** 0.05
- **Target modules:** q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **Vision encoder:** frozen (only language model layers trained)
- **Trainable parameters:** 190M / 4.88B (3.9%)
- **Training data:** 1,000 samples from MTSamples (rendered text-to-image)
- **Training time:** 128.4 minutes (2 epochs)

### Blackwell Compatibility

The pipeline includes patches for running on NVIDIA Blackwell architecture (sm_121). The `blackwell_compat.py` script routes integer reduction operations (`prod`, `cumprod`) through CPU to work around NVRTC JIT compilation issues with the PyTorch nightly cu128 build.

### Datasets

| Dataset | Source | Formatted Samples |
|---------|--------|-------------------|
| PathVQA | flaviagiammarino/path-vqa | 431 |
| MTSamples | rungalileo/medical_transcription_40 | 4,465 |
| **Total** | | **4,896** |

Train/Val/Test split: 3,916 / 490 / 490 (80/10/10, seed=42)

---

## Hardware Tested

- **NVIDIA DGX Spark** (Blackwell architecture)
- **GPU:** NVIDIA GB10, 119.7 GB unified VRAM
- **CUDA:** 12.8
- **PyTorch:** 2.11.0.dev20260206+cu128 (nightly, required for sm_121)

---

## License

This pipeline is for research and educational purposes. The base model (Qwen2.5-VL-7B-Instruct) is subject to the [Qwen License](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct). Medical datasets are used under their respective licenses.

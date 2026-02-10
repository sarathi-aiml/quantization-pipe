# Medical Vision Model Pipeline — Claude Code CLI Agent Instructions

> **IMPORTANT**: This file contains INSTRUCTIONS ONLY. You are the agent — write all code yourself. Do not ask the user for code. Figure out the implementation details, handle errors, and validate every step before proceeding to the next phase.

** Note** : the project was intrupted in the middle while runing below tasks. check where it left off and start from there , make sure to validate previously done tasks. 

---

## Project Goal

Build a complete, production-grade pipeline that takes Qwen2.5-VL-7B-Instruct from HuggingFace, quantizes it, fine-tunes it for medical document understanding, benchmarks every stage, deploys it as an API, and generates comprehensive documentation. This showcases end-to-end MLOps capability: model optimization, domain adaptation, evaluation, and deployment.

---

## Global Rules (Follow These Always)

1. **Validate every step** — after each action, verify it worked. Check files exist, outputs are correct, models load, scripts run without errors.
2. **Benchmark everything** — every model variant (base, quantized, fine-tuned) must have recorded metrics: VRAM usage, inference latency, load time, and model size on disk.
3. **Save all metrics as JSON** — store every benchmark result in a `benchmarks/` directory as structured JSON files so they can be compared later.
4. **Handle errors gracefully** — if something fails (e.g., a specific quantization method isn't supported for this architecture), implement a fallback approach and document why.
5. **Document as you go** — maintain a running `docs/BUILD_LOG.md` that logs what you did at each phase, what worked, what didn't, decisions made, and actual results.
6. **Production quality** — all code must have proper error handling, type hints, docstrings, and logging. No quick hacks.
7. **Clean project structure** — organize code into logical directories: `scripts/`, `api/`, `models/`, `datasets/`, `benchmarks/`, `docs/`, `tests/`.

---

## Phase 1: Environment Setup

### What To Do
- Create the project directory structure with all necessary subdirectories.
- Create a `requirements.txt` with all dependencies needed for the full pipeline: model loading, quantization (AutoAWQ, bitsandbytes, auto-gptq), fine-tuning (peft, trl, transformers, accelerate), datasets, evaluation metrics (rouge-score, nltk, scikit-learn), API serving (fastapi, uvicorn, python-multipart), demo UI (gradio), and utilities (psutil, GPUtil, matplotlib, seaborn, huggingface_hub).
- Install all dependencies.
- Write and run a validation script that confirms: Python 3.10+, PyTorch with CUDA available, GPU name and VRAM, and that all critical libraries import successfully (transformers, peft, trl, bitsandbytes, fastapi, gradio).
- Save the environment info (Python version, PyTorch version, CUDA version, GPU name, total VRAM, installed package versions) to `benchmarks/environment.json`.
- Start `docs/BUILD_LOG.md` with a header, timestamp, and environment details.

### Validation Gate
- All pip installs succeed.
- Validation script confirms CUDA is available and all imports work.
- `benchmarks/environment.json` exists and is valid JSON.
- `docs/BUILD_LOG.md` exists with environment details.

**Do not proceed to Phase 2 until all validations pass.**

---

## Phase 2: Download Base Model & Baseline Benchmarking

### Step 2.1: Download Model
- Download `Qwen/Qwen2.5-VL-7B-Instruct` from HuggingFace to `models/qwen25-vl-7b-base/`.
- Use `huggingface_hub` snapshot_download or equivalent. Skip non-essential files like README, LICENSE.
- After download, calculate and print the total model size on disk in GB.
- Log the download details (model ID, size, download time) to the build log.

### Step 2.2: Baseline Benchmark
- Load the model in FP16 precision with `device_map="auto"`. Use flash attention 2 if available, otherwise fall back to default attention.
- Create 5 synthetic medical document test images programmatically. These should be images of rendered text that simulate medical documents — include things like: a patient diagnosis with medications and vitals, a radiology report with findings, a prescription with drug names and dosages, lab results with values and units, and a discharge summary with multiple conditions. Make these realistic with proper medical terminology.
- Save these test images to `datasets/test_images/` — they will be reused in every benchmark phase for consistency.
- Run inference on all 5 images with medical extraction prompts. For each, measure: inference latency (with CUDA synchronization for accurate timing), and capture the model's response.
- Record GPU memory usage: pre-load, post-load, and peak during inference.
- Save all results to `benchmarks/baseline_fp16.json` including: model name, precision, load time, VRAM metrics, per-sample latency and responses, aggregate latency stats (avg, min, max, p50).
- Print a clean summary table of key metrics.
- Unload the model and clear GPU cache after benchmarking.
- Update the build log with baseline results.

### Validation Gate
- Model files exist in `models/qwen25-vl-7b-base/`.
- 5 test images exist in `datasets/test_images/`.
- `benchmarks/baseline_fp16.json` exists with valid metrics (latencies > 0, VRAM > 0).
- Model responses are non-empty and contain medical terminology.
- Build log is updated.

**Do not proceed to Phase 3 until all validations pass.**

---

## Phase 3: Quantization of Base Model

### Step 3.1: Quantize
- Attempt AWQ INT4 quantization first using AutoAWQ. Configure with: zero_point=True, q_group_size=128, w_bit=4.
- **If AWQ fails** (which is likely for vision-language models), fall back to bitsandbytes NF4 quantization. This is expected — document the failure reason in the build log and proceed with NF4.
- If using AWQ: save the quantized model to `models/qwen25-vl-7b-awq-int4/`.
- If using BNB NF4: save a configuration file to `models/qwen25-vl-7b-bnb-nf4/` documenting the quantization config used, since BNB models are loaded on-the-fly from the base model.
- Record the quantization method used, time taken, and any errors encountered in `benchmarks/quantization_info.json`.
- Calculate and compare model size on disk (for AWQ) or expected VRAM reduction (for BNB).
- Update build log.

### Step 3.2: Benchmark Quantized Model
- Load the quantized model (whichever method succeeded).
- Run inference on the **exact same 5 test images** from Phase 2 with the **exact same prompts**.
- Collect the same metrics: VRAM, latency, responses.
- Save results to `benchmarks/quantized_int4.json`.
- Generate a comparison table: FP16 baseline vs INT4 quantized — showing VRAM reduction %, latency change %, and whether responses are still coherent.
- Print the comparison.
- Unload model, clear cache.
- Update build log with quantization results and comparison.

### Validation Gate
- Quantization config/model files exist.
- `benchmarks/quantization_info.json` exists with method and timing.
- `benchmarks/quantized_int4.json` exists with valid metrics.
- Quantized model VRAM is lower than FP16 baseline.
- Model still produces coherent medical responses (not gibberish).
- Build log is updated.

**Do not proceed to Phase 4 until all validations pass.**

---

## Phase 4: Dataset Acquisition & Preparation

### Step 4.1: Download Datasets
Download these freely available medical datasets from HuggingFace (no credentials or special access required):

1. **ROCO** (Radiology Objects in COntext) — search HuggingFace for a ROCO radiology dataset. It contains radiology images paired with captions. Download up to 3000 samples.
2. **PubMedVision** — search for `FreedomIntelligence/PubMedVision` on HuggingFace. It contains medical image-text pairs from PubMed. Download up to 2000 samples from the alignment split.
3. **MTSamples** — search for a medical transcription samples dataset on HuggingFace. It contains transcribed medical reports across specialties. Download all available samples.

For each dataset:
- If the download fails, log the error and try an alternative medical dataset. Don't let one failed download block the pipeline.
- Save raw datasets to `datasets/raw/{dataset_name}/`.
- Log download status, sample counts, and any failures.

Save a summary to `datasets/download_summary.json` with counts per dataset.

### Step 4.2: Format for Training
Convert all downloaded data into Qwen2.5-VL instruction-tuning format. The format should be a list of conversation examples, each with a user message containing an image and a medical extraction prompt, and an assistant message containing the expected response.

Specific formatting instructions:
- **For ROCO/PubMedVision** (datasets that already have images): save images to `datasets/formatted/images/`, pair each image with a randomly selected medical extraction prompt (create a variety of prompts like "Extract all medical terms from this image", "What diagnoses are shown?", "List all medications and dosages", etc.), and use the dataset's caption/answer as the assistant response.
- **For MTSamples** (text-only dataset): render the medical text onto images to simulate scanned documents. Create white background images with the medical text drawn on them using a monospace font. This gives the model practice reading medical text from document images.
- Discard any samples where the image is missing or the text is too short (< 50 characters).
- Create a pool of at least 8-10 diverse medical extraction prompts and randomly assign them to samples.

After formatting:
- Shuffle all samples with a fixed random seed (42) for reproducibility.
- Split into train/val/test at 80/10/10 ratio.
- Save as `datasets/formatted/train.json`, `datasets/formatted/val.json`, `datasets/formatted/test.json`.
- Save split summary to `datasets/formatted/data_summary.json`.
- Update build log with dataset statistics.

### Validation Gate
- At least 2 of the 3 datasets downloaded successfully.
- Formatted images exist in `datasets/formatted/images/`.
- `train.json`, `val.json`, `test.json` all exist and are valid JSON.
- `data_summary.json` shows: training set > 500 samples (ideally 2000+), val and test sets are non-empty.
- Spot-check: load 3 random training samples, verify each has an image path that exists and a non-empty prompt and response.
- Build log updated.

**Do not proceed to Phase 5 until all validations pass.**

---

## Phase 5: LoRA Fine-Tuning

### Step 5.1: Configure and Run Fine-Tuning
Set up LoRA fine-tuning with these specifications:

**Model Loading:**
- Load the base Qwen2.5-VL-7B with bitsandbytes NF4 quantization (4-bit) for memory-efficient training (QLoRA approach).
- Use bfloat16 compute dtype.
- Prepare the model for k-bit training using peft utilities.

**What to Freeze and What to Train:**
- FREEZE the entire vision encoder — do not train any parameters with "visual" in the parameter name.
- TRAIN the LLM backbone using LoRA adapters on the attention and MLP projection layers: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj.

**LoRA Configuration:**
- Rank: 64
- Alpha: 128
- Dropout: 0.05
- Bias: none
- Task type: CAUSAL_LM

**Training Configuration:**
- Epochs: 2 (increase to 3 if loss isn't decreasing well)
- Batch size: 2 (reduce to 1 if OOM)
- Gradient accumulation steps: 8 (so effective batch size = 16)
- Learning rate: 1e-4
- Warmup ratio: 0.05
- Weight decay: 0.01
- Use gradient checkpointing for memory efficiency
- Max sequence length: 1024
- Save checkpoints every 200 steps, keep last 3
- Evaluate on validation set every 200 steps
- Use bfloat16 mixed precision

**Training Execution:**
- Use SFTTrainer from the `trl` library, or a custom training loop if SFTTrainer doesn't work well with the VL model's multimodal inputs.
- If SFTTrainer has issues with the image-text format, implement a custom training loop using the standard transformers Trainer or a manual PyTorch training loop. Handle this gracefully.
- Print trainable parameter count and percentage before training starts.
- Monitor and log training loss.

**After Training:**
- Save the LoRA adapter weights to `models/qwen25-vl-7b-medical-lora/`.
- Save the processor/tokenizer alongside the adapter.
- Save training metrics to `benchmarks/training_metrics.json`: total training time, final loss, epochs completed, learning rate used, trainable parameter count and percentage, training samples count.
- Update build log with training results.

### Validation Gate
- LoRA adapter files exist in `models/qwen25-vl-7b-medical-lora/` (should contain adapter_config.json, adapter_model files).
- `benchmarks/training_metrics.json` exists with valid metrics.
- Training loss decreased from start to end (final loss < initial loss).
- Trainable parameter percentage should be roughly 1-5% of total parameters.
- Build log updated.

**Do not proceed to Phase 6 until all validations pass.**

---

## Phase 6: Evaluate Fine-Tuned Model

### Step 6.1: Create Medical Evaluation Test Suite
Create a dedicated evaluation test suite with 3-5 carefully crafted medical document test cases. Each test case should have:
- A medical document image (rendered text simulating a real medical document).
- A list of expected medical terms that should be extracted (e.g., drug names, diagnoses, procedures).
- A list of expected clinical values (e.g., dosages, lab values, vital signs with units).

Make these test cases challenging and realistic. Include: abbreviations (BID, QD, PRN, TID), medical abbreviations (COPD, DM, CKD, MI), drug names with dosages, lab values with units and reference flags (H/L), and clinical measurements.

These must be DIFFERENT from the 5 synthetic test images used in Phase 2/3 — this is a separate evaluation set specifically for measuring medical extraction accuracy.

### Step 6.2: Evaluate Both Models
Run evaluation on two model variants using the exact same test suite:

1. **Base model (quantized NF4, no fine-tuning)** — this is the control.
2. **Fine-tuned model (LoRA adapter loaded on top of NF4 base)** — merge the LoRA adapter for faster inference using `merge_and_unload()`.

For each model, on each test case, measure:
- **Medical term extraction accuracy**: what percentage of expected medical terms appear in the model's response (case-insensitive matching).
- **Clinical value extraction accuracy**: what percentage of expected values (dosages, lab numbers) appear in the response.
- **Combined accuracy**: average of term and value accuracy.
- **Inference latency** with CUDA synchronization.
- **GPU memory usage**.

### Step 6.3: Also Benchmark on Original Test Images
Load the fine-tuned model and run inference on the **same 5 test images from Phase 2** to get latency and VRAM numbers that are directly comparable to the FP16 and quantized benchmarks.

### Step 6.4: Compile All Results
- Save detailed evaluation results to `benchmarks/evaluation_finetuned.json`.
- Save the Phase 2 test image results to `benchmarks/finetuned_standard_bench.json`.
- Print a comprehensive comparison table showing all three stages side by side: Base FP16 → Quantized INT4 → Fine-Tuned LoRA, with columns for VRAM, latency, and medical accuracy (where applicable).
- Update build log.

### Validation Gate
- `benchmarks/evaluation_finetuned.json` exists with per-test-case results for both models.
- `benchmarks/finetuned_standard_bench.json` exists.
- Fine-tuned model medical term accuracy ≥ base model accuracy (improvement expected, even if small).
- All responses are coherent (no gibberish, no empty outputs).
- Comparison table is printed and logged.
- Build log updated.

**Do not proceed to Phase 7 until all validations pass.**

---

## Phase 7: API Deployment

### Step 7.1: Build FastAPI Server
Create a production-grade FastAPI application with these endpoints:

- **POST /extract** — accepts an uploaded image file, runs the fine-tuned model to extract medical terms, diagnoses, medications, lab values. Returns structured JSON with: extracted content, inference latency, model info.
- **POST /analyze** — accepts an uploaded image file and an optional custom prompt. Runs inference with the provided prompt (or a default medical analysis prompt). Returns the same structured response.
- **GET /health** — returns server health status: model loaded (bool), GPU available, GPU name, current VRAM usage.
- **GET /benchmark** — returns all benchmark JSON files compiled into a single response so users can see the full pipeline performance data.

Server requirements:
- Load the fine-tuned model (base + LoRA merged) on startup with NF4 quantization.
- Add CORS middleware for cross-origin access.
- Proper error handling — return 503 if model isn't loaded yet, 400 for bad inputs, proper error messages.
- Use Pydantic response models for all endpoints.
- Log all requests with timestamps and latencies.

Save as `api/server.py`.

### Step 7.2: Build Gradio Demo UI
Create a Gradio demo interface that connects to the FastAPI server:

- **Tab 1: Medical Term Extraction** — image upload → extract button → displays extracted medical information with latency.
- **Tab 2: Custom Analysis** — image upload + custom prompt textbox → analyze button → displays results.
- **Tab 3: Benchmarks** — button to load and display all benchmark data from the API.

Include a title, description explaining what the model is and how it was built (Qwen2.5-VL-7B + LoRA fine-tuning + INT4 quantization).

Save as `api/demo_ui.py`.

### Step 7.3: Create Dockerfile
Create a Dockerfile that:
- Uses NVIDIA CUDA base image.
- Installs Python dependencies.
- Copies the project.
- Exposes ports 8000 (API) and 7860 (Gradio).
- Runs both the API server and Gradio demo.

### Step 7.4: Create API Test Script
Write a test script that:
- Starts the API server.
- Waits for the health endpoint to confirm model is loaded.
- Creates a test medical document image.
- Calls `/extract` and `/analyze` endpoints.
- Verifies responses are valid and contain medical information.
- Prints test results.

Save as `tests/test_api.py`.

### Step 7.5: Run and Validate
- Start the API server.
- Run the test script.
- Verify all endpoints return expected responses.
- Update build log.

### Validation Gate
- `api/server.py`, `api/demo_ui.py`, `Dockerfile`, `tests/test_api.py` all exist.
- API server starts without errors.
- `/health` returns `model_loaded: true`.
- `/extract` returns medical terms from a test image.
- `/analyze` returns a coherent medical analysis.
- `/benchmark` returns compiled benchmark data.
- Build log updated.

**Do not proceed to Phase 8 until all validations pass.**

---

## Phase 8: Final Documentation & Report

### Step 8.1: Generate Comprehensive Benchmark Report
Create a script that reads ALL benchmark JSON files and generates a final report. Save as `scripts/generate_report.py`.

The report should compile into `benchmarks/FINAL_REPORT.json` containing:
- Pipeline description.
- Environment info (GPU, CUDA, Python, PyTorch versions).
- Stage-by-stage comparison table with all metrics.
- Training details (duration, loss, LoRA config, parameter efficiency).
- Quantization details (method, size reduction).
- Medical accuracy evaluation results (base vs fine-tuned).
- API deployment status.

Also generate a human-readable `docs/BENCHMARK_REPORT.md` with:
- Formatted comparison tables.
- Key findings and insights.
- Performance highlights suitable for sharing on social media.

### Step 8.2: Create Project README
Generate `README.md` at the project root with:
- Project title and one-line description.
- Pipeline overview diagram (using text/ASCII or mermaid notation).
- Key results table pulled from benchmarks.
- How to run the full pipeline.
- How to start the API server and demo.
- Project structure overview.
- Technical details: model architecture, quantization method, LoRA configuration, datasets used.
- Docker deployment instructions.

### Step 8.3: Create Social Media Content Draft
Generate `docs/CONTENT_DRAFT.md` with a draft LinkedIn/Twitter post about this project. The post should:
- Use a neutral, technical voice (no first person, no personal attribution).
- Lead with the most impressive metric (e.g., VRAM reduction %, accuracy improvement, latency).
- Mention the specific tools/techniques: Qwen2.5-VL-7B, AWQ/NF4 quantization, LoRA fine-tuning, medical domain adaptation.
- Reference running on a home GPU cluster (connects to existing infrastructure content).
- Include 2-3 key benchmark numbers.
- Be concise — under 200 words for LinkedIn, under 280 characters for Twitter.

### Step 8.4: Final Project Validation
Run a final validation sweep:
- Verify all expected files exist in the project.
- Verify all benchmark JSON files are valid and contain data.
- Verify the build log has entries for all 8 phases.
- Verify the README is comprehensive.
- Print a final project summary.

### Validation Gate
- `benchmarks/FINAL_REPORT.json` exists with all pipeline metrics.
- `docs/BENCHMARK_REPORT.md` exists with formatted tables.
- `README.md` exists at project root with all required sections.
- `docs/CONTENT_DRAFT.md` exists with social media post drafts.
- `docs/BUILD_LOG.md` has entries for all 8 phases.
- All benchmark JSON files in `benchmarks/` are valid.

---

## Final Project Structure (Expected)

```
medical-vision-pipeline/
├── README.md
├── requirements.txt
├── Dockerfile
├── run_pipeline.sh
├── scripts/
│   ├── download_model.py
│   ├── benchmark_baseline.py
│   ├── quantize_base.py
│   ├── benchmark_quantized.py
│   ├── download_datasets.py
│   ├── prepare_training_data.py
│   ├── finetune_lora.py
│   ├── evaluate_finetuned.py
│   └── generate_report.py
├── api/
│   ├── server.py
│   └── demo_ui.py
├── tests/
│   └── test_api.py
├── models/
│   ├── qwen25-vl-7b-base/
│   ├── qwen25-vl-7b-awq-int4/ or qwen25-vl-7b-bnb-nf4/
│   └── qwen25-vl-7b-medical-lora/
├── datasets/
│   ├── raw/
│   ├── test_images/
│   └── formatted/
│       ├── images/
│       ├── train.json
│       ├── val.json
│       ├── test.json
│       └── data_summary.json
├── benchmarks/
│   ├── environment.json
│   ├── baseline_fp16.json
│   ├── quantization_info.json
│   ├── quantized_int4.json
│   ├── training_metrics.json
│   ├── evaluation_finetuned.json
│   ├── finetuned_standard_bench.json
│   └── FINAL_REPORT.json
├── docs/
│   ├── BUILD_LOG.md
│   ├── BENCHMARK_REPORT.md
│   └── CONTENT_DRAFT.md
└── logs/
```

---

## Troubleshooting Guide (For the Agent)

- **OOM during fine-tuning**: Reduce batch size to 1, increase gradient accumulation to 16. If still OOM, reduce max_seq_length to 512.
- **AWQ quantization fails for VL model**: This is expected. Fall back to bitsandbytes NF4. Document the failure.
- **Dataset download fails**: Try alternative datasets. Search HuggingFace for: "medical image", "radiology", "clinical notes", "medical VQA". The pipeline should work with whatever medical data is available.
- **Flash Attention 2 not available**: Fall back to default attention. Install flash-attn if possible, but don't block on it.
- **SFTTrainer incompatible with multimodal inputs**: Implement a custom training loop. The key challenge is handling image inputs alongside text — process them through the model's processor first.
- **Low accuracy after fine-tuning**: Check that training loss decreased. If not, lower learning rate to 2e-5 or 5e-5. Ensure the training data format matches what the model expects.
- **Import errors**: Reinstall the specific package. Check version compatibility.

---

## Key Reminder

**You are the agent. Write all code yourself. This document tells you WHAT to do and WHAT to validate. The HOW is your job. Make it production quality.**

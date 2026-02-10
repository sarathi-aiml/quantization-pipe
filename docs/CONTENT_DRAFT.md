# Social Media Content Drafts

Drafts for sharing the Medical Vision Pipeline project results.

---

## LinkedIn Post (under 200 words)

**Quantizing a 7B Vision-Language Model for Medical Document Understanding**

NF4 quantization reduced VRAM usage by 64.2% and cut inference latency in half for Qwen2.5-VL-7B-Instruct, a vision-language model adapted for medical document extraction.

The full pipeline -- quantization, QLoRA fine-tuning, evaluation, and REST API deployment -- was built and benchmarked on an NVIDIA DGX Spark with a Blackwell GB10 GPU.

Key results:
- VRAM: 15.8 GB (FP16) to 5.7 GB (NF4) -- 64.2% reduction
- Inference: 31.1s to 15.2s average latency -- 51% faster
- Medical accuracy: 83.5% on clinical documents (ICU flowsheets, cardiology notes, lab panels, operative notes)
- QLoRA training: 190M trainable parameters (3.9%), 43% loss reduction in 2 hours

The NF4-quantized base model outperformed the fine-tuned variant on medical extraction, demonstrating that Qwen2.5-VL-7B already has strong clinical document understanding out of the box. Fine-tuning with domain-matched structured clinical data (rather than rendered transcription text) would likely close this gap.

Deployed as a FastAPI service with a Gradio demo, containerized with Docker.

#MachineLearning #MedicalAI #Quantization #VisionLanguageModels #NF4 #LoRA #Qwen

---

## Twitter/X Post (under 280 characters)

NF4 quantization on Qwen2.5-VL-7B for medical doc extraction: 64% less VRAM (15.8 GB to 5.7 GB), 51% faster inference. 83.5% accuracy on clinical documents. Full pipeline on a DGX Spark -- quantize, fine-tune with QLoRA, deploy via FastAPI.

---

## Twitter/X Thread (alternative, 3 posts)

**Post 1/3:**
Built an end-to-end pipeline: quantize Qwen2.5-VL-7B-Instruct to NF4, fine-tune with QLoRA, evaluate on medical documents, deploy via REST API.

Result: 64.2% VRAM reduction, 51% faster inference, 83.5% medical extraction accuracy.

**Post 2/3:**
The NF4-quantized base model (no fine-tuning) scored 83.5% on challenging clinical docs -- ICU flowsheets, cardiology consults, complex lab panels. Fine-tuning on 1K MTSamples with QLoRA (rank 64, 190M params, 2 hours) reduced training loss 43% but did not improve eval accuracy due to domain mismatch.

**Post 3/3:**
Pipeline runs on an NVIDIA DGX Spark (Blackwell GB10). Required PyTorch nightly for sm_121 support and custom patches for NVRTC JIT.

Served via FastAPI + Docker, with Gradio demo. All benchmarks published.

Stack: bitsandbytes NF4, PEFT/LoRA, transformers 4.57, FastAPI, Gradio.

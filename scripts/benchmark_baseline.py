"""Baseline FP16 benchmark for Qwen2.5-VL-7B-Instruct on synthetic medical images."""

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

# Apply Blackwell GPU compatibility patches before any model loading
sys.path.insert(0, str(Path(__file__).parent))
from blackwell_compat import apply_blackwell_patches
apply_blackwell_patches()

from PIL import Image, ImageDraw, ImageFont
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-base"
TEST_IMAGES_DIR = PROJECT_ROOT / "datasets" / "test_images"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

# Medical document content for 5 synthetic test images
MEDICAL_DOCUMENTS = [
    {
        "filename": "patient_diagnosis.png",
        "title": "PATIENT ASSESSMENT & DIAGNOSIS",
        "content": [
            "Patient: John M. Richardson  DOB: 03/15/1958  MRN: 4827391",
            "Date of Visit: 01/22/2026  Provider: Dr. Sarah Chen, MD",
            "",
            "CHIEF COMPLAINT: Chest pain and shortness of breath x 3 days",
            "",
            "VITAL SIGNS:",
            "  BP: 158/94 mmHg    HR: 102 bpm    RR: 22/min",
            "  Temp: 98.6 F       SpO2: 93% on RA    Weight: 198 lbs",
            "",
            "ASSESSMENT:",
            "  1. Acute coronary syndrome, NSTEMI - Troponin I 2.4 ng/mL",
            "  2. Hypertension, uncontrolled",
            "  3. Type 2 Diabetes Mellitus (HbA1c 8.2%)",
            "  4. Chronic Kidney Disease, Stage 3 (eGFR 42 mL/min)",
            "",
            "MEDICATIONS ORDERED:",
            "  - Aspirin 325 mg PO STAT, then 81 mg PO QD",
            "  - Heparin 5000 units IV bolus, then 1000 units/hr",
            "  - Metoprolol 25 mg PO BID",
            "  - Lisinopril 10 mg PO QD",
            "  - Metformin 1000 mg PO BID",
            "  - Atorvastatin 80 mg PO QHS",
        ],
        "prompt": "Extract all medical diagnoses, medications with dosages, and vital signs from this patient assessment document.",
    },
    {
        "filename": "radiology_report.png",
        "title": "RADIOLOGY REPORT — CHEST CT WITH CONTRAST",
        "content": [
            "Patient: Maria L. Santos  DOB: 07/28/1972  MRN: 5913847",
            "Study Date: 01/20/2026  Radiologist: Dr. James Park, MD",
            "Exam: CT Chest with IV Contrast  Technique: Helical acquisition",
            "",
            "CLINICAL INDICATION: Persistent cough x 6 weeks, hemoptysis,",
            "  weight loss. R/O malignancy.",
            "",
            "FINDINGS:",
            "  LUNGS: 3.2 x 2.8 cm spiculated mass in the right upper lobe",
            "    (series 4, image 87). Ipsilateral hilar lymphadenopathy",
            "    measuring 1.8 cm. Left lung clear. No pleural effusion.",
            "",
            "  MEDIASTINUM: Subcarinal lymph node enlarged at 2.1 cm.",
            "    No pericardial effusion. Heart size normal.",
            "",
            "  BONES: Lytic lesion in T8 vertebral body, 1.2 cm,",
            "    suspicious for metastatic disease.",
            "",
            "IMPRESSION:",
            "  1. Right upper lobe mass highly suspicious for primary",
            "     bronchogenic carcinoma. LUNG-RADS 4B.",
            "  2. Ipsilateral hilar and subcarinal lymphadenopathy",
            "     suggesting nodal metastasis (N2 disease).",
            "  3. T8 lytic lesion concerning for osseous metastasis.",
            "",
            "RECOMMENDATION: PET/CT and tissue biopsy recommended.",
        ],
        "prompt": "List all findings from this radiology report including measurements, locations, and the radiologist's impression.",
    },
    {
        "filename": "prescription.png",
        "title": "PRESCRIPTION / MEDICATION ORDER",
        "content": [
            "Physician: Dr. Emily Watson, MD   DEA: BW4837291",
            "Clinic: Valley Internal Medicine   Ph: (555) 234-5678",
            "Patient: Robert K. Thompson   DOB: 11/03/1965",
            "Date: 01/21/2026",
            "",
            "Rx 1: Lisinopril 20 mg tablets",
            "  Sig: Take 1 tablet by mouth once daily in the morning",
            "  Disp: #30    Refills: 5",
            "",
            "Rx 2: Amlodipine 5 mg tablets",
            "  Sig: Take 1 tablet by mouth once daily",
            "  Disp: #30    Refills: 5",
            "",
            "Rx 3: Metformin 500 mg tablets",
            "  Sig: Take 1 tablet by mouth twice daily with meals",
            "  Disp: #60    Refills: 3",
            "",
            "Rx 4: Atorvastatin 40 mg tablets",
            "  Sig: Take 1 tablet by mouth at bedtime",
            "  Disp: #30    Refills: 5",
            "",
            "Rx 5: Pantoprazole 40 mg delayed-release tablets",
            "  Sig: Take 1 tablet by mouth once daily 30 min before",
            "        breakfast",
            "  Disp: #30    Refills: 2",
            "",
            "ALLERGIES: Penicillin (rash), Sulfa drugs (anaphylaxis)",
        ],
        "prompt": "Extract all prescribed medications with their dosages, directions, quantities, and refill counts from this prescription.",
    },
    {
        "filename": "lab_results.png",
        "title": "LABORATORY RESULTS — COMPREHENSIVE METABOLIC PANEL",
        "content": [
            "Patient: Angela D. Foster   DOB: 09/14/1980   MRN: 6284015",
            "Collection: 01/19/2026 07:30   Reported: 01/19/2026 11:45",
            "Ordering: Dr. Michael Rivera, MD   Status: FINAL",
            "",
            "TEST                    RESULT    UNITS       REFERENCE     FLAG",
            "--------------------------------------------------------------",
            "Glucose, Fasting        142       mg/dL       70-100        H",
            "BUN                      28       mg/dL       7-20          H",
            "Creatinine               1.8      mg/dL       0.6-1.2       H",
            "eGFR                     38       mL/min      >60           L",
            "Sodium                  139       mEq/L       136-145",
            "Potassium                5.4      mEq/L       3.5-5.1       H",
            "Chloride                101       mEq/L       98-106",
            "CO2                      20       mEq/L       23-29         L",
            "Calcium                  9.2      mg/dL       8.5-10.5",
            "Total Protein            6.8      g/dL        6.0-8.3",
            "Albumin                  3.1      g/dL        3.5-5.5       L",
            "Bilirubin, Total         0.9      mg/dL       0.1-1.2",
            "Alk Phos                 95       U/L         44-147",
            "AST (SGOT)               42       U/L         10-40         H",
            "ALT (SGPT)               58       U/L         7-56          H",
            "HbA1c                    9.1      %           <5.7          H",
            "",
            "CRITICAL: Potassium 5.4 — Provider notified at 12:02.",
        ],
        "prompt": "Extract all lab values with their results, units, reference ranges, and flags from this laboratory report.",
    },
    {
        "filename": "discharge_summary.png",
        "title": "DISCHARGE SUMMARY",
        "content": [
            "Patient: William T. Baker   DOB: 05/22/1950   MRN: 3751962",
            "Admit: 01/15/2026   Discharge: 01/21/2026   LOS: 6 days",
            "Attending: Dr. Lisa Nguyen, MD — Hospitalist Service",
            "",
            "ADMISSION DIAGNOSES:",
            "  1. Community-acquired pneumonia (CAP), right lower lobe",
            "  2. Acute on chronic systolic heart failure (EF 30%)",
            "  3. Atrial fibrillation with RVR",
            "  4. COPD exacerbation",
            "",
            "HOSPITAL COURSE:",
            "  Treated with IV Ceftriaxone 1g Q24H and Azithromycin",
            "  500 mg QD x 5 days. Diuresis with Furosemide 40 mg IV",
            "  BID. Rate control with Diltiazem drip, transitioned to",
            "  Metoprolol Succinate 50 mg PO QD. Prednisone taper for",
            "  COPD: 40 mg x 5d, 20 mg x 5d, 10 mg x 5d.",
            "",
            "DISCHARGE MEDICATIONS:",
            "  - Metoprolol Succinate 50 mg PO QD",
            "  - Furosemide 40 mg PO BID",
            "  - Potassium Chloride 20 mEq PO BID",
            "  - Apixaban 5 mg PO BID",
            "  - Lisinopril 5 mg PO QD",
            "  - Albuterol MDI 2 puffs Q4-6H PRN",
            "  - Prednisone taper (see above)",
            "  - Levofloxacin 750 mg PO QD x 3 more days",
            "",
            "FOLLOW-UP: PCP in 1 week. Cardiology in 2 weeks.",
            "  Repeat CXR in 6 weeks. Echo in 3 months.",
        ],
        "prompt": "Extract all diagnoses, medications with dosages, and follow-up instructions from this discharge summary.",
    },
]


def create_medical_document_image(doc: dict[str, Any], output_path: Path) -> None:
    """Render a medical document as a realistic image."""
    width, height = 900, 1100
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to use a monospace font, fall back to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
    except OSError:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 16)
            body_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 13)
        except OSError:
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()

    # Draw border
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(0, 0, 0), width=2)

    # Draw title
    y = 40
    draw.text((40, y), doc["title"], fill=(0, 0, 128), font=title_font)
    y += 30
    draw.line([(40, y), (width - 40, y)], fill=(0, 0, 0), width=1)
    y += 15

    # Draw content lines
    for line in doc["content"]:
        draw.text((40, y), line, fill=(0, 0, 0), font=body_font)
        y += 18
        if y > height - 60:
            break

    img.save(output_path, "PNG")
    logger.info(f"Created test image: {output_path.name}")


def get_gpu_memory_mb() -> dict[str, float]:
    """Get current GPU memory usage in MB."""
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024**2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024**2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024**2), 2),
    }


def main() -> None:
    # Step 1: Create test images
    TEST_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Creating 5 synthetic medical document test images...")
    for doc in MEDICAL_DOCUMENTS:
        create_medical_document_image(doc, TEST_IMAGES_DIR / doc["filename"])
    logger.info(f"Test images saved to {TEST_IMAGES_DIR}")

    # Step 2: Load model
    logger.info("Loading Qwen2.5-VL-7B-Instruct in FP16...")
    torch.cuda.reset_peak_memory_stats()
    pre_load_mem = get_gpu_memory_mb()

    load_start = time.time()

    # Load model with explicit max_memory to avoid accelerate underestimating
    # unified memory on DGX Spark / Blackwell systems
    total_vram = torch.cuda.get_device_properties(0).total_memory
    max_memory = {0: int(total_vram * 0.9), "cpu": "32GB"}

    load_kwargs = {
        "dtype": torch.float16,
        "device_map": "auto",
        "max_memory": max_memory,
    }

    # Try flash_attention_2, fall back to default
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(MODEL_DIR),
            attn_implementation="flash_attention_2",
            **load_kwargs,
        )
        logger.info("Using Flash Attention 2")
    except (ImportError, ValueError):
        logger.info("Flash Attention 2 not available, using default attention")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(MODEL_DIR),
            **load_kwargs,
        )
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    load_time = round(time.time() - load_start, 2)

    post_load_mem = get_gpu_memory_mb()
    logger.info(f"Model loaded in {load_time}s, VRAM: {post_load_mem['allocated_mb']} MB")

    # Step 3: Run inference on all 5 test images
    results = []
    latencies = []

    for doc in MEDICAL_DOCUMENTS:
        image_path = TEST_IMAGES_DIR / doc["filename"]
        image = Image.open(image_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": doc["prompt"]},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        # Warm-up synchronization
        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=512)

        torch.cuda.synchronize()
        latency = round(time.time() - start, 4)
        latencies.append(latency)

        # Decode only new tokens
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        response = processor.decode(generated_ids, skip_special_tokens=True)

        results.append({
            "image": doc["filename"],
            "prompt": doc["prompt"],
            "response": response,
            "latency_seconds": latency,
        })
        logger.info(f"  {doc['filename']}: {latency}s, response length: {len(response)} chars")

    peak_mem = get_gpu_memory_mb()

    # Step 4: Compile benchmark
    benchmark = {
        "model_name": "Qwen2.5-VL-7B-Instruct",
        "precision": "FP16",
        "model_dir": str(MODEL_DIR),
        "load_time_seconds": load_time,
        "vram": {
            "pre_load_mb": pre_load_mem,
            "post_load_mb": post_load_mem,
            "peak_mb": peak_mem,
        },
        "inference_results": results,
        "latency_stats": {
            "avg_seconds": round(sum(latencies) / len(latencies), 4),
            "min_seconds": round(min(latencies), 4),
            "max_seconds": round(max(latencies), 4),
            "p50_seconds": round(sorted(latencies)[len(latencies) // 2], 4),
        },
    }

    output_path = BENCHMARKS_DIR / "baseline_fp16.json"
    with open(output_path, "w") as f:
        json.dump(benchmark, f, indent=2)
    logger.info(f"Baseline benchmark saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("BASELINE FP16 BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Model Load Time:          {load_time}s")
    print(f"VRAM (post-load):         {post_load_mem['allocated_mb']} MB")
    print(f"VRAM (peak):              {peak_mem['max_allocated_mb']} MB")
    print(f"Avg Inference Latency:    {benchmark['latency_stats']['avg_seconds']}s")
    print(f"Min Inference Latency:    {benchmark['latency_stats']['min_seconds']}s")
    print(f"Max Inference Latency:    {benchmark['latency_stats']['max_seconds']}s")
    print("-" * 70)
    for r in results:
        resp_preview = r["response"][:80].replace("\n", " ")
        print(f"  {r['image']:30s} {r['latency_seconds']:6.2f}s  {resp_preview}...")
    print("=" * 70)

    # Clean up
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Model unloaded, GPU cache cleared.")


if __name__ == "__main__":
    main()

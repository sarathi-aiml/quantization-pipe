"""Phase 6: Evaluate fine-tuned LoRA model against base model on medical document understanding.

This script creates a medical evaluation test suite with synthetic clinical documents,
evaluates both the base NF4-quantized model and the LoRA-fine-tuned model on medical
term and clinical value extraction accuracy, then benchmarks the fine-tuned model on
the original 5 test images for comparable latency/VRAM numbers against Phase 2/3.

Target hardware: NVIDIA DGX Spark (GB10 Blackwell, ~120 GB unified VRAM).
"""

import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Blackwell GPU compatibility patches -- must be applied before model loading
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from blackwell_compat import apply_blackwell_patches
apply_blackwell_patches()

from PIL import Image, ImageDraw, ImageFont
from transformers import BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-base"
LORA_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-medical-lora"
TEST_IMAGES_DIR = PROJECT_ROOT / "datasets" / "test_images"
EVAL_IMAGES_DIR = PROJECT_ROOT / "datasets" / "eval_images"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"

# ---------------------------------------------------------------------------
# Step 6.1 -- Medical Evaluation Test Suite
# ---------------------------------------------------------------------------
# These are DIFFERENT from the 5 standard test images. Each one exercises a
# distinct class of challenging medical document content: dense abbreviations,
# multi-drug regimens, complex lab panels, procedure notes, and referral letters.
# ---------------------------------------------------------------------------

EVAL_TEST_CASES: list[dict[str, Any]] = [
    {
        "id": "eval_medication_reconciliation",
        "filename": "eval_med_reconciliation.png",
        "title": "MEDICATION RECONCILIATION FORM",
        "content": [
            "Patient: Helen R. Kowalski   DOB: 02/11/1947   MRN: 8291053",
            "Reconciled by: PharmD J. Alvarez   Date: 01/25/2026",
            "",
            "ACTIVE MEDICATIONS:",
            " 1. Metoprolol Tartrate 50 mg PO BID (for HTN/AF)",
            " 2. Warfarin 5 mg PO QHS (for AF, INR goal 2.0-3.0)",
            " 3. Furosemide 80 mg PO QAM + 40 mg PO QPM (for CHF)",
            " 4. Spironolactone 25 mg PO QD (for CHF, K+ 4.1 mEq/L)",
            " 5. Metformin 1000 mg PO BID (for DM2, HbA1c 7.8%)",
            " 6. Glipizide 10 mg PO BID (for DM2)",
            " 7. Atorvastatin 80 mg PO QHS (for hyperlipidemia)",
            " 8. Gabapentin 300 mg PO TID (for peripheral neuropathy)",
            " 9. Omeprazole 20 mg PO QAM (for GERD)",
            "10. Albuterol MDI 2 puffs Q4-6H PRN (for COPD)",
            "11. Tiotropium 18 mcg INH QD (for COPD)",
            "",
            "ALLERGIES: PCN (anaphylaxis), Sulfa (rash), Codeine (N/V)",
            "",
            "COMORBIDITIES: HTN, AF, CHF (EF 35%), DM2, COPD,",
            "  CKD Stage 3b (eGFR 38), peripheral neuropathy, GERD",
            "",
            "LAST LABS (01/22/2026):",
            "  INR 2.4  |  K+ 4.1 mEq/L  |  Cr 1.6 mg/dL",
            "  BUN 32 mg/dL  |  HbA1c 7.8%  |  BNP 580 pg/mL",
        ],
        "prompt": (
            "Extract every medication with its exact dosage, frequency, and indication. "
            "Also list all comorbidities and relevant lab values from this medication "
            "reconciliation form."
        ),
        "expected_terms": [
            "Metoprolol Tartrate", "Warfarin", "Furosemide", "Spironolactone",
            "Metformin", "Glipizide", "Atorvastatin", "Gabapentin",
            "Omeprazole", "Albuterol", "Tiotropium",
            "HTN", "AF", "CHF", "DM2", "COPD", "CKD", "GERD",
            "peripheral neuropathy", "anaphylaxis",
        ],
        "expected_values": [
            "50 mg", "5 mg", "80 mg", "40 mg", "25 mg", "1000 mg",
            "10 mg", "300 mg", "20 mg", "18 mcg",
            "EF 35%", "eGFR 38", "INR 2.4", "K+ 4.1", "Cr 1.6",
            "BUN 32", "HbA1c 7.8%", "BNP 580",
            "BID", "QHS", "QAM", "QPM", "QD", "TID", "PRN",
        ],
    },
    {
        "id": "eval_icu_flowsheet",
        "filename": "eval_icu_flowsheet.png",
        "title": "ICU FLOWSHEET -- PROGRESS NOTE",
        "content": [
            "Patient: Marcus T. Williams   MRN: 7314829   Bed: MICU-4B",
            "Date: 01/24/2026 06:00   Attending: Dr. R. Kapoor, MD",
            "",
            "VENTILATOR SETTINGS:",
            "  Mode: SIMV/PS   FiO2: 0.45   PEEP: 8 cmH2O",
            "  Vt: 450 mL   RR set: 14   PS: 10 cmH2O",
            "  ABG (05:30): pH 7.32, pCO2 48, pO2 72, HCO3 24, BE -2",
            "  P/F Ratio: 160 (moderate ARDS)",
            "",
            "HEMODYNAMICS (06:00):",
            "  HR: 112 bpm (sinus tach)   MAP: 62 mmHg",
            "  CVP: 14 mmHg   ScvO2: 68%",
            "  Norepinephrine 0.15 mcg/kg/min (titrating for MAP >65)",
            "  Vasopressin 0.04 units/min (fixed dose)",
            "",
            "SEDATION / ANALGESIA:",
            "  Propofol 30 mcg/kg/min   RASS: -3",
            "  Fentanyl 75 mcg/hr   CPOT: 1",
            "",
            "I&O (last 12h): IN 2850 mL | OUT 1200 mL | NET +1650 mL",
            "  UOP: 25 mL/hr (oliguric, considering CRRT)",
            "",
            "DRIPS: Norepinephrine, Vasopressin, Propofol, Fentanyl,",
            "  Insulin gtt 4 units/hr (BG 198 mg/dL), Heparin gtt",
            "  800 units/hr (PTT target 60-80s, last PTT 54s)",
            "",
            "DX: Septic shock 2/2 pneumonia, moderate ARDS,",
            "  AKI Stage 2 (Cr 2.8, baseline 0.9), DIC (plt 62K,",
            "  fibrinogen 128, D-dimer >20), coagulopathy",
        ],
        "prompt": (
            "Extract all ventilator settings, hemodynamic parameters, drip rates, "
            "lab values, intake/output totals, and diagnoses from this ICU flowsheet."
        ),
        "expected_terms": [
            "SIMV", "PEEP", "ARDS", "ABG", "Norepinephrine", "Vasopressin",
            "Propofol", "Fentanyl", "Insulin", "Heparin",
            "RASS", "CPOT", "CRRT", "septic shock", "AKI", "DIC",
            "coagulopathy", "oliguric", "sinus tach",
        ],
        "expected_values": [
            "FiO2: 0.45", "PEEP: 8", "450 mL", "pH 7.32", "pCO2 48",
            "pO2 72", "HCO3 24", "P/F Ratio: 160",
            "HR: 112", "MAP: 62", "CVP: 14", "ScvO2: 68%",
            "0.15 mcg/kg/min", "0.04 units/min",
            "30 mcg/kg/min", "75 mcg/hr", "RASS: -3", "CPOT: 1",
            "2850 mL", "1200 mL", "+1650 mL", "25 mL/hr",
            "4 units/hr", "BG 198", "800 units/hr", "PTT 54",
            "Cr 2.8", "plt 62K", "fibrinogen 128", "D-dimer >20",
        ],
    },
    {
        "id": "eval_cardiology_consult",
        "filename": "eval_cardiology_consult.png",
        "title": "CARDIOLOGY CONSULTATION NOTE",
        "content": [
            "Patient: Dorothy A. Chen   DOB: 08/19/1955   MRN: 4028617",
            "Consult requested by: Dr. P. Okafor (Hospitalist)",
            "Cardiologist: Dr. N. Ramirez, FACC   Date: 01/23/2026",
            "",
            "REASON FOR CONSULT: New-onset CHF, reduced EF",
            "",
            "ECHO (01/22/2026):",
            "  LVEF: 25%   LVEDD: 6.2 cm   LVESD: 5.1 cm",
            "  Grade III diastolic dysfunction (restrictive pattern)",
            "  Moderate MR (ERO 0.25 cm2)   Mild TR   RVSP 52 mmHg",
            "  GLS: -8.2% (severely reduced, normal < -18%)",
            "",
            "ECG: NSR at 78 bpm, LBBB (QRS 156 ms), LAE, LVH by",
            "  voltage criteria, Q waves in V1-V3 (old anterior MI)",
            "",
            "CATH LAB (01/23/2026):",
            "  LM: 30% stenosis   LAD: 90% proximal (culprit),",
            "  stented with 3.0x28mm DES   LCx: 50% mid   RCA: 70%",
            "  LVEDP: 28 mmHg   PCWP: 24 mmHg   CI: 1.8 L/min/m2",
            "",
            "PLAN:",
            "  1. GDMT for HFrEF: Sacubitril/Valsartan 24/26 mg BID,",
            "     Carvedilol 3.125 mg BID, Eplerenone 25 mg QD,",
            "     Dapagliflozin 10 mg QD",
            "  2. DAPT: ASA 81 mg QD + Ticagrelor 90 mg BID x 12 mo",
            "  3. High-intensity statin: Rosuvastatin 20 mg QHS",
            "  4. CRT-D evaluation in 3 months if LVEF <35%",
            "  5. Cardiac rehab referral",
        ],
        "prompt": (
            "Extract all echocardiographic measurements, catheterization findings, "
            "ECG findings, medications with dosages, and the management plan from "
            "this cardiology consultation."
        ),
        "expected_terms": [
            "CHF", "HFrEF", "LVEF", "LVEDD", "LVESD", "MR", "TR", "RVSP",
            "GLS", "LBBB", "LAE", "LVH", "NSR", "MI", "DES",
            "LVEDP", "PCWP", "GDMT", "DAPT", "CRT-D",
            "Sacubitril/Valsartan", "Carvedilol", "Eplerenone",
            "Dapagliflozin", "Ticagrelor", "Rosuvastatin",
            "diastolic dysfunction", "restrictive pattern",
        ],
        "expected_values": [
            "25%", "6.2 cm", "5.1 cm", "0.25 cm2", "52 mmHg",
            "-8.2%", "78 bpm", "156 ms",
            "30%", "90%", "50%", "70%",
            "3.0x28mm", "28 mmHg", "24 mmHg", "1.8 L/min/m2",
            "24/26 mg", "3.125 mg", "25 mg", "10 mg",
            "81 mg", "90 mg", "20 mg",
            "BID", "QD", "QHS",
        ],
    },
    {
        "id": "eval_complex_lab_panel",
        "filename": "eval_complex_labs.png",
        "title": "LABORATORY RESULTS -- HEMATOLOGY / COAGULATION / HEPATIC",
        "content": [
            "Patient: James P. Nakamura   DOB: 12/05/1968   MRN: 5539201",
            "Collection: 01/24/2026 06:00   Status: FINAL",
            "Ordering: Dr. S. Patel, MD (Oncology)",
            "",
            "COMPLETE BLOOD COUNT:",
            "  TEST              RESULT   UNITS      REF         FLAG",
            "  WBC                2.1     K/uL       4.5-11.0      L",
            "  RBC                3.12    M/uL       4.5-5.9       L",
            "  Hemoglobin         8.4     g/dL       13.5-17.5     L",
            "  Hematocrit         25.8    %          38.3-48.6     L",
            "  MCV                82.7    fL         80-100",
            "  Platelets          48      K/uL       150-400       L",
            "  ANC                0.8     K/uL       1.5-8.0     L/C",
            "",
            "COAGULATION:",
            "  PT                 18.2    sec        11.0-13.5     H",
            "  INR                1.6               0.9-1.1        H",
            "  PTT                42      sec        25-35          H",
            "  Fibrinogen         158     mg/dL      200-400        L",
            "  D-Dimer            8.4     ug/mL FEU  <0.50         H",
            "",
            "HEPATIC FUNCTION:",
            "  AST (SGOT)         142     U/L        10-40          H",
            "  ALT (SGPT)         198     U/L        7-56           H",
            "  ALP                312     U/L        44-147         H",
            "  GGT                245     U/L        8-61           H",
            "  Total Bilirubin    4.8     mg/dL      0.1-1.2        H",
            "  Direct Bilirubin   3.6     mg/dL      0.0-0.3        H",
            "  Albumin            2.4     g/dL       3.5-5.5        L",
            "  LDH                890     U/L        140-280        H",
            "",
            "CRITICAL: ANC 0.8 (neutropenic), Platelets 48K",
            "  -- Oncology notified at 07:15",
        ],
        "prompt": (
            "Extract every lab test with its result, units, reference range, and "
            "abnormal flag. Identify all critical values from this laboratory report."
        ),
        "expected_terms": [
            "WBC", "RBC", "Hemoglobin", "Hematocrit", "MCV", "Platelets", "ANC",
            "PT", "INR", "PTT", "Fibrinogen", "D-Dimer",
            "AST", "ALT", "ALP", "GGT", "Bilirubin", "Albumin", "LDH",
            "neutropenic", "SGOT", "SGPT",
        ],
        "expected_values": [
            "2.1", "3.12", "8.4", "25.8", "82.7", "48", "0.8",
            "18.2", "1.6", "42", "158", "8.4",
            "142", "198", "312", "245", "4.8", "3.6", "2.4", "890",
            "K/uL", "g/dL", "mg/dL", "U/L", "sec",
        ],
    },
    {
        "id": "eval_surgical_op_note",
        "filename": "eval_operative_note.png",
        "title": "OPERATIVE NOTE -- LAPAROSCOPIC CHOLECYSTECTOMY",
        "content": [
            "Patient: Patricia M. O'Brien   DOB: 04/30/1973   MRN: 6195734",
            "Date: 01/23/2026   Surgeon: Dr. A. Gutierrez, FACS",
            "Anesthesia: General (ETT), Dr. L. Kim, MD",
            "",
            "PREOP DX: Acute cholecystitis with cholelithiasis",
            "POSTOP DX: Same, plus gangrenous cholecystitis",
            "",
            "PROCEDURE: Laparoscopic cholecystectomy, converted to",
            "  open due to severe inflammation and unclear anatomy",
            "",
            "FINDINGS:",
            "  - Gallbladder: distended, gangrenous wall, 2.1 cm",
            "    impacted stone in Hartmann's pouch",
            "  - Calot's triangle: dense adhesions, critical view",
            "    of safety NOT achieved laparoscopically",
            "  - CBD: 6 mm (normal), no stones on IOC",
            "  - Liver: mildly congested, no focal lesions",
            "",
            "EBL: 250 mL   Duration: 142 min   Fluids: LR 2400 mL",
            "",
            "SPECIMENS: Gallbladder to pathology (frozen + permanent)",
            "",
            "POSTOP ORDERS:",
            "  - NPO until flatus, then clear liquid diet",
            "  - Morphine PCA: 1 mg Q6min, lockout 10mg/hr",
            "  - Ketorolac 30 mg IV Q6H x 48hr (then PO ibuprofen)",
            "  - Piperacillin/Tazobactam 4.5 g IV Q8H",
            "  - Enoxaparin 40 mg SQ QD (DVT prophylaxis)",
            "  - Ondansetron 4 mg IV Q6H PRN nausea",
            "  - Incentive spirometry Q1H while awake",
            "  - Ambulate POD1, advance diet as tolerated",
        ],
        "prompt": (
            "Extract the diagnoses, procedure details, operative findings with "
            "measurements, estimated blood loss, medications with dosages, and "
            "postoperative orders from this operative note."
        ),
        "expected_terms": [
            "cholecystitis", "cholelithiasis", "gangrenous",
            "laparoscopic cholecystectomy", "Hartmann's pouch",
            "Calot's triangle", "critical view of safety",
            "CBD", "IOC", "Morphine", "PCA", "Ketorolac",
            "Piperacillin/Tazobactam", "Enoxaparin", "Ondansetron",
            "DVT prophylaxis", "incentive spirometry", "NPO",
            "ETT", "EBL", "FACS",
        ],
        "expected_values": [
            "2.1 cm", "6 mm", "250 mL", "142 min", "2400 mL",
            "1 mg", "10mg/hr", "30 mg", "4.5 g", "40 mg", "4 mg",
            "Q6min", "Q6H", "Q8H", "QD", "PRN", "Q1H",
            "POD1", "SQ",
        ],
    },
]


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def get_gpu_memory_mb() -> dict[str, float]:
    """Return current GPU memory statistics in megabytes."""
    return {
        "allocated_mb": round(torch.cuda.memory_allocated(0) / (1024 ** 2), 2),
        "reserved_mb": round(torch.cuda.memory_reserved(0) / (1024 ** 2), 2),
        "max_allocated_mb": round(torch.cuda.max_memory_allocated(0) / (1024 ** 2), 2),
    }


def create_eval_document_image(doc: dict[str, Any], output_path: Path) -> None:
    """Render a medical document as a high-fidelity synthetic image on white background."""
    width, height = 920, 1200
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Font selection with graceful fallback
    try:
        title_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 16
        )
        body_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13
        )
    except OSError:
        try:
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf", 16
            )
            body_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 13
            )
        except OSError:
            title_font = ImageFont.load_default()
            body_font = ImageFont.load_default()

    # Border
    draw.rectangle([(20, 20), (width - 20, height - 20)], outline=(0, 0, 0), width=2)

    # Title
    y = 40
    draw.text((40, y), doc["title"], fill=(0, 0, 128), font=title_font)
    y += 30
    draw.line([(40, y), (width - 40, y)], fill=(0, 0, 0), width=1)
    y += 15

    # Body lines
    for line in doc["content"]:
        draw.text((40, y), line, fill=(0, 0, 0), font=body_font)
        y += 18
        if y > height - 60:
            break

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, "PNG")
    logger.info(f"Created eval image: {output_path.name}")


def compute_extraction_accuracy(
    response: str,
    expected_terms: list[str],
    expected_values: list[str],
) -> dict[str, Any]:
    """Score how many expected medical terms and clinical values appear in the response.

    Matching is case-insensitive. Returns term accuracy, value accuracy, and the
    combined (averaged) accuracy.
    """
    response_lower = response.lower()

    term_hits = [t for t in expected_terms if t.lower() in response_lower]
    value_hits = [v for v in expected_values if v.lower() in response_lower]

    term_accuracy = len(term_hits) / len(expected_terms) if expected_terms else 0.0
    value_accuracy = len(value_hits) / len(expected_values) if expected_values else 0.0
    combined_accuracy = (term_accuracy + value_accuracy) / 2.0

    return {
        "term_accuracy": round(term_accuracy * 100, 2),
        "value_accuracy": round(value_accuracy * 100, 2),
        "combined_accuracy": round(combined_accuracy * 100, 2),
        "terms_found": len(term_hits),
        "terms_total": len(expected_terms),
        "terms_missed": sorted(set(expected_terms) - set(term_hits)),
        "values_found": len(value_hits),
        "values_total": len(expected_values),
        "values_missed": sorted(set(expected_values) - set(value_hits)),
    }


def build_bnb_config() -> BitsAndBytesConfig:
    """Return the standard BNB NF4 quantization config used throughout the pipeline."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def get_max_memory() -> dict:
    """Compute max_memory mapping for model loading (80% GPU, 32 GB CPU)."""
    total_vram = torch.cuda.get_device_properties(0).total_memory
    return {0: int(total_vram * 0.80), "cpu": "32GB"}


def load_base_model_nf4() -> tuple:
    """Load Qwen2.5-VL-7B-Instruct with BNB NF4 quantization (no LoRA).

    Returns:
        Tuple of (model, processor, load_time_seconds).
    """
    logger.info("Loading BASE model (NF4 quantized, no LoRA)...")
    torch.cuda.reset_peak_memory_stats()

    load_start = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        quantization_config=build_bnb_config(),
        device_map="auto",
        max_memory=get_max_memory(),
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    load_time = round(time.time() - load_start, 2)

    logger.info(
        f"Base model loaded in {load_time}s | "
        f"VRAM: {get_gpu_memory_mb()['allocated_mb']} MB"
    )
    return model, processor, load_time


def load_finetuned_model_nf4() -> tuple:
    """Load Qwen2.5-VL-7B-Instruct with BNB NF4 + LoRA adapter, then merge.

    The LoRA weights are merged into the base model via merge_and_unload() so
    that inference runs at native NF4 speed without adapter overhead.

    Returns:
        Tuple of (model, processor, load_time_seconds).
    """
    logger.info("Loading FINE-TUNED model (NF4 base + LoRA adapter, merged)...")
    torch.cuda.reset_peak_memory_stats()

    load_start = time.time()
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        quantization_config=build_bnb_config(),
        device_map="auto",
        max_memory=get_max_memory(),
    )
    logger.info("Base model loaded, attaching LoRA adapter...")

    model = PeftModel.from_pretrained(base_model, str(LORA_DIR))
    model = model.merge_and_unload()
    logger.info("LoRA adapter merged and unloaded.")

    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    load_time = round(time.time() - load_start, 2)

    logger.info(
        f"Fine-tuned model loaded in {load_time}s | "
        f"VRAM: {get_gpu_memory_mb()['allocated_mb']} MB"
    )
    return model, processor, load_time


def run_inference(
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int = 512,
) -> tuple[str, float]:
    """Run a single VLM inference pass and return (response_text, latency_seconds).

    Uses torch.cuda.synchronize() around generation for accurate latency.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    torch.cuda.synchronize()
    latency = round(time.time() - start, 4)

    generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    return response, latency


def unload_model(model: Any, processor: Any | None = None) -> None:
    """Delete a model (and optionally processor) and reclaim GPU memory."""
    del model
    if processor is not None:
        del processor
    gc.collect()
    torch.cuda.empty_cache()
    logger.info(
        f"Model unloaded | VRAM after cleanup: "
        f"{get_gpu_memory_mb()['allocated_mb']} MB"
    )


# ---------------------------------------------------------------------------
# Step 6.2 -- Evaluate both models on the evaluation suite
# ---------------------------------------------------------------------------

def evaluate_model_on_suite(
    model: Any,
    processor: Any,
    model_label: str,
) -> list[dict[str, Any]]:
    """Run all evaluation test cases against *model* and return per-case results.

    Each result includes the response, latency, accuracy scores, and GPU memory.
    """
    results: list[dict[str, Any]] = []

    for tc in EVAL_TEST_CASES:
        image_path = EVAL_IMAGES_DIR / tc["filename"]
        image = Image.open(image_path)

        response, latency = run_inference(model, processor, image, tc["prompt"])
        accuracy = compute_extraction_accuracy(
            response, tc["expected_terms"], tc["expected_values"]
        )
        mem = get_gpu_memory_mb()

        result = {
            "test_id": tc["id"],
            "image": tc["filename"],
            "prompt": tc["prompt"],
            "response": response,
            "latency_seconds": latency,
            "accuracy": accuracy,
            "gpu_memory_mb": mem,
        }
        results.append(result)

        logger.info(
            f"  [{model_label}] {tc['id']}: "
            f"latency={latency}s | "
            f"term_acc={accuracy['term_accuracy']}% | "
            f"value_acc={accuracy['value_accuracy']}% | "
            f"combined={accuracy['combined_accuracy']}%"
        )

    return results


# ---------------------------------------------------------------------------
# Step 6.3 -- Benchmark fine-tuned model on original 5 test images
# ---------------------------------------------------------------------------

def benchmark_on_standard_images(
    model: Any,
    processor: Any,
    baseline_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Run the fine-tuned model on the original 5 standard test images.

    Uses the exact same prompts from baseline_fp16.json so latency / VRAM numbers
    are directly comparable to Phase 2 and Phase 3.
    """
    results: list[dict[str, Any]] = []

    for baseline_entry in baseline_results:
        image_path = TEST_IMAGES_DIR / baseline_entry["image"]
        image = Image.open(image_path)

        response, latency = run_inference(
            model, processor, image, baseline_entry["prompt"]
        )

        results.append({
            "image": baseline_entry["image"],
            "prompt": baseline_entry["prompt"],
            "response": response,
            "latency_seconds": latency,
        })
        logger.info(
            f"  [Standard] {baseline_entry['image']}: "
            f"{latency}s, {len(response)} chars"
        )

    return results


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def summarize_eval_results(results: list[dict[str, Any]]) -> dict[str, float]:
    """Compute aggregate statistics across evaluation results."""
    latencies = [r["latency_seconds"] for r in results]
    term_accs = [r["accuracy"]["term_accuracy"] for r in results]
    value_accs = [r["accuracy"]["value_accuracy"] for r in results]
    combined_accs = [r["accuracy"]["combined_accuracy"] for r in results]

    return {
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 4),
        "min_latency_seconds": round(min(latencies), 4),
        "max_latency_seconds": round(max(latencies), 4),
        "avg_term_accuracy_pct": round(sum(term_accs) / len(term_accs), 2),
        "avg_value_accuracy_pct": round(sum(value_accs) / len(value_accs), 2),
        "avg_combined_accuracy_pct": round(sum(combined_accs) / len(combined_accs), 2),
    }


def print_comparison_table(
    fp16_bench: dict[str, Any],
    int4_bench: dict[str, Any],
    base_eval_summary: dict[str, float],
    ft_eval_summary: dict[str, float],
    ft_standard_bench: dict[str, Any],
) -> None:
    """Print a comprehensive three-way comparison table to stdout."""
    sep = "=" * 90
    dash = "-" * 90

    print(f"\n{sep}")
    print("COMPREHENSIVE MODEL COMPARISON: Base FP16 -> Quantized INT4 -> Fine-Tuned LoRA")
    print(sep)
    print(
        f"{'Metric':<40} {'FP16':>14} {'NF4 (base)':>14} {'NF4+LoRA':>14}"
    )
    print(dash)

    # VRAM
    fp16_vram = fp16_bench["vram"]["post_load_mb"]["allocated_mb"]
    int4_vram = int4_bench["vram"]["post_load_mb"]["allocated_mb"]
    ft_vram = ft_standard_bench["vram"]["post_load_mb"]["allocated_mb"]
    print(f"{'VRAM Post-Load (MB)':<40} {fp16_vram:>14.1f} {int4_vram:>14.1f} {ft_vram:>14.1f}")

    fp16_peak = fp16_bench["vram"]["peak_mb"]["max_allocated_mb"]
    int4_peak = int4_bench["vram"]["peak_mb"]["max_allocated_mb"]
    ft_peak = ft_standard_bench["vram"]["peak_mb"]["max_allocated_mb"]
    print(f"{'VRAM Peak (MB)':<40} {fp16_peak:>14.1f} {int4_peak:>14.1f} {ft_peak:>14.1f}")

    # Load time
    fp16_load = fp16_bench["load_time_seconds"]
    int4_load = int4_bench["load_time_seconds"]
    ft_load = ft_standard_bench["load_time_seconds"]
    print(f"{'Model Load Time (s)':<40} {fp16_load:>14.2f} {int4_load:>14.2f} {ft_load:>14.2f}")

    print(dash)
    print("  STANDARD BENCHMARK (5 original test images)")
    print(dash)

    # Standard benchmark latency
    fp16_lat = fp16_bench["latency_stats"]["avg_seconds"]
    int4_lat = int4_bench["latency_stats"]["avg_seconds"]
    ft_lat = ft_standard_bench["latency_stats"]["avg_seconds"]
    print(f"{'Avg Inference Latency (s)':<40} {fp16_lat:>14.4f} {int4_lat:>14.4f} {ft_lat:>14.4f}")

    fp16_min = fp16_bench["latency_stats"]["min_seconds"]
    int4_min = int4_bench["latency_stats"]["min_seconds"]
    ft_min = ft_standard_bench["latency_stats"]["min_seconds"]
    print(f"{'Min Inference Latency (s)':<40} {fp16_min:>14.4f} {int4_min:>14.4f} {ft_min:>14.4f}")

    fp16_max = fp16_bench["latency_stats"]["max_seconds"]
    int4_max = int4_bench["latency_stats"]["max_seconds"]
    ft_max = ft_standard_bench["latency_stats"]["max_seconds"]
    print(f"{'Max Inference Latency (s)':<40} {fp16_max:>14.4f} {int4_max:>14.4f} {ft_max:>14.4f}")

    print(dash)
    print("  MEDICAL EVALUATION SUITE (5 new eval test cases)")
    print(dash)

    # Eval suite accuracy (base vs fine-tuned only; FP16 not run on eval suite)
    print(f"{'Avg Term Accuracy (%)':<40} {'N/A':>14} {base_eval_summary['avg_term_accuracy_pct']:>13.2f}% {ft_eval_summary['avg_term_accuracy_pct']:>13.2f}%")
    print(f"{'Avg Value Accuracy (%)':<40} {'N/A':>14} {base_eval_summary['avg_value_accuracy_pct']:>13.2f}% {ft_eval_summary['avg_value_accuracy_pct']:>13.2f}%")
    print(f"{'Avg Combined Accuracy (%)':<40} {'N/A':>14} {base_eval_summary['avg_combined_accuracy_pct']:>13.2f}% {ft_eval_summary['avg_combined_accuracy_pct']:>13.2f}%")
    print(f"{'Avg Eval Latency (s)':<40} {'N/A':>14} {base_eval_summary['avg_latency_seconds']:>14.4f} {ft_eval_summary['avg_latency_seconds']:>14.4f}")

    print(dash)
    print("  VRAM REDUCTION vs FP16")
    print(dash)
    int4_reduction = round((1 - int4_vram / fp16_vram) * 100, 1)
    ft_reduction = round((1 - ft_vram / fp16_vram) * 100, 1)
    print(f"{'VRAM Savings (%)':<40} {'--':>14} {int4_reduction:>13.1f}% {ft_reduction:>13.1f}%")

    # Accuracy delta
    base_combined = base_eval_summary["avg_combined_accuracy_pct"]
    ft_combined = ft_eval_summary["avg_combined_accuracy_pct"]
    delta = round(ft_combined - base_combined, 2)
    direction = "+" if delta >= 0 else ""
    print(f"{'LoRA Accuracy Delta (combined)':<40} {'--':>14} {'(baseline)':>14} {direction}{delta:>12.2f}%")

    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate the full Phase 6 evaluation pipeline."""
    BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 6.1: Create the medical evaluation test-suite images
    # ------------------------------------------------------------------
    logger.info("Step 6.1: Creating medical evaluation test suite images...")
    for tc in EVAL_TEST_CASES:
        create_eval_document_image(tc, EVAL_IMAGES_DIR / tc["filename"])
    logger.info(f"Created {len(EVAL_TEST_CASES)} eval images in {EVAL_IMAGES_DIR}")

    # ------------------------------------------------------------------
    # Step 6.2a: Evaluate BASE model (NF4, no LoRA) on eval suite
    # ------------------------------------------------------------------
    logger.info("Step 6.2a: Evaluating BASE model on medical eval suite...")
    base_model, base_processor, base_load_time = load_base_model_nf4()
    base_post_load_mem = get_gpu_memory_mb()

    base_eval_results = evaluate_model_on_suite(base_model, base_processor, "BASE")
    base_peak_mem = get_gpu_memory_mb()
    base_eval_summary = summarize_eval_results(base_eval_results)

    logger.info(
        f"Base eval done | avg_combined_accuracy={base_eval_summary['avg_combined_accuracy_pct']}%"
    )

    # Clean up base model before loading fine-tuned
    unload_model(base_model, base_processor)

    # ------------------------------------------------------------------
    # Step 6.2b: Evaluate FINE-TUNED model on eval suite
    # ------------------------------------------------------------------
    logger.info("Step 6.2b: Evaluating FINE-TUNED model on medical eval suite...")
    ft_model, ft_processor, ft_load_time = load_finetuned_model_nf4()
    ft_post_load_mem = get_gpu_memory_mb()

    ft_eval_results = evaluate_model_on_suite(ft_model, ft_processor, "FINE-TUNED")
    ft_eval_peak_mem = get_gpu_memory_mb()
    ft_eval_summary = summarize_eval_results(ft_eval_results)

    logger.info(
        f"Fine-tuned eval done | avg_combined_accuracy={ft_eval_summary['avg_combined_accuracy_pct']}%"
    )

    # ------------------------------------------------------------------
    # Step 6.3: Benchmark fine-tuned model on original 5 test images
    # ------------------------------------------------------------------
    logger.info("Step 6.3: Benchmarking fine-tuned model on standard 5 test images...")

    # Load baseline prompts from Phase 2
    baseline_fp16_path = BENCHMARKS_DIR / "baseline_fp16.json"
    with open(baseline_fp16_path) as f:
        fp16_bench = json.load(f)

    standard_results = benchmark_on_standard_images(
        ft_model, ft_processor, fp16_bench["inference_results"]
    )

    ft_standard_peak_mem = get_gpu_memory_mb()
    standard_latencies = [r["latency_seconds"] for r in standard_results]

    ft_standard_bench = {
        "model_name": "Qwen2.5-VL-7B-Instruct + LoRA (medical)",
        "precision": "NF4 (bitsandbytes) + LoRA merged",
        "adapter_path": str(LORA_DIR),
        "load_time_seconds": ft_load_time,
        "vram": {
            "pre_load_mb": {"allocated_mb": 0.0, "reserved_mb": 0.0, "max_allocated_mb": 0.0},
            "post_load_mb": ft_post_load_mem,
            "peak_mb": ft_standard_peak_mem,
        },
        "inference_results": standard_results,
        "latency_stats": {
            "avg_seconds": round(sum(standard_latencies) / len(standard_latencies), 4),
            "min_seconds": round(min(standard_latencies), 4),
            "max_seconds": round(max(standard_latencies), 4),
            "p50_seconds": round(
                sorted(standard_latencies)[len(standard_latencies) // 2], 4
            ),
        },
    }

    # Clean up fine-tuned model
    unload_model(ft_model, ft_processor)

    # ------------------------------------------------------------------
    # Step 6.4: Save results
    # ------------------------------------------------------------------
    logger.info("Step 6.4: Saving evaluation results...")

    # Detailed evaluation results
    evaluation_output = {
        "phase": "6 -- Fine-tuned model evaluation",
        "eval_test_cases": len(EVAL_TEST_CASES),
        "base_model": {
            "label": "Qwen2.5-VL-7B-Instruct (NF4, no LoRA)",
            "load_time_seconds": base_load_time,
            "vram_post_load_mb": base_post_load_mem,
            "vram_peak_mb": base_peak_mem,
            "eval_results": base_eval_results,
            "summary": base_eval_summary,
        },
        "finetuned_model": {
            "label": "Qwen2.5-VL-7B-Instruct (NF4 + LoRA merged)",
            "adapter_path": str(LORA_DIR),
            "load_time_seconds": ft_load_time,
            "vram_post_load_mb": ft_post_load_mem,
            "vram_peak_mb": ft_eval_peak_mem,
            "eval_results": ft_eval_results,
            "summary": ft_eval_summary,
        },
        "accuracy_delta": {
            "term_accuracy_delta_pct": round(
                ft_eval_summary["avg_term_accuracy_pct"]
                - base_eval_summary["avg_term_accuracy_pct"],
                2,
            ),
            "value_accuracy_delta_pct": round(
                ft_eval_summary["avg_value_accuracy_pct"]
                - base_eval_summary["avg_value_accuracy_pct"],
                2,
            ),
            "combined_accuracy_delta_pct": round(
                ft_eval_summary["avg_combined_accuracy_pct"]
                - base_eval_summary["avg_combined_accuracy_pct"],
                2,
            ),
        },
    }

    eval_path = BENCHMARKS_DIR / "evaluation_finetuned.json"
    with open(eval_path, "w") as f:
        json.dump(evaluation_output, f, indent=2, default=str)
    logger.info(f"Detailed evaluation saved to {eval_path}")

    # Standard benchmark comparable to Phase 2/3
    standard_bench_path = BENCHMARKS_DIR / "finetuned_standard_bench.json"
    with open(standard_bench_path, "w") as f:
        json.dump(ft_standard_bench, f, indent=2, default=str)
    logger.info(f"Standard benchmark saved to {standard_bench_path}")

    # ------------------------------------------------------------------
    # Print comprehensive comparison table
    # ------------------------------------------------------------------
    # Load INT4 benchmark from Phase 3
    int4_bench_path = BENCHMARKS_DIR / "quantized_int4.json"
    if int4_bench_path.exists():
        with open(int4_bench_path) as f:
            int4_bench = json.load(f)
    else:
        logger.warning(
            f"{int4_bench_path} not found -- using placeholder for comparison table"
        )
        int4_bench = {
            "load_time_seconds": 0,
            "vram": {
                "post_load_mb": {"allocated_mb": 0},
                "peak_mb": {"max_allocated_mb": 0},
            },
            "latency_stats": {
                "avg_seconds": 0,
                "min_seconds": 0,
                "max_seconds": 0,
            },
        }

    print_comparison_table(
        fp16_bench=fp16_bench,
        int4_bench=int4_bench,
        base_eval_summary=base_eval_summary,
        ft_eval_summary=ft_eval_summary,
        ft_standard_bench=ft_standard_bench,
    )

    # ------------------------------------------------------------------
    # Per-case detail table
    # ------------------------------------------------------------------
    print(f"\n{'='*90}")
    print("PER-CASE EVALUATION DETAIL")
    print(f"{'='*90}")
    print(
        f"{'Test Case':<35} {'Model':<12} {'Term%':>7} {'Value%':>7} "
        f"{'Combined%':>10} {'Latency':>8}"
    )
    print("-" * 90)

    for base_r, ft_r in zip(base_eval_results, ft_eval_results):
        ba = base_r["accuracy"]
        fa = ft_r["accuracy"]
        print(
            f"{base_r['test_id']:<35} {'BASE':<12} "
            f"{ba['term_accuracy']:>6.1f}% {ba['value_accuracy']:>6.1f}% "
            f"{ba['combined_accuracy']:>9.1f}% {base_r['latency_seconds']:>7.2f}s"
        )
        print(
            f"{'':<35} {'FINE-TUNED':<12} "
            f"{fa['term_accuracy']:>6.1f}% {fa['value_accuracy']:>6.1f}% "
            f"{fa['combined_accuracy']:>9.1f}% {ft_r['latency_seconds']:>7.2f}s"
        )
        print("-" * 90)

    print(f"{'='*90}")
    logger.info("Phase 6 evaluation complete.")


if __name__ == "__main__":
    main()

"""LoRA fine-tuning of Qwen2.5-VL-7B-Instruct for medical document understanding.

Optimized for speed: subsamples training data, disables intermediate eval,
uses efficient data loading on DGX Spark Blackwell GPU.
"""

import gc
import json
import logging
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from blackwell_compat import apply_blackwell_patches
apply_blackwell_patches()

from PIL import Image
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    TrainingArguments,
    Trainer,
)
from torch.utils.data import Dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-base"
OUTPUT_DIR = PROJECT_ROOT / "models" / "qwen25-vl-7b-medical-lora"
BENCHMARKS_DIR = PROJECT_ROOT / "benchmarks"
FORMATTED_DIR = PROJECT_ROOT / "datasets" / "formatted"

# Import once at module level instead of per-sample
from qwen_vl_utils import process_vision_info


class MedicalVisionDataset(Dataset):
    """Dataset for medical vision-language fine-tuning.

    Subsamples data to max_samples for practical training times on
    multimodal models with per-sample image processing.
    """

    def __init__(
        self,
        data_path: Path,
        processor,
        max_length: int = 512,
        max_samples: int = 0,
    ):
        with open(data_path) as f:
            all_samples = json.load(f)

        if max_samples > 0 and len(all_samples) > max_samples:
            random.seed(42)
            self.samples = random.sample(all_samples, max_samples)
            logger.info(
                f"Subsampled {max_samples} from {len(all_samples)} in {data_path.name}"
            )
        else:
            self.samples = all_samples
            logger.info(f"Loaded {len(self.samples)} samples from {data_path.name}")

        self.processor = processor
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        image_path = PROJECT_ROOT / sample["image_path"]

        try:
            image = Image.open(image_path).convert("RGB")
            # Resize aggressively to speed up vision encoder processing
            max_dim = 256
            if max(image.size) > max_dim:
                ratio = max_dim / max(image.size)
                new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                image = image.resize(new_size, Image.LANCZOS)
        except Exception:
            image = Image.new("RGB", (128, 128), (255, 255, 255))

        prompt = sample["prompt"]
        response = sample["response"][:600]  # Shorter responses for speed

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": response}],
            },
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # Create labels: mask everything except assistant response
        labels = input_ids.clone()
        assistant_marker = self.processor.tokenizer.encode(
            "assistant\n", add_special_tokens=False
        )
        marker_len = len(assistant_marker)
        text_ids = input_ids.tolist()

        mask_end = 0
        for i in range(len(text_ids) - marker_len + 1):
            if text_ids[i : i + marker_len] == assistant_marker:
                mask_end = i + marker_len
        if mask_end > 0:
            labels[:mask_end] = -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if pixel_values is not None:
            result["pixel_values"] = (
                pixel_values.squeeze(0) if pixel_values.dim() > 3 else pixel_values
            )
        if image_grid_thw is not None:
            result["image_grid_thw"] = (
                image_grid_thw.squeeze(0) if image_grid_thw.dim() > 1 else image_grid_thw
            )

        return result


class MedicalVLCollator:
    """Custom data collator for variable-length multimodal inputs."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: list[dict]) -> dict:
        input_ids = torch.stack([f["input_ids"] for f in features])
        attention_mask = torch.stack([f["attention_mask"] for f in features])
        labels = torch.stack([f["labels"] for f in features])

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if "pixel_values" in features[0] and features[0]["pixel_values"] is not None:
            pixel_values = torch.cat(
                [f["pixel_values"] for f in features if f.get("pixel_values") is not None],
                dim=0,
            )
            batch["pixel_values"] = pixel_values

        if "image_grid_thw" in features[0] and features[0]["image_grid_thw"] is not None:
            image_grid_thw = torch.stack(
                [f["image_grid_thw"] for f in features if f.get("image_grid_thw") is not None]
            )
            batch["image_grid_thw"] = image_grid_thw

        return batch


class LoggingCallback:
    """Simple callback to print training progress."""

    def __init__(self):
        self.step_start = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            elapsed = round(time.time() - self.step_start, 1)
            loss = logs.get("loss", "N/A")
            lr = logs.get("learning_rate", "N/A")
            step = state.global_step
            total = state.max_steps
            pct = round(step / total * 100, 1) if total > 0 else 0
            print(
                f"  Step {step}/{total} ({pct}%) | "
                f"loss={loss} | lr={lr} | elapsed={elapsed}s",
                flush=True,
            )


def main() -> None:
    """Run LoRA fine-tuning with speed-optimized configuration."""

    # Configuration — optimized for speed on DGX Spark
    LORA_RANK = 64
    LORA_ALPHA = 128
    LORA_DROPOUT = 0.05
    LEARNING_RATE = 2e-4  # Slightly higher LR for fewer steps
    NUM_EPOCHS = 2
    BATCH_SIZE = 4
    GRAD_ACCUM = 4  # Effective batch = 16
    MAX_SEQ_LENGTH = 512
    MAX_TRAIN_SAMPLES = 1000  # Subsample for practical training time
    WARMUP_RATIO = 0.05
    WEIGHT_DECAY = 0.01

    print(f"\n{'='*60}", flush=True)
    print("PHASE 5: LoRA FINE-TUNING", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Training samples: {MAX_TRAIN_SAMPLES} (subsampled)", flush=True)
    print(f"Epochs: {NUM_EPOCHS}", flush=True)
    print(f"Batch size: {BATCH_SIZE} x {GRAD_ACCUM} accum = {BATCH_SIZE * GRAD_ACCUM} effective", flush=True)
    print(f"Max seq length: {MAX_SEQ_LENGTH}", flush=True)
    print(f"LoRA rank: {LORA_RANK}, alpha: {LORA_ALPHA}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Load model with BNB NF4
    logger.info("Loading base model with BNB NF4 quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    total_vram = torch.cuda.get_device_properties(0).total_memory
    max_memory = {0: int(total_vram * 0.85), "cpu": "32GB"}

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(MODEL_DIR),
        quantization_config=bnb_config,
        device_map="auto",
        max_memory=max_memory,
    )
    processor = AutoProcessor.from_pretrained(str(MODEL_DIR))
    logger.info("Model loaded successfully.")

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # Freeze vision encoder
    frozen_count = 0
    for name, param in model.named_parameters():
        if "visual" in name:
            param.requires_grad = False
            frozen_count += 1
    logger.info(f"Frozen {frozen_count} vision encoder parameters")

    # Configure LoRA
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)

    # Print parameter stats
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_pct = round(trainable_params / total_params * 100, 2)
    print(f"\nTrainable: {trainable_params:,} / {total_params:,} ({trainable_pct}%)\n", flush=True)

    # Load datasets (subsampled)
    train_dataset = MedicalVisionDataset(
        FORMATTED_DIR / "train.json", processor, MAX_SEQ_LENGTH,
        max_samples=MAX_TRAIN_SAMPLES,
    )
    val_dataset = MedicalVisionDataset(
        FORMATTED_DIR / "val.json", processor, MAX_SEQ_LENGTH,
        max_samples=100,  # Small val set for speed
    )

    collator = MedicalVLCollator(processor)

    # Calculate expected steps
    steps_per_epoch = len(train_dataset) // BATCH_SIZE // GRAD_ACCUM
    total_steps = steps_per_epoch * NUM_EPOCHS
    print(f"Expected: {steps_per_epoch} optimizer steps/epoch, {total_steps} total\n", flush=True)

    # Training arguments — optimized for speed
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        bf16=True,
        logging_steps=5,
        save_strategy="no",  # Save only at end for speed
        eval_strategy="no",  # Skip intermediate eval for speed
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,  # More workers for faster data loading
        dataloader_prefetch_factor=4,  # Prefetch more batches
        remove_unused_columns=False,
        report_to="none",
        max_grad_norm=1.0,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    # Train
    logger.info("Starting LoRA fine-tuning...")
    train_start = time.time()

    try:
        train_result = trainer.train()
        training_time = round(time.time() - train_start, 2)
        logger.info(f"Training completed in {training_time}s")
        train_loss = train_result.training_loss
        metrics = train_result.metrics
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        training_time = round(time.time() - train_start, 2)
        train_loss = -1
        metrics = {"error": str(e)}

    # Save LoRA adapter
    logger.info("Saving LoRA adapter...")
    model.save_pretrained(str(OUTPUT_DIR))
    processor.save_pretrained(str(OUTPUT_DIR))
    logger.info(f"LoRA adapter saved to {OUTPUT_DIR}")

    # Verify adapter files exist
    adapter_config = OUTPUT_DIR / "adapter_config.json"
    if adapter_config.exists():
        logger.info("adapter_config.json verified")
    else:
        logger.error("adapter_config.json NOT FOUND — save may have failed")

    # Save training metrics
    training_metrics = {
        "training_time_seconds": training_time,
        "final_loss": train_loss,
        "epochs_completed": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "trainable_params": trainable_params,
        "total_params": total_params,
        "trainable_pct": trainable_pct,
        "batch_size": BATCH_SIZE,
        "grad_accum_steps": GRAD_ACCUM,
        "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
        "max_seq_length": MAX_SEQ_LENGTH,
        "training_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "total_training_samples_available": 3916,
        "subsample_note": f"Used {len(train_dataset)} of 3916 for practical training time on multimodal VL model",
        "metrics": metrics,
    }

    metrics_path = BENCHMARKS_DIR / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(training_metrics, f, indent=2, default=str)
    logger.info(f"Training metrics saved to {metrics_path}")

    print(f"\n{'='*60}", flush=True)
    print("TRAINING SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Training time: {training_time}s ({training_time/60:.1f} min)", flush=True)
    print(f"Final loss: {train_loss}", flush=True)
    print(f"Trainable params: {trainable_params:,} ({trainable_pct}%)", flush=True)
    print(f"Training samples: {len(train_dataset)}", flush=True)
    print(f"Adapter saved: {OUTPUT_DIR}", flush=True)
    print(f"{'='*60}\n", flush=True)

    # Clean up
    del model
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

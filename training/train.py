"""
Main training orchestration script.

Brings together data, model, and training loop.
Handles checkpoint resume logic for Colab disconnects.

Usage (from Colab notebook):
    from training.train import run_training
    trainer = run_training()
"""

import os
from pathlib import Path

# IMPORTANT: import unsloth BEFORE trl/transformers so its monkey-patches
# (fused LoRA Triton kernels, fast cross-entropy, etc.) apply to the
# SFTTrainer training loop. Importing in the wrong order silently disables
# most of the speedup. See docs/v2_evaluation.md for context.
import unsloth  # noqa: F401  (import-for-side-effects)

from trl import SFTConfig, SFTTrainer

from training.config import (
    ADAPTER_DIR,
    BATCH_SIZE,
    CHECKPOINT_DIR,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOGGING_DIR,
    LR_SCHEDULER,
    MAX_SEQ_LENGTH,
    NUM_EPOCHS,
    SAVE_STEPS,
    SAVE_TOTAL_LIMIT,
    USE_BF16,
    USE_FP16,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from training.data_prep import (
    filter_by_token_length,
    format_prompts,
    load_and_filter_dataset,
)
from training.model import apply_lora, load_base_model


def find_last_checkpoint() -> str | None:
    """
    Find the latest checkpoint in the checkpoint directory on Google Drive.

    Returns:
        Path to the latest checkpoint directory, or None if no checkpoint exists.
    """
    if not os.path.isdir(CHECKPOINT_DIR):
        return None

    checkpoints = [
        d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")
    ]

    if not checkpoints:
        return None

    # Sort by step number (checkpoint-100, checkpoint-200, etc.)
    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[-1]))[-1]
    checkpoint_path = os.path.join(CHECKPOINT_DIR, latest)
    return checkpoint_path


def get_training_args() -> SFTConfig:
    """Create the SFTConfig with all hyperparameters from config."""

    return SFTConfig(
        # Output & Checkpointing (Google Drive)
        output_dir=CHECKPOINT_DIR,
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        # Training
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER,
        # Precision
        fp16=USE_FP16,
        bf16=USE_BF16,
        # Logging (Google Drive)
        logging_steps=10,
        logging_dir=LOGGING_DIR,
        # Evaluation
        eval_strategy="epoch",
        load_best_model_at_end=False,
        # Performance
        # gradient_checkpointing is handled by Unsloth via
        # use_gradient_checkpointing="unsloth" in apply_lora(). Enabling it
        # here too would double-wrap and break Unsloth's optimized impl.
        gradient_checkpointing=False,
        max_grad_norm=0.3,
        # adamw_8bit is Unsloth's recommended optimizer on T4 — lower VRAM
        # than paged_adamw_32bit with no accuracy loss for LoRA training.
        optim="adamw_8bit",
        # SFT-specific
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=True,
    )


def run_training():
    """
    Execute the full training pipeline:

    1. Load & filter dataset (from cache or download)
    2. Format prompts
    3. Load model with QLoRA
    4. Apply LoRA adapters
    5. Filter by token length
    6. Train (with auto-resume from checkpoint)
    7. Save LoRA adapter to Google Drive

    Returns:
        The SFTTrainer instance (for post-training analysis).
    """

    print("=" * 60)
    print("🚀 PR Review AI — LoRA Fine-Tuning")
    print("=" * 60)

    # ---- Step 1: Data ----
    dataset = load_and_filter_dataset()
    dataset = format_prompts(dataset)

    # ---- Step 2: Model ----
    model, tokenizer = load_base_model()

    # ---- Step 3: Token length filter (needs tokenizer) ----
    dataset = filter_by_token_length(dataset, tokenizer)

    # ---- Step 4: LoRA ----
    model = apply_lora(model)

    # ---- Step 5: Training args ----
    training_args = get_training_args()

    # ---- Step 6: Trainer ----
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    # ---- Step 7: Resume logic ----
    last_checkpoint = find_last_checkpoint()
    if last_checkpoint:
        print(f"\n🔄 Resuming from checkpoint: {last_checkpoint}")
    else:
        print("\n🆕 No checkpoint found. Starting fresh.")

    # ---- Step 8: Train! ----
    print("\n" + "=" * 60)
    print("🏋️ Training started...")
    print("=" * 60 + "\n")

    trainer.train(resume_from_checkpoint=last_checkpoint)

    # ---- Step 9: Save LoRA adapter ----
    print(f"\n💾 Saving LoRA adapter to: {ADAPTER_DIR}")
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)

    # Print adapter size
    adapter_size = sum(
        f.stat().st_size for f in Path(ADAPTER_DIR).rglob("*") if f.is_file()
    )
    print(f"   Adapter size: {adapter_size / 1e6:.1f} MB")
    print("\n✅ Training complete!")

    return trainer

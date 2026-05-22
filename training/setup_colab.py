"""
Colab environment setup — run this FIRST in your notebook.

Handles:
1. Installing pip dependencies
2. Mounting Google Drive
3. Creating project directories
4. Detecting GPU type and recommending precision settings

Usage (first cell in Colab notebook):
    from training.setup_colab import setup
    gpu_info = setup()
"""

import os
import subprocess
import sys


def install_dependencies() -> None:
    """
    Install Unsloth + its pinned training stack for v2 (Plan A++).

    Per docs/BUG_TRACKING.md "v2 Unsloth Migration" section, we follow
    Unsloth's official install line rather than pinning transformers/trl
    ourselves — Unsloth pulls in the exact compatible versions as
    transitive deps. Manually pinning them here would fight Unsloth's
    resolver and re-ignite the SFTTrainer version war.

    TODO(plan-a++): After the 500-sample smoke test passes, lock this to a
    specific Unsloth commit SHA (not `main`) so re-runs stay reproducible:
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git@<sha>"

    Rollback: if Unsloth blocks progress, revert this function to the v1
    pins below and run Plan A (10K × 2 epochs, vanilla HF stack):
        "transformers>=4.46.0,<4.48.0"
        "trl>=0.12.0,<1.0.0"
        "peft>=0.10.0"
        "bitsandbytes>=0.43.0"
        "accelerate>=0.30.0"
        "datasets>=2.19.0"
    """

    # Unsloth's colab-new extra installs torch/xformers/bitsandbytes/peft/
    # trl/transformers at versions it has tested together. Leave version
    # resolution to Unsloth — do NOT add extra pins here.
    packages = [
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
        "datasets>=2.19.0",  # stack-agnostic, Unsloth doesn't pin this
    ]

    print("📦 Installing Unsloth + training stack...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q"] + packages,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    print("   ✅ Dependencies installed")

    # Echo versions so the smoke-test notebook output captures what actually
    # got resolved — critical for locking the commit SHA later.
    try:
        import transformers, trl, peft  # noqa
        import unsloth  # noqa
        print(
            f"   unsloth={unsloth.__version__}, "
            f"transformers={transformers.__version__}, "
            f"trl={trl.__version__}, peft={peft.__version__}"
        )
    except Exception as e:
        print(f"   ⚠️  Could not print resolved versions: {e}")


def mount_google_drive() -> None:
    """Mount Google Drive at /content/drive."""

    try:
        from google.colab import drive

        if not os.path.ismount("/content/drive"):
            drive.mount("/content/drive")
            print("📁 Google Drive mounted")
        else:
            print("📁 Google Drive already mounted")
    except ImportError:
        print("⚠️  Not running on Colab — skipping Drive mount")
        print("   (Paths will still be created but may not persist)")


def create_directories() -> None:
    """Create project directories on Google Drive."""

    from training.config import (
        ADAPTER_DIR,
        CACHED_DATASET_DIR,
        CHECKPOINT_DIR,
        DRIVE_PROJECT_DIR,
        LOGGING_DIR,
    )

    dirs = [DRIVE_PROJECT_DIR, CHECKPOINT_DIR, LOGGING_DIR, ADAPTER_DIR, CACHED_DATASET_DIR]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    print(f"📂 Project directories created at: {DRIVE_PROJECT_DIR}")


def detect_gpu() -> dict:
    """
    Detect GPU type and recommend precision settings.

    Returns:
        Dict with gpu_name, vram_gb, recommended_precision, and recommended_batch_size.
    """

    import torch

    if not torch.cuda.is_available():
        print("⚠️  No GPU detected! Training will be extremely slow.")
        return {"gpu_name": "None", "vram_gb": 0, "precision": "fp32", "batch_size": 1}

    gpu_name = torch.cuda.get_device_name(0)
    vram_bytes = torch.cuda.get_device_properties(0).total_memory
    vram_gb = vram_bytes / (1024**3)

    # Recommend settings based on GPU
    if "A100" in gpu_name:
        precision = "bf16"
        batch_size = 8
    elif "V100" in gpu_name:
        precision = "fp16"
        batch_size = 4
    elif "T4" in gpu_name:
        precision = "fp16"
        batch_size = 4
    elif "L4" in gpu_name:
        precision = "bf16"
        batch_size = 4
    else:
        precision = "fp16"
        batch_size = 2

    info = {
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "precision": precision,
        "batch_size": batch_size,
    }

    print(f"\n🖥️  GPU Detected: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    print(f"   Recommended precision: {precision}")
    print(f"   Recommended batch size: {batch_size}")

    # Check if config matches recommendation
    from training.config import BATCH_SIZE, USE_BF16, USE_FP16

    current_precision = "bf16" if USE_BF16 else ("fp16" if USE_FP16 else "fp32")
    if current_precision != precision:
        print(f"\n   ⚠️  Your config uses {current_precision}, but {precision} is recommended for {gpu_name}.")
        print("      Update USE_FP16/USE_BF16 in training/config.py if needed.")

    if BATCH_SIZE != batch_size:
        print(f"   ⚠️  Your config uses batch_size={BATCH_SIZE}, recommended: {batch_size}")

    return info


def setup() -> dict:
    """
    Full Colab setup — call this first.

    Returns:
        GPU info dict.
    """

    print("=" * 60)
    print("⚙️  PR Review AI — Colab Setup")
    print("=" * 60 + "\n")

    install_dependencies()
    mount_google_drive()
    create_directories()
    gpu_info = detect_gpu()

    print("\n" + "=" * 60)
    print("✅ Setup complete! Ready to train.")
    print("=" * 60)

    return gpu_info

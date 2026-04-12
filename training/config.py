"""
Centralized configuration for LoRA fine-tuning.

All paths, hyperparameters, and model settings are defined here.
Change values here to tune the training without touching other files.
"""

import os


# =============================================================================
# Google Drive Paths (persist across Colab disconnects)
# =============================================================================

DRIVE_PROJECT_DIR = "/content/drive/MyDrive/pr-review-ai"
CHECKPOINT_DIR = os.path.join(DRIVE_PROJECT_DIR, "checkpoints")
LOGGING_DIR = os.path.join(DRIVE_PROJECT_DIR, "logs")
ADAPTER_DIR = os.path.join(DRIVE_PROJECT_DIR, "code-review-lora-adapter")
CACHED_DATASET_DIR = os.path.join(DRIVE_PROJECT_DIR, "dataset-cache")


# =============================================================================
# Dataset
# =============================================================================

DATASET_NAME = "ronantakizawa/github-codereview"
LANGUAGE_FILTER = "Python"          # Filter by file-level language
QUALITY_THRESHOLD = 0.5             # Minimum quality_score (0.0 - 1.0)
EXCLUDE_NEGATIVE = True             # v3: exclude is_negative=True ("no issues found") examples — 25% of data was teaching the model to say "no issues found"
MIN_CODE_LINES = 5                  # Minimum lines in before/after code
MAX_CODE_LINES = 200                # Maximum lines in before/after code
TRAIN_SAMPLE_LIMIT = 15000          # v2 (Plan A++): 10K → 15K. Unsloth's ~2x speedup makes 15K×2 epochs fit in ~20h / ~40 CU on T4.


# =============================================================================
# Model
# =============================================================================

MODEL_NAME = "codellama/CodeLlama-7b-Instruct-hf"
MAX_SEQ_LENGTH = 2048               # Max tokens per training sample


# =============================================================================
# LoRA Hyperparameters
# =============================================================================

LORA_R = 16                         # Rank (8-64; higher = more capacity, more VRAM)
LORA_ALPHA = 32                     # Scaling factor (typically 2x of r)
LORA_DROPOUT = 0.05                 # Dropout for regularization
LORA_TARGET_MODULES = [             # Attention layers to insert LoRA into
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


# =============================================================================
# Training Hyperparameters
# =============================================================================

NUM_EPOCHS = 1                      # v3: back to 1 — v2 showed epoch 2 added memorization, not generalization (val loss flat)
BATCH_SIZE = 1                      # Per-device batch size (T4 needs 1 with 2048 seq len)
GRADIENT_ACCUMULATION_STEPS = 8     # Effective batch = BATCH_SIZE * this = 8
LEARNING_RATE = 2e-4
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
LR_SCHEDULER = "cosine"

# Checkpointing
SAVE_STEPS = 200
SAVE_TOTAL_LIMIT = 3

# Evaluation
EVAL_STEPS = 200

# Precision (set ONE to True based on your GPU)
USE_FP16 = True                     # For T4
USE_BF16 = False                    # For A100 (set USE_FP16=False if using this)


# =============================================================================
# Inference
# =============================================================================

GENERATION_MAX_TOKENS = 512
GENERATION_TEMPERATURE = 0.7
GENERATION_TOP_P = 0.9
GENERATION_REPETITION_PENALTY = 1.1


# =============================================================================
# Prompt Templates
# =============================================================================

PROMPT_TEMPLATE = """### Instruction:
You are a senior code reviewer. Compare the before and after versions of the code below. Identify potential issues and provide improvement suggestions.

### File: {file_path}

### Before:
{before_code}

### After:
{after_code}

### Review:
{reviewer_comment}"""

INFERENCE_TEMPLATE = """### Instruction:
You are a senior code reviewer. Compare the before and after versions of the code below. Identify potential issues and provide improvement suggestions.

### File: {file_path}

### Before:
{before_code}

### After:
{after_code}

### Review:
"""

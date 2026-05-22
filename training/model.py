"""
Model loading, quantization, and LoRA configuration.

v2 (Plan A++): Training path migrated to Unsloth's FastLanguageModel for
~2x speedup and ~40-50% less VRAM via custom Triton kernels for the LoRA
forward/backward. Inference path deliberately stays on vanilla
Transformers + PEFT so the API server (api/model_service.py) doesn't need
Unsloth as a runtime dependency — adapters trained under Unsloth load
bit-identically via peft.PeftModel.from_pretrained.

See docs/v2_evaluation.md and docs/BUG_TRACKING.md (Unsloth section) for
the migration rationale and version-compatibility caveats.
"""

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from training.config import (
    ADAPTER_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MAX_SEQ_LENGTH,
    MODEL_NAME,
)


def load_base_model():
    """
    Load base model + tokenizer via Unsloth FastLanguageModel (4-bit QLoRA).

    Unsloth bundles the 4-bit quantization config, dtype selection, and
    tokenizer setup into one call, so we no longer build a BitsAndBytesConfig
    or call prepare_model_for_kbit_training manually.

    `dtype=None` lets Unsloth auto-pick (bf16 on Ampere+, fp16 on T4) — this
    is the recommended default and matches our USE_FP16=True config on T4.
    """
    # IMPORTANT: unsloth must be imported before transformers/trl so its
    # monkey-patches take effect. Lazy-import here because the API server
    # (which imports training.model indirectly via training.inference) must
    # NOT require unsloth at runtime.
    from unsloth import FastLanguageModel

    print(f"🧠 Loading base model via Unsloth: {MODEL_NAME}")
    print(f"   Quantization: 4-bit (Unsloth-managed), max_seq_length={MAX_SEQ_LENGTH}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,              # auto: bf16 on Ampere+, fp16 on T4
        load_in_4bit=True,
    )

    # Match v1 tokenizer setup so data_prep / packing behave identically.
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("   ✅ Base model loaded")
    return model, tokenizer


def apply_lora(model):
    """
    Attach LoRA adapters via Unsloth's patched get_peft_model.

    This replaces the v1 pipeline of (prepare_model_for_kbit_training →
    LoraConfig → peft.get_peft_model). Unsloth's version:
      - Installs the Triton fused LoRA kernels (the whole point of migrating)
      - Handles kbit prep internally
      - Uses its own VRAM-efficient gradient checkpointing impl
        via use_gradient_checkpointing="unsloth"
    """
    from unsloth import FastLanguageModel

    print(f"\n🔧 Applying LoRA via Unsloth (r={LORA_R}, alpha={LORA_ALPHA})")
    print(f"   Target modules: {LORA_TARGET_MODULES}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    model.print_trainable_parameters()
    return model


def load_for_inference(adapter_path: str | None = None):
    """
    Load base model + trained LoRA adapter for inference.

    Intentionally uses vanilla Transformers + PEFT (NOT Unsloth) so this path
    works on the API server / Docker image without pulling in Unsloth and its
    tight version pins. Unsloth-trained adapters are fully compatible with
    peft.PeftModel.from_pretrained at inference time — this is verified as a
    pre-flight check before kicking off the v2 training run (see
    docs/v2_evaluation.md "Pre-flight checks").
    """
    if adapter_path is None:
        adapter_path = ADAPTER_DIR

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"🧠 Loading base model for inference: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"🔌 Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("   ✅ Model ready for inference")
    return model, tokenizer

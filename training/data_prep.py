"""
Dataset loading, filtering, and prompt formatting.

Handles:
- Loading from Hugging Face or cached copy on Google Drive
- Filtering by language, quality, and code length
- Formatting into training prompt templates
- Token-length filtering (requires tokenizer)
"""

import os

from datasets import DatasetDict, load_dataset

from training.config import (
    CACHED_DATASET_DIR,
    DATASET_NAME,
    EXCLUDE_NEGATIVE,
    LANGUAGE_FILTER,
    MAX_CODE_LINES,
    MAX_SEQ_LENGTH,
    MIN_CODE_LINES,
    PROMPT_TEMPLATE,
    QUALITY_THRESHOLD,
    TRAIN_SAMPLE_LIMIT,
)


def load_and_filter_dataset() -> DatasetDict:
    """
    Load the code review dataset, filter for Python + quality, and cache to Drive.

    On first run: downloads from HuggingFace, filters, saves to Google Drive.
    On subsequent runs: loads directly from Drive cache (skips 653MB download).

    Returns:
        DatasetDict with train/validation/test splits, filtered and ready.
    """

    cache_marker = os.path.join(CACHED_DATASET_DIR, "dataset_dict.json")

    if os.path.exists(cache_marker):
        print("📦 Loading cached filtered dataset from Google Drive...")
        dataset = DatasetDict.load_from_disk(CACHED_DATASET_DIR)
        print(
            f"   Loaded: train={len(dataset['train']):,}, "
            f"val={len(dataset['validation']):,}, "
            f"test={len(dataset['test']):,}"
        )
        return dataset

    print("🌐 No cache found. Downloading from Hugging Face...")
    dataset = load_dataset(DATASET_NAME)

    # --- Filter 1: Python files only ---
    print(f"🔍 Filtering: language == '{LANGUAGE_FILTER}'")
    dataset = dataset.filter(lambda x: x["language"] == LANGUAGE_FILTER)

    # --- Filter 2: Quality threshold ---
    print(f"🔍 Filtering: quality_score >= {QUALITY_THRESHOLD}")
    dataset = dataset.filter(lambda x: x["quality_score"] >= QUALITY_THRESHOLD)

    # --- Filter 3: Exclude negative examples (v3) ---
    if EXCLUDE_NEGATIVE:
        print("🔍 Filtering: excluding is_negative=True ('no issues found' examples)")
        before = len(dataset["train"])
        dataset = dataset.filter(lambda x: not x["is_negative"])
        after = len(dataset["train"])
        print(f"   Removed {before - after:,} negative examples from train split")

    # --- Filter 4: Code length bounds ---
    print(f"🔍 Filtering: code lines in [{MIN_CODE_LINES}, {MAX_CODE_LINES}]")
    dataset = dataset.filter(
        lambda x: (
            MIN_CODE_LINES <= x["before_lines"] <= MAX_CODE_LINES
            and MIN_CODE_LINES <= x["after_lines"] <= MAX_CODE_LINES
        )
    )

    # --- Filter 5: Subsample training set to fit Colab time limits ---
    if TRAIN_SAMPLE_LIMIT and len(dataset["train"]) > TRAIN_SAMPLE_LIMIT:
        print(f"🎲 Subsampling train set: {len(dataset['train']):,} → {TRAIN_SAMPLE_LIMIT:,}")
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(TRAIN_SAMPLE_LIMIT))

    # Save to Google Drive
    os.makedirs(CACHED_DATASET_DIR, exist_ok=True)
    dataset.save_to_disk(CACHED_DATASET_DIR)
    print(f"💾 Filtered dataset cached to {CACHED_DATASET_DIR}")

    _print_dataset_stats(dataset)
    return dataset


def format_prompts(dataset: DatasetDict) -> DatasetDict:
    """
    Add a 'text' column with the formatted training prompt for each example.

    Uses before_code + after_code (not diff_context) for full reviewer context.
    """

    def _format_single(example: dict) -> dict:
        return {
            "text": PROMPT_TEMPLATE.format(
                file_path=example["file_path"],
                before_code=example["before_code"],
                after_code=example["after_code"],
                reviewer_comment=example["reviewer_comment"],
            )
        }

    print("📝 Formatting prompts (before_code + after_code → review)...")
    dataset = dataset.map(_format_single)
    return dataset


def filter_by_token_length(dataset: DatasetDict, tokenizer) -> DatasetDict:
    """
    Remove samples whose formatted text exceeds MAX_SEQ_LENGTH tokens.

    Must be called after format_prompts() and after tokenizer is loaded.
    """

    def _within_limit(example: dict) -> bool:
        token_ids = tokenizer(example["text"], truncation=False)["input_ids"]
        return len(token_ids) <= MAX_SEQ_LENGTH

    before_count = len(dataset["train"])
    print(f"✂️  Filtering by token length (max {MAX_SEQ_LENGTH} tokens)...")
    dataset = dataset.filter(_within_limit)
    after_count = len(dataset["train"])
    dropped = before_count - after_count
    print(f"   Dropped {dropped:,} train samples ({dropped/before_count:.1%}) exceeding token limit")

    return dataset


def _print_dataset_stats(dataset: DatasetDict) -> None:
    """Print summary statistics for the filtered dataset."""

    print("\n" + "=" * 50)
    print("📊 Dataset Statistics")
    print("=" * 50)

    for split_name in ["train", "validation", "test"]:
        if split_name in dataset:
            print(f"  {split_name:>12}: {len(dataset[split_name]):>8,} samples")

    train = dataset["train"]
    df = train.to_pandas()

    print("\n📂 Comment type distribution:")
    for ctype, count in df["comment_type"].value_counts().items():
        pct = count / len(df) * 100
        print(f"  {ctype:>20}: {count:>6,} ({pct:.1f}%)")

    neg_count = df["is_negative"].sum()
    print(f"\n🔄 Negative examples: {neg_count:,}/{len(df):,} ({neg_count/len(df):.1%})")
    print(f"📊 Quality score: mean={df['quality_score'].mean():.3f}, "
          f"median={df['quality_score'].median():.3f}")
    print("=" * 50 + "\n")


"""Prepare paired datasets for Standard SFT vs COCA SFT."""

import json
import random
from pathlib import Path

SEED = 42
N_TRAIN = 1000
N_VAL = 200

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COCA_ROOT = PROJECT_ROOT.parent / "COCA"

SOURCE_DIR = COCA_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def load_dataset(filename):
    path = SOURCE_DIR / filename

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def has_coca_structure(output):
    """Check whether COCA-specific sections are present."""
    tags = [
        "<think>", "</think>",
        "<concept>", "</concept>",
        "<response>", "</response>",
    ]
    return all(tag in output for tag in tags)


def index_by_instruction(dataset, require_coca=False):
    """Deduplicate examples using their instruction."""
    indexed = {}

    for example in dataset:
        instruction = example["instruction"].strip()

        if require_coca and not has_coca_structure(example["output"]):
            continue

        if instruction and instruction not in indexed:
            indexed[instruction] = example

    return indexed


def main():
    normal = load_dataset("safety_beaver_normal.json")
    coca = load_dataset("safety_beaver_alpaca.json")

    normal_index = index_by_instruction(normal)
    coca_index = index_by_instruction(coca, require_coca=True)

    common = sorted(set(normal_index) & set(coca_index))

    rng = random.Random(SEED)
    rng.shuffle(common)

    assert len(common) >= N_TRAIN + N_VAL, (
        "Not enough paired examples."
    )

    splits = {
        "train": common[:N_TRAIN],
        "val": common[N_TRAIN:N_TRAIN + N_VAL],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split, instructions in splits.items():
        for condition, index in [
            ("normal", normal_index),
            ("coca", coca_index),
        ]:
            examples = [index[key] for key in instructions]

            output_path = OUTPUT_DIR / f"{condition}_{split}.json"

            with output_path.open("w", encoding="utf-8") as file:
                json.dump(examples, file, indent=2, ensure_ascii=False)

            print(f"Saved {len(examples)} examples to {output_path}")

    assert not (set(splits["train"]) & set(splits["val"]))

    print(f"\nEligible paired instructions: {len(common)}")
    print(f"Training pairs: {N_TRAIN}")
    print(f"Validation pairs: {N_VAL}")
    print("Train/validation overlap: 0")


if __name__ == "__main__":
    main()

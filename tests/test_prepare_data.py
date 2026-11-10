
"""Tests for paired COCA datasets."""

import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load(condition, split):
    path = DATA_DIR / f"{condition}_{split}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def instructions(rows):
    return [row["instruction"] for row in rows]


def test_dataset_sizes():
    for condition in ("normal", "coca"):
        assert len(load(condition, "train")) == 1000
        assert len(load(condition, "val")) == 200


def test_paired_instructions():
    for split in ("train", "val"):
        normal = load("normal", split)
        coca = load("coca", split)

        assert instructions(normal) == instructions(coca)


def test_no_data_leakage():
    train = set(instructions(load("normal", "train")))
    val = set(instructions(load("normal", "val")))

    assert train.isdisjoint(val)


def test_no_duplicate_instructions():
    for condition in ("normal", "coca"):
        for split in ("train", "val"):
            prompts = instructions(load(condition, split))
            assert len(prompts) == len(set(prompts))


def test_coca_structure():
    tags = (
        "<think>", "</think>",
        "<concept>", "</concept>",
        "<response>", "</response>",
    )

    for split in ("train", "val"):
        for example in load("coca", split):
            output = example["output"]
            assert all(tag in output for tag in tags)


def test_required_fields():
    for condition in ("normal", "coca"):
        for split in ("train", "val"):
            for example in load(condition, split):
                assert all(
                    key in example
                    for key in ("instruction", "input", "output")
                )
                assert isinstance(example["output"], str)
                assert example["output"].strip()

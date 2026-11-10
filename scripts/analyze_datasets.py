
"""Exploratory analysis of paired COCA datasets."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "results" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load(condition, split):
    path = DATA_DIR / f"{condition}_{split}.json"
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_section(text, tag):
    match = re.search(
        rf"<{tag}>(.*?)</{tag}>",
        text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else ""


records = []

for split in ("train", "val"):
    standard = load("normal", split)
    coca = load("coca", split)

    for condition, examples in [
        ("standard", standard),
        ("coca", coca),
    ]:
        for example in examples:
            output = example["output"]

            record = {
                "split": split,
                "condition": condition,
                "instruction": example["instruction"],
                "n_chars": len(output),
                "n_words": len(output.split()),
            }

            if condition == "coca":
                for tag in ("think", "concept", "response"):
                    section = extract_section(output, tag)
                    record[f"{tag}_words"] = len(section.split())

            records.append(record)

df = pd.DataFrame(records)

# 1. Summary statistics
summary = (
    df.groupby(["split", "condition"])["n_words"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)

print("\nResponse length statistics:")
print(summary.round(2))

summary.to_csv(OUTPUT_DIR / "length_summary.csv")

# 2. Distribution of response lengths
fig, ax = plt.subplots(figsize=(9, 5))

for condition in ("standard", "coca"):
    values = df[
        (df["split"] == "train")
        & (df["condition"] == condition)
    ]["n_words"]

    ax.hist(
        values,
        bins=40,
        alpha=0.5,
        label=condition,
        density=True,
    )

ax.set_xlabel("Response length (words)")
ax.set_ylabel("Density")
ax.set_title("Training response length distributions")
ax.legend()

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "response_lengths.png", dpi=200)
plt.close(fig)

# 3. Paired length ratios
train = df[df["split"] == "train"]

pivot = train.pivot(
    index="instruction",
    columns="condition",
    values="n_words",
)

ratio = pivot["coca"] / pivot["standard"].replace(0, np.nan)

print(f"\nMedian COCA/Standard length ratio: {ratio.median():.2f}")

# 4. COCA section lengths
coca_train = df[
    (df["split"] == "train")
    & (df["condition"] == "coca")
]

section_columns = [
    "think_words",
    "concept_words",
    "response_words",
]

section_means = coca_train[section_columns].mean()

fig, ax = plt.subplots(figsize=(7, 4))
section_means.plot.bar(ax=ax)

ax.set_ylabel("Mean number of words")
ax.set_title("COCA response structure")
ax.tick_params(axis="x", rotation=0)

fig.tight_layout()
fig.savefig(OUTPUT_DIR / "coca_sections.png", dpi=200)
plt.close(fig)

df.to_csv(OUTPUT_DIR / "dataset_statistics.csv", index=False)

print(f"\nResults saved in {OUTPUT_DIR}")

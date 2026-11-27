"""A no-dependency class demonstration that reads frozen aggregate results."""
import json
from pathlib import Path
from statistics import mean

data = json.loads(Path(__file__).with_name("evidence.json").read_text())
print("Can a reader recover the original maze clue?")
for description, stage in (("Starting picture", "raw"), ("Processed picture", "encoder"),
                           ("Chances for each category", "probabilities"),
                           ("One sampled summary", "sample")):
    approximate = "≈" if stage == "sample" else ""
    print(f"{description:26s} {approximate}{mean(data['reset'][stage]):.0%}")
print("Chance with two equally likely clues: 50%")

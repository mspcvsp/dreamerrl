"""A no-dependency class demonstration that reads frozen aggregate results."""
import json
from pathlib import Path
from statistics import mean

data = json.loads(Path(__file__).with_name("evidence.json").read_text())
print("Can a reader recover the original maze clue?")
for description, stage in (("Encoder", "encoder"), ("Soft probabilities", "probabilities"),
                           ("One hard sample", "sample")):
    print(f"{description:20s} {mean(data['reset'][stage]):.0%}")
print("Chance with two equally likely clues: 50%")

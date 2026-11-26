"""Read-only adapter for matched T-maze trajectory artifacts."""
import json

import numpy as np
import torch

from dreamerrl.eval.memory.interfaces import Episode, Labels, paired_episodes


CUE_CELL = 7


class TMazeArchive:
    def __init__(self, path):
        data = json.loads(path.read_text())
        self.rows = {}
        for condition in ("paired", "neutral", "identical", "irrelevant"):
            self.rows[condition] = {}
            for row in data[condition]:
                group, cue = row["seed"], row["cue"]
                trace = row["trace"]
                if [f["lag"] for f in trace] != list(range(9)):
                    raise ValueError("Unexpected T-maze lag or horizon")
                obs = torch.tensor([f["glyphs"] for f in trace]).flatten(1)
                actions = torch.tensor([f["action"] for f in trace])
                rewards = torch.tensor([f["reward"] for f in trace])
                if actions[:-1].any() or rewards[:-1].any() or rewards[-1] != (1 if actions[-1] == cue else -1):
                    raise ValueError("Unexpected stem action or branch reward")
                if [f["terminal"] for f in trace] != [False] * 8 + [True]:
                    raise ValueError("Unexpected episode boundary")
                self.rows[condition][group, cue] = Episode((group, cue), group, obs, actions, rewards, True, False)

    def collect(self, episode_ids, condition="paired"):
        return [self.rows[condition][key] for key in episode_ids]

    def labels(self, episode_id):
        return Labels(np.array(episode_id[1]), np.array(True))

    def model_input(self, observations):
        return observations.float()

    def matched_arrays(self, levels):
        ids = [(level, cue) for level in levels for cue in (0, 1)]
        paired, neutral, identical, irrelevant = [self.collect(ids, condition) for condition in self.rows]
        for i in range(0, len(paired), 2):
            a, b = paired[i:i+2]
            paired_episodes(a, b, 0)
            if not torch.equal((a.observations != b.observations).nonzero(), torch.tensor([[0, CUE_CELL]])):
                raise ValueError("The cue must change exactly one cell")
            if not torch.equal(neutral[i].observations, neutral[i+1].observations):
                raise ValueError("Neutral cue histories differ")
        for i, row in enumerate(paired):
            for other, expected in ((neutral[i], [[0, CUE_CELL]]), (irrelevant[i], [[0, 15]]), (identical[i], [])):
                paired_episodes(row, other, 0)
                if (row.observations != other.observations).nonzero().tolist() != expected:
                    raise ValueError("Archived intervention differs outside its designed input")
        return {name: self.model_input(torch.stack([e.observations for e in cases]))
                for name, cases in zip(self.rows, (paired, neutral, identical, irrelevant), strict=True)}, torch.stack(
                    [e.actions for e in paired])

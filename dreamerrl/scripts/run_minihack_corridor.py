"""Train and evaluate Dreamer on MiniHack corridors connecting two rooms.

Uses the standard MiniHack runner, including random/untrained baselines,
fixed validation seeds, checkpoints, and resumable training. Select larger
layouts with --env MiniHack-Corridor-R3-v0 or MiniHack-Corridor-R5-v0.
"""

from dreamerrl.scripts.run_minihack import main as run_minihack


def main():
    run_minihack(default_env="MiniHack-Corridor-R2-v0", default_episode_steps=1000)


if __name__ == "__main__":
    main()

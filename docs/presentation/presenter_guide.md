# Presenter guide

**Learning Through Imagination** is a 25–30 minute session for Scouts new to AI. Bring two clue cards labeled LEFT and RIGHT, a sheet to cover a card, and the PDF. The visual lesson can run with the PDF alone. Reserve about five minutes for discussion.

| Slide | What to say or do | Ask the audience |
|---|---|---|
| 1 · Opening | Explain that this project tested a learning agent in a model of a computer network. Use the phrase “practice world.” | Where have you practiced before doing the real thing? |
| 2 · Network | A signal can mean something needs attention. The defender sees part of the situation. Use a smoke-alarm analogy without naming attack methods. | What might a defender want to protect? |
| 3 · Reinforcement learning | Observation is what the agent sees. Action is what it tries. Reward is feedback about the result. | If a reward gives the wrong signal, what might the agent learn? |
| 4 · Feedback | Choose a route to camp. Announce feedback such as a wrong turn or taking less time. Ask for a revised choice. | What other reward might you care about besides speed? |
| 5 · Imagination | Before moving, picture taking either path. The world model’s predictions can guide practice and can be wrong. | When would you stop trusting a prediction? |
| 6 · DreamerV3 | The project builds on a published algorithm. The model learns a compact description and uses it to imagine possible experiences. | Why might a compact description help? What might it leave out? |
| 7 · CAGE2 | The diagram is conceptual. Researchers test on simulated systems with permission, using a known scenario and scoring rules. | What could go wrong if someone tried the same tests on a real network? |
| 8 · Performance | In this archived comparison the flat agent made each choice directly. The other used a higher-level controller as well. Higher scores are better. B-line and Meander are simulator opponent names. Sleep is idle. The slide shows averages to one decimal across three training seeds. | Which design looks better for B-line? Does your answer change for Meander? |
| 9 · T-maze | Show one card for two seconds, cover it, then count eight pretend steps together. Ask everyone to point to the rewarded branch. Repeat with the other cue. Keep the gestures and delay identical. | What information did the path itself give you? |
| 10 · Bottleneck | Ask one Scout to write “maze” only and pass that note forward. Ask a second to write “maze, left” or “maze, right.” Both summaries save space, but only one keeps the needed clue. In the recorded model runs, the cue was readable in the probabilities. A single hard choice left about a coin flip’s worth of accuracy. | When is a short summary too short? |
| 11 · Utilization | In a second round, let a Scout hold the clue card while another person always picks left. Remembering the information and using it are separate. The frozen model’s special-state intervention showed the same separation in this test. | How would you tell whether a decision used the clue? |
| 12 · Tools | Python ran the software. Git and GitHub kept reviewable versions. An AI assistant helped write checks, and people verified the claims against saved results. | What would you test before trusting a generated change? |
| 13 · Demo | Show the saved stage figure or run the command below. Keep the discussion focused on predictions and evidence. | Which stages would you investigate first? |
| 14 · Careers | Give examples of software engineering, defensive monitoring, and experimental research. Recommend learning Python and collaborating on a small project. | Which skill would you try this month and why? |
| 15 · Wrap | Invite questions and show the source links. Ask participants to explain “read the clue” and “use the clue” in their own words. | What would make your proposed test fair? |

## Three-minute CPU demo

From the repository root, run this with Python 3. It reads the tracked, aggregate snapshot extracted by the existing diagnostic reporting workflow. It needs no simulator, weights, packages, or internet connection.

```bash
python3 docs/publication_figures/read_saved_results.py
```

Before revealing the output, ask the group to predict which stage first drops near chance. Read the three printed numbers, then compare them with slide 10. To run the lesson without a terminal, use slide 10 and reveal the bars one at a time. An educator can also open `docs/publication_figures/evidence.json` and find the same values under `reset`.

The case study uses frozen results. The live clue game is a teaching exercise. It supplies no new evidence about the trained agent.

## Accessibility and setup

Read the chart labels aloud. Use the words LEFT and RIGHT on clue cards so the exercise works without color recognition. Let Scouts point, speak, or write their answer. If the projector is small, zoom the deck to fit the screen and read the chart values aloud. The repository and paper links appear on the final slide.

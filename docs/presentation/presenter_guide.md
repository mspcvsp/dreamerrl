# Presenter guide

**Learning Through Imagination** is a 25–30 minute Scout lesson. Bring two clue cards labeled LEFT and RIGHT, a sheet to cover a card, and the deck. Reserve five minutes for questions.

| Slide | What to say or do | Ask the group |
|---|---|---|
| 1 · Can you use AI to build AI? | Introduce two stories. An agent learned in a practice world. A person directed AI assistants to help build and test that agent. | What might you build if you could get help with one difficult part? |
| 2 · What is AI? | Give examples like finding a pattern, drafting code and learning from feedback. Explain that different AI systems do different jobs. | Which of those jobs needs feedback from the world? |
| 3 · Reinforcement learning | An observation is what the agent sees. An action is what it tries. A reward is feedback. Let the group pick a route to camp and revise it after feedback. | What should count as a good route besides speed? |
| 4 · Imagination | Ask Scouts to picture two routes before moving. DreamerV3 predicts what might happen and practices from those predictions. | How could you find out whether a prediction is reliable? |
| 5 · Safe simulation | CAGE2 provides a practice network. The diagram is conceptual. Try computer-security exercises only in approved learning environments. | Why would we keep the exercise off a real network? |
| 6 · Development tools | Name Microsoft Copilot and OpenAI Codex. I uploaded text to Microsoft Copilot manually, outside VS Code. Both tools assisted the project. DreamerV3 learned in the simulator. GitHub hosts the project, and Git records changes. | Which tool helped with development, and which system learned to choose actions? |
| 7 · The conductor | A conductor sets the goal, explains requirements, tests outputs, reviews evidence and decides what to change. Point out the return arrow. | If AI helps write the code, what is the human’s job? How could you explain a task clearly enough to get help? |
| 8 · Build it in pieces | I directed work on smaller pieces such as reading an observation, testing a model component and checking a result. The assistants helped implement and debug them. | What small piece would you build and check first? |
| 9 · Performance | Flat agents choose actions directly. The other agents had two decision levels. These are local simulator averages over three training seeds. Higher scores are better. | Does the same design lead in both active opponent scenarios? |
| 10 · T-maze exercise | Show a card briefly, cover it and count eight pretend steps. Ask Scouts to indicate the branch. Repeat with the other cue using the same delay and gestures. | Could you work out the answer from your current view alone? |
| 11 · Losing the clue | A reader could find the clue in the model’s probabilities. One hard choice compressed the distinction to about chance in this test. A note saying only “maze” illustrates what compression can leave out. | When is a short summary too short? |
| 12 · Verify | The frozen intervention made the clue readable at the turn, yet the model’s choices stayed at chance. A plausible earlier plot label also needed correction after tracing the code. | What would you do if an AI answer sounded right but failed a test? How would you check AI-generated code? |
| 13 · Start small | Keep the saved-results demo below. Ask for help with one part of a small supervised project. Inspect outputs before changing the next part. | What STEM project would you try with AI assistance? |
| 14 · Build skills | Practice Python and basic math. Learn to describe a problem, design a fair test and explain why a result is trustworthy. | Why does knowing the underlying science help you direct or correct an assistant? |
| 15 · Keep learning | Invite each Scout to suggest a project and a check that could prove the first answer wrong. | What would you change if your first test failed? |

## Three-minute CPU demo

From the repository root, run:

```bash
python3 docs/publication_figures/read_saved_results.py
```

Predict which stage first drops near chance, then compare the printed values with slide 11. The command reads the tracked aggregate snapshot and needs no packages, weights, simulator, internet connection or training. Without a terminal, use slide 11 and reveal each bar in order.

## Supervised experimentation

Use age-appropriate tools with adult supervision and follow the tool and group policies. Keep names, credentials, private files and network details out of prompts. Start with made-up data, a local toy project or an approved simulator. Have an adult review generated commands before anyone runs them. Begin with a small test that could reveal a mistake. Curiosity, learning, experimentation and verification are engineering skills.

Read labels aloud and use the words LEFT and RIGHT on clue cards. Let Scouts point, speak or write their answers.

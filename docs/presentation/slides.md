---
title: "Learning Through Imagination"
subtitle: "Exploring Artificial Intelligence and Cybersecurity"
---

## Can AI help defend a network?

![A computer network has a defender, workstations, and a service.](figures/network.png){width=94%}

A defender sees clues and has to decide what to do.

## What is reinforcement learning?

![Agent and environment exchange actions, observations, and rewards.](figures/loop.png){width=94%}

The agent learns which choices tend to earn better rewards.

## Try an action. Get feedback.

- What can you see?
- What can you do?
- How will you know it helped?

Try a route to camp. The feedback helps you choose next time.

## Can an AI practice in its imagination?

![One observation leads to two imagined choices.](figures/imagine.png){width=94%}

A world model predicts what could happen after a choice.

## Meet DreamerV3

![Three stages show seeing the world, building a model, and practicing choices.](figures/architecture.png){width=94%}

Dreamer uses its predictions to practice without taking every action in the simulator.

## How can we test cyber defense safely?

![The simulated Blue defender observes workstations and a service.](figures/network.png){width=94%}

CAGE2 gives a defensive agent a practice network. All activity stays inside the simulator.

## Did more planning levels help?

![Flat and two-level agents had similar average simulator scores.](figures/behavior.png){width=94%}

Higher scores are better. Neither design led against every simulated opponent.

## Can you remember the clue?

![A clue at the start of a T-shaped maze disappears before the turn.](figures/maze.png){width=94%}

Remember LEFT or RIGHT. I’ll cover the clue while we take eight steps.

## Where did the AI lose the clue?

![The clue is readable in the scene and probabilities, but chance after one hard choice.](figures/bottleneck.png){width=94%}

The model kept a faint clue until it picked one category.

## Is remembering enough?

![A special reset made the clue readable at the turn, with chance-level choices.](figures/intervention.png){width=94%}

When researchers gave the model two distinct starting states, its branch choices stayed at chance.

## How did the project come together?

![Read on GitHub, view a result, and check the evidence.](figures/workflow.png){width=94%}

Python ran the tools. GitHub tracked changes. An AI assistant helped write tests.

## What can you try?

- Start with the saved T-maze result.
- Predict which stage loses the clue.
- Compare your prediction with the evidence.

Open-source tools let you begin with an existing algorithm.

## What skills might you build?

![Building software, defending systems, and testing ideas are three paths.](figures/careers.png){width=94%}

Start with Python, a small project, and a teammate.

## What would you test next?

- [Explore DreamerRL on GitHub](https://github.com/mspcvsp/dreamerrl)
- [Read about DreamerV3](https://www.nature.com/articles/s41586-025-08744-2)
- [Explore the CAGE2 simulator](https://github.com/cage-challenge/cage-challenge-2)

Which clue would you keep, and how would you check that the AI used it?

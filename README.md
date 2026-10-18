# ARC-AGI-3 — Heuristic Explorer Agent

An AI agent that plays the games of the **ARC Prize 2026 — ARC-AGI-3**
competition, plus the small set of tools used to build, test and submit it.

## Why we built this

We wanted to enter ARC Prize 2026 on Kaggle — one of the hardest "can an AI
actually think?" challenges around — and learn first-hand how to build an agent
that has to figure out a game by itself. This repository is the record of that
attempt: the agent, the notebooks we submitted, and the offline test rig we used
to try ideas before spending a Kaggle submission.

## What the competition asks you to do

ARC-AGI-3 is a set of **interactive games the agent has never seen before**.

- Every game is a turn-based world drawn on a **64×64 grid** of colours.
- The agent is given **only the current screen**. There are no instructions, no
  rules, no explanation of what the colours or objects mean.
- By acting — moving, or clicking a point on the grid — the agent sees what
  changes and gradually has to work out the rules of the game and complete its
  levels.
- **The score rewards efficiency, not brute force.** Per level:

  ```
  score = min(115, (baseline_actions / your_actions)^2 * 100)
  ```

  so finishing in half as many actions quadruples the score. Flailing around
  for thousands of moves scores close to zero.
- **Constraints:** the notebook must run in under 9 hours, and **internet is
  disabled** during the official run. Any model the agent uses has to run
  locally inside the notebook.
- Top-scoring entrants win prizes, and are required to open source their
  solutions.

## What is in this repository

| Path | What it is |
|---|---|
| `my_agent.py` | The agent: explores each level by least-visited (state, action), clicks on real objects, ignores moves that change nothing. |
| `arc3_baseline.ipynb` | The notebook submitted to Kaggle. |
| `kaggle_submit/` | The same kernel wrapped for the Kaggle CLI. |
| `gptoss_my_agent_cell.py` | A second, LLM-based agent that runs a local `gpt-oss-120b` model under vLLM. |
| `arc3_gptoss_improved.ipynb` | The assembled notebook for that LLM agent. |
| `HOWTO_gptoss.md` | Step-by-step guide for running the LLM agent on Kaggle. |
| `tools/local_gateway.py` | Serves the public games locally so ideas can be tested fully offline. |
| `agent_eval_results.txt` | Benchmark numbers from those local runs. |

## Result

The heuristic agent scored **0.16** on the public leaderboard (the official
random sample scores 0.18, and leading entries are around 67). It works, but it
is not competitive — which is exactly why the repository also contains the LLM
track.

## Credits and license

This project builds on work by others, used under their own licenses:
[`ARC-AGI-3-Agents`](https://github.com/arcprize/ARC-AGI-3-Agents) (ARC Prize,
MIT), the official Kaggle sample submission (Dries Smit / Tufa Labs,
Apache-2.0), and the official GPT-OSS Kaggle template (Greg Kamradt,
Apache-2.0).

Our own code is released under **CC-BY-4.0** — see [`LICENSE`](LICENSE).

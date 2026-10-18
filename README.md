# ARC-AGI-3 — Heuristic Explorer Agent

A small, dependency-light agent for the
[ARC Prize 2026 — ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3)
Kaggle competition, plus the tooling used to build and submit it.

The agent plays the interactive ARC-AGI-3 environments with **no LLM and no
internet access**: it explores each level with a least-visited
`(screen-state, action)` policy, clicks real objects instead of random pixels,
and suppresses no-op moves.

## Kaggle result

| Agent | Public leaderboard score |
|---|---|
| Leading entries | ~67 |
| Official sample — "Stochastic Goose" | 0.25 |
| Official sample — "Random Agent" | 0.18 |
| **This agent** | **0.16** |

Kaggle score is a 0–100 scale, and the metric rewards *efficiency*:

```
score per level = min(115, (baseline_actions / actions_taken)^2 * 100)
```

...aggregated with weights by level index. So a strategy that brute-forces its
way through levels is punished: fast, deliberate play dominates.

This baseline did not beat the official random sample. The lesson (and the
reason the repo now also contains the LLM track below) is that an action-capped
heuristic is not enough — you need real game understanding.

## How it decides actions

1. **Least-visited search** — pick the `(coarse screen, action)` pair tried the
   fewest times, so coverage is even instead of luck-dependent.
2. **Object-aware clicking** — `ACTION6` targets cells containing non-background
   pixels (rarest colour first) rather than uniform-random coordinates.
3. **No-op suppression** — an action that left the screen identical is penalised
   so it is retried less often.
4. **Epsilon-greedy escape hatch** (ε = 0.2) — a small share of moves stays
   random so the policy cannot lock into a deterministic dead end.

Knobs: `MYAGENT_ACTION_CAP`, `MYAGENT_SEED`, `MYAGENT_MAX_CLICKS`.

## LLM agent (GPT-OSS-120B)

The prize-competitive track is an LLM agent running a local `gpt-oss-120b`
under vLLM on Kaggle's RTX Pro 6000 (internet off, weights attached as Kaggle
inputs). It answers one guided-JSON action per turn.

Our version differs from the stock template in four ways:

| Aspect | Stock template | This repo |
|---|---|---|
| Frame rendering | Python list literals | indexed **hex rows** (~12.4k → ~4.3k chars, ≈3× cheaper) |
| Action memory | none | last 8 actions + screen-diff (`ACTION3 -> grid 0: NO CHANGE`) |
| `ACTION_MAX_TOKENS` | 8192 | 4096 — the JSON answer is tiny |
| `MAX_ACTIONS` | 50 | 60 |

See [`HOWTO_gptoss.md`](HOWTO_gptoss.md) for the full walkthrough,
[`gptoss_my_agent_cell.py`](gptoss_my_agent_cell.py) for the agent source, and
[`arc3_gptoss_improved.ipynb`](arc3_gptoss_improved.ipynb) for the assembled
notebook.

## Repository layout

| Path | Purpose |
|---|---|
| `my_agent.py` | The heuristic agent (`MyAgent`). The strategy file. |
| `build_notebook.py` | Injects `my_agent.py` into the starter notebook. |
| `arc3_baseline.ipynb` | Generated Kaggle submission notebook (heuristic track). |
| `kaggle_submit/` | Ready-to-push kernel folder (`submission.ipynb` + `kernel-metadata.json`). |
| `gptoss_my_agent_cell.py` | Improved LLM agent (notebook cell body). |
| `build_gptoss_notebook.py` | Assembles the LLM notebook from the official template. |
| `arc3_gptoss_improved.ipynb` | Generated Kaggle submission notebook (LLM track). |
| `fetch_template.py` | Re-downloads the official GPT-OSS template from the Kaggle API. |
| `test_gptoss_agent.py` | Offline unit checks for the LLM agent's prompt/parse helpers. |
| `tools/local_gateway.py` | Offline stand-in for the competition gateway. |
| `agent_eval_results.txt` | Local benchmark numbers vs. the random baseline. |

## Local evaluation

The game engine used by Kaggle is the same one published on PyPI, so the agent
can be tested fully offline (no API key, no internet).

```bash
# 1. Clone the official framework next to this repo
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git

# 2. Apply this agent
cp my_agent.py ARC-AGI-3-Agents/agents/templates/my_agent.py
#    ... and register it in ARC-AGI-3-Agents/agents/__init__.py as "myagent"

# 3. Serve the public games locally, then play them all
cd ARC-AGI-3-Agents
python ../tools/local_gateway.py          # terminal 1
MPLBACKEND=agg python main.py --agent myagent   # terminal 2
```

Add `--game ls20` to run a single game, and set `MYAGENT_ACTION_CAP` to keep a
test run short.

Rough numbers from a 25-game, 500-action local sweep
(`agent_eval_results.txt`): this agent reaches a median score around **0.10**
and typically completes 2–7 levels, versus the capped random baseline at
**~0.03** with 1 level.

## Notes

- Runs fully offline: `OPERATION_MODE=offline` locally, the competition gateway
  during the Kaggle rerun. Internet must be off on the Kaggle notebook.
- Competition Mode rules are respected: one scorecard, one `make` per
  environment, resets only.

## Attribution

This repo contains and builds on third-party work. Their licenses apply to
their respective files:

- [`ARC-AGI-3-Agents`](https://github.com/arcprize/ARC-AGI-3-Agents) — the agent
  framework, © ARC Prize, MIT. Cloned separately (not vendored here).
- `arc3-sample-submission-stochastic-goose.ipynb` — the official Kaggle sample
  submission, © Dries Smit / Tufa Labs, Apache-2.0. Used as the starting point
  for `arc3_baseline.ipynb`.
- `template_gptoss.ipynb` — the official "ARC-AGI-3: GPT-OSS-120B" template,
  © Greg Kamradt, Apache-2.0. Used as the starting point for
  `arc3_gptoss_improved.ipynb`.

Everything else — the agent logic, notebook builders, offline gateway wrapper
and tests — is this project's own work.

## License

Original code in this repository is released under **CC-BY-4.0** — see
[`LICENSE`](LICENSE).

"""Exploratory agent for ARC-AGI-3.

Improves on the random baseline with cheap, game-agnostic ideas:

1. Least-visited search: actions are scored by how often the current
   (coarse screen state, action) pair has already been tried, and the least
   visited one wins. Coverage is therefore even instead of luck-dependent.
2. Object-aware clicking: ACTION6 targets the cells that actually contain
   non-background pixels rather than uniform-random coordinates.
3. No-op suppression: an action that left the screen unchanged is penalised,
   so the agent stops burning moves on dead ends.
4. Epsilon-greedy escape hatch: a small fraction of moves stays random so the
   policy cannot lock into a deterministic dead end.

All limits can be overridden with env vars (MYAGENT_*), which keeps the
defaults identical to what the competition notebook runs.
"""

import hashlib
import json
import logging
import os
import random
import time
from typing import Any

import numpy as np

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


CONFIG = {
    "seed": _env_int("MYAGENT_SEED", 42),
    "per_game_action_cap": _env_int("MYAGENT_ACTION_CAP", 500),
    "global_time_budget_s": 7.5 * 3600,  # notebook limit is 9 h; stop early to fail safe
    "dry_run": False,  # True = tiny limits so a test submission finishes in minutes
    "verbose": True,
}

MAX_CLICK_TARGETS = _env_int("MYAGENT_MAX_CLICKS", 24)
CLICK_STEP = 4  # display pixels per coarse cell (sprites are at least a few px wide)
NOOP_PENALTY = 4  # extra visit weight for an action that changed nothing
EXPLORE_EPSILON = 0.2  # fraction of moves that ignore the visits table
VISITS_CAP = 20000  # bound memory across many parallel games

ACTION_CAP = 40 if CONFIG["dry_run"] else CONFIG["per_game_action_cap"]
TIME_BUDGET_S = 300.0 if CONFIG["dry_run"] else CONFIG["global_time_budget_s"]

# Swarm starts one thread per game inside this process, so this start time is
# shared by every game and acts as a global deadline.
GLOBAL_START = time.time()

logger = logging.getLogger("myagent")

_SIMPLE_IDS = [a.value for a in GameAction if a is not GameAction.RESET and a.is_simple()]


def log_outcome(game_id: str, outcome: str, actions: int, levels: int, elapsed_s: float) -> None:
    """One summary line per game with the numbers that drive scoring."""
    logger.info(
        "game=%s outcome=%s actions=%d levels=%d elapsed=%.0fs",
        game_id, outcome, actions, levels, elapsed_s,
    )


class MyAgent(Agent):
    """Explore each level evenly: least-visited (state, action), smart clicks."""

    MAX_ACTIONS = ACTION_CAP  # base class stops the loop at this count
    KEEP_FRAMES = 3  # unbounded frame history would exhaust RAM over 100+ parallel games

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.logged_level = -1
        # Per-game RNG: games run in parallel threads, so a shared global seed
        # would be raced between games and stop being deterministic. Seeding by
        # game_id keeps each game's escape hatch independent but reproducible.
        self.rng = random.Random(f"{CONFIG['seed']}|{self.game_id}")
        # (coarse_state, action_signature) -> visit weight.
        self.visits: dict[tuple[Any, ...], int] = {}
        # Signature of the action that produced the current screen, so the next
        # iteration can tell whether it was a no-op.
        self.pending_sig: tuple[Any, ...] | None = None
        self.pending_state: tuple[Any, ...] | None = None
        self.last_frame_sig: bytes | None = None
        logger.info("game=%s start", self.game_id)

    def append_frame(self, frame: FrameData) -> None:
        """Sliding window over frames: keep only the latest few."""
        self.frames.append(frame)
        if len(self.frames) > self.KEEP_FRAMES:
            self.frames = self.frames[-self.KEEP_FRAMES:]
        if frame.guid:
            self.guid = frame.guid
        if hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(json.loads(frame.model_dump_json()))

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        elapsed_s = time.time() - self.timer
        if latest_frame.state is GameState.WIN:
            log_outcome(self.game_id, "WIN", self.action_counter, latest_frame.levels_completed, elapsed_s)
            return True
        if time.time() - GLOBAL_START > TIME_BUDGET_S:
            log_outcome(self.game_id, "time_budget", self.action_counter, latest_frame.levels_completed, elapsed_s)
            return True
        if self.action_counter >= self.MAX_ACTIONS:
            log_outcome(self.game_id, "action_cap", self.action_counter, latest_frame.levels_completed, elapsed_s)
            return True
        return False

    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        if CONFIG["verbose"] and latest_frame.levels_completed != self.logged_level:
            self.logged_level = latest_frame.levels_completed
            logger.info(
                "game=%s level=%d actions=%d elapsed=%.0fs",
                self.game_id, latest_frame.levels_completed,
                self.action_counter, time.time() - self.timer,
            )

        # Game not started yet, or the previous attempt failed: restart.
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._forget_pending()
            return self._reset_action(latest_frame)

        grid = self._frame_grid(latest_frame)
        self._credit_previous(grid)
        state = self._state_key(grid, latest_frame.levels_completed)
        candidates = self._candidates(latest_frame, grid)

        best_sig, best_data, best_count = None, None, None
        for sig, data in candidates:
            count = self.visits.get((state, sig), 0)
            if best_count is None or count < best_count:
                best_sig, best_data, best_count = sig, data, count

        if self.rng.random() < EXPLORE_EPSILON:
            # Escape hatch: occasionally ignore the visits table entirely.
            chosen = self._random_choice(latest_frame, candidates)
        else:
            chosen = (best_sig, best_data)

        if chosen[0] is None:  # no usable action reported: fall back to a simple move
            chosen = ((_SIMPLE_IDS[0],), None)

        best_sig, best_data = chosen
        self._bump(state, best_sig)
        self.pending_sig = best_sig
        self.pending_state = state
        self.last_frame_sig = self._frame_sig(grid)

        action = GameAction.from_id(best_sig[0])
        if action.is_complex() and best_data:
            action.set_data(best_data)
        action.reasoning = f"explore {action.name}{best_data or ''} visits={best_count}"
        return action

    def _random_choice(
        self,
        latest_frame: FrameData,
        candidates: list[tuple[tuple[Any, ...], dict[str, int] | None]],
    ) -> tuple[tuple[Any, ...] | None, dict[str, int] | None]:
        """Pick uniformly: a listed candidate, else a raw random click."""
        if candidates and self.rng.random() < 0.7:
            return self.rng.choice(candidates)
        if GameAction.ACTION6.value in latest_frame.available_actions:
            x, y = self.rng.randrange(64), self.rng.randrange(64)
            return ((GameAction.ACTION6.value, x, y), {"x": x, "y": y})
        if candidates:
            return self.rng.choice(candidates)
        return (None, None)

    # ------------------------------------------------------------------ helpers

    def _reset_action(self, latest_frame: FrameData) -> GameAction:
        action = GameAction.RESET
        action.reasoning = f"reset from {latest_frame.state.name}"
        return action

    def _forget_pending(self) -> None:
        self.pending_sig = None
        self.pending_state = None
        self.last_frame_sig = None

    def _credit_previous(self, grid: np.ndarray | None) -> None:
        """Penalise the previous action if it produced an identical screen."""
        if self.pending_sig is None or self.pending_state is None or grid is None:
            return
        if self._frame_sig(grid) == self.last_frame_sig:
            key = (self.pending_state, self.pending_sig)
            self.visits[key] = self.visits.get(key, 0) + NOOP_PENALTY

    def _bump(self, state: tuple[Any, ...], sig: tuple[Any, ...]) -> None:
        if len(self.visits) >= VISITS_CAP:
            self.visits.clear()
        key = (state, sig)
        self.visits[key] = self.visits.get(key, 0) + 1

    def _frame_grid(self, latest_frame: FrameData) -> np.ndarray | None:
        """Last 2D colour grid of the observation, or None when absent."""
        frames = latest_frame.frame
        if not frames:
            return None
        try:
            return np.asarray(frames[-1], dtype=np.int16)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _frame_sig(grid: np.ndarray | None) -> bytes | None:
        if grid is None:
            return None
        return hashlib.blake2b(grid.tobytes(), digest_size=16).digest()

    @staticmethod
    def _state_key(grid: np.ndarray | None, levels: int) -> tuple[Any, ...]:
        """Coarse screen state so visit counts generalise across small changes."""
        if grid is None:
            return ("empty", levels)
        coarse = grid[::4, ::4].tobytes()
        return (coarse, levels)

    def _candidates(
        self, latest_frame: FrameData, grid: np.ndarray | None
    ) -> list[tuple[tuple[Any, ...], dict[str, int] | None]]:
        ids = [i for i in latest_frame.available_actions if i != GameAction.RESET.value]
        if not ids:
            ids = list(_SIMPLE_IDS)

        out: list[tuple[tuple[Any, ...], dict[str, int] | None]] = []
        for i in sorted(ids):
            if i == GameAction.ACTION6.value:
                continue
            out.append(((i,), None))

        if GameAction.ACTION6.value in ids:
            for x, y in self._click_targets(grid):
                out.append(((GameAction.ACTION6.value, x, y), {"x": x, "y": y}))
        return out

    def _click_targets(self, grid: np.ndarray | None) -> list[tuple[int, int]]:
        """Coarse cells that contain non-background pixels, rarest colour first."""
        if grid is None:
            step = CLICK_STEP
            return [(x, y) for y in range(step, 64, step * 2) for x in range(step, 64, step * 2)]

        values, counts = np.unique(grid, return_counts=True)
        background = int(values[counts.argmax()])
        rarity = {int(v): int(c) for v, c in zip(values, counts)}

        scored: list[tuple[int, int, tuple[int, int]]] = []
        for cy in range(0, 64, CLICK_STEP):
            for cx in range(0, 64, CLICK_STEP):
                cell = grid[cy:cy + CLICK_STEP, cx:cx + CLICK_STEP]
                ys, xs = np.nonzero(cell != background)
                if ys.size == 0:
                    continue
                colours = cell[ys, xs]
                rarest = min(rarity.get(int(c), 0) for c in colours)
                # Click the centre of mass of the interesting pixels in the cell.
                px = cx + int(round(xs.mean()))
                py = cy + int(round(ys.mean()))
                scored.append((rarest, px, py))

        scored.sort()
        return [(px, py) for _, px, py in scored[:MAX_CLICK_TARGETS]]

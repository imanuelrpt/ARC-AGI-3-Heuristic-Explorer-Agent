%%writefile /kaggle/working/my_agent.py

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import openai
from openai import OpenAI as OpenAIClient

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from PIL import ImageDraw

if not hasattr(ImageDraw, "Coords"):
    ImageDraw.Coords = Any

try:
    from .llm_agents import LLM
except ImportError:
    from agents.templates.llm_agents import LLM

from arcengine import FrameData, GameAction

# Agent Version for Local Experiment Versioning
MY_AGENT_VERSION = "gpt-oss-hex-memory-v3"

# Model Input
# -----------
AGENT_MODEL_NAME = "gpt-oss-120b"
MODEL_ID = "/kaggle/input/models/danielhanchen/gpt-oss-120b/transformers/default/1"
VLLM_MODEL_PATH_ENV = "VLLM_MODEL_PATH"

# Server Startup Configuration
# ----------------------------
VLLM_HOST = "127.0.0.1"
VLLM_PORT = 8000
OPENAI_BASE_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/v1"
HEALTH_URL = f"http://{VLLM_HOST}:{VLLM_PORT}/health"
MODELS_URL = f"{OPENAI_BASE_URL}/models"
OPENAI_API_KEY = "EMPTY"  # Dummy key
VLLM_STARTUP_TIMEOUT_SEC = 1800
VLLM_POLL_INTERVAL_SEC = 5.0

# vLLM Engine Parameters
# ----------------------
MAX_NUM_SEQS = 12
MAX_MODEL_LEN = 64000
KV_CACHE_DTYPE = "fp8"
TENSOR_PARALLEL_SIZE = 1
ENFORCE_EAGER = True

_HEX = "0123456789abcdef"

vllm_process: subprocess.Popen | None = None
TIKTOKEN_ENCODINGS_DIR = None


def start_vllm_server(timeout_s: int = VLLM_STARTUP_TIMEOUT_SEC) -> subprocess.Popen:
    """Start or reuse the local vLLM server and block until it is healthy."""
    global vllm_process

    configure_openai_env(include_tiktoken=True)

    if _server_process_is_ready(vllm_process):
        print(f"vLLM is already running at {OPENAI_BASE_URL}")
        return vllm_process  # type: ignore[return-value]

    if vllm_process is not None and vllm_process.poll() is None:
        print(f"Existing vLLM process pid={vllm_process.pid} is not ready; restarting.")
        stop_vllm_server()

    cmd = build_vllm_command()
    print("\n--- starting vLLM ---")
    print(" ".join(cmd))

    vllm_process = subprocess.Popen(
        cmd,
        stdout=None,
        stderr=subprocess.STDOUT,
        env=build_vllm_env(),
        text=True,
    )
    print(f"Started vLLM pid={vllm_process.pid}")

    wait_for_vllm_server(vllm_process, timeout_s=timeout_s)
    return vllm_process


def stop_vllm_server(timeout_s: int = 30) -> None:
    """Terminate the managed vLLM subprocess if this agent started one."""
    global vllm_process

    if vllm_process is None or vllm_process.poll() is not None:
        vllm_process = None
        return

    print(f"Stopping vLLM pid={vllm_process.pid}")
    vllm_process.terminate()
    try:
        vllm_process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"vLLM pid={vllm_process.pid} did not stop cleanly; killing.")
        vllm_process.kill()
        vllm_process.wait()
    vllm_process = None


def configure_openai_env(include_tiktoken: bool = False) -> None:
    """Point OpenAI-compatible clients and tiktoken at the local Kaggle assets."""
    os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    if include_tiktoken:
        os.environ["TIKTOKEN_ENCODINGS_BASE"] = str(get_tiktoken_encodings_dir())


def build_vllm_command(
    model_id: str | None = None,
    served_model_name: str = AGENT_MODEL_NAME,
) -> list[str]:
    """Build the vLLM server command used by the notebook and agent startup."""
    model_id = model_id or os.getenv(VLLM_MODEL_PATH_ENV) or MODEL_ID
    args = [
        "--served-model-name",
        served_model_name,
        "--host",
        VLLM_HOST,
        "--port",
        str(VLLM_PORT),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "openai",
        "--max-num-seqs",
        str(MAX_NUM_SEQS),
        "--max-model-len",
        str(MAX_MODEL_LEN),
        "--kv-cache-dtype",
        KV_CACHE_DTYPE,
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
    ]
    if ENFORCE_EAGER:
        args.append("--enforce-eager")

    vllm_exe = shutil.which("vllm")
    if vllm_exe:
        return [vllm_exe, "serve", model_id, *args]

    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_id,
        *args,
    ]


def build_vllm_env() -> dict[str, str]:
    """Create the environment passed to the vLLM subprocess."""
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = OPENAI_API_KEY
    env["OPENAI_BASE_URL"] = OPENAI_BASE_URL
    env["TIKTOKEN_ENCODINGS_BASE"] = str(get_tiktoken_encodings_dir())
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    return env


def wait_for_vllm_server(
    process: subprocess.Popen,
    timeout_s: int = VLLM_STARTUP_TIMEOUT_SEC,
    poll_interval_s: float = VLLM_POLL_INTERVAL_SEC,
) -> None:
    """Wait until vLLM responds to health/model checks or the process exits."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"vLLM exited with code {process.returncode}. "
                "Check the output above for the vLLM error log."
            )
        if server_ready(timeout=2.0):
            print(f"vLLM is ready at {OPENAI_BASE_URL}")
            return
        time.sleep(poll_interval_s)

    raise TimeoutError(
        f"Timed out waiting for vLLM after {timeout_s}s. "
        "Check the output above for the vLLM startup log."
    )


def server_ready(timeout: float = 2.0) -> bool:
    """Return True once either vLLM health or model listing succeeds."""
    return _url_ok(HEALTH_URL, timeout=timeout) or _url_ok(MODELS_URL, timeout=timeout)


def get_tiktoken_encodings_dir() -> Path:
    """Locate cached tiktoken encoding files bundled as Kaggle inputs."""
    global TIKTOKEN_ENCODINGS_DIR
    if TIKTOKEN_ENCODINGS_DIR is not None:
        return TIKTOKEN_ENCODINGS_DIR

    for root in [Path("/kaggle/input"), Path("/kaggle/working")]:
        if not root.exists():
            continue
        for candidate_dir in root.rglob("*"):
            if not candidate_dir.is_dir():
                continue
            if (
                (candidate_dir / "cl100k_base.tiktoken").exists()
                and (candidate_dir / "o200k_base.tiktoken").exists()
            ):
                TIKTOKEN_ENCODINGS_DIR = candidate_dir
                print("TIKTOKEN_ENCODINGS_DIR =", candidate_dir)
                return candidate_dir

    raise FileNotFoundError(
        "Could not find a directory containing both "
        "cl100k_base.tiktoken and o200k_base.tiktoken."
    )


def _server_process_is_ready(process: subprocess.Popen | None) -> bool:
    return process is not None and process.poll() is None and server_ready(timeout=2.0)


def _url_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (OSError, URLError):
        return False


# ----------------- Agent Definition ------------------------------


class MyAgent(LLM):
    """ARC LLM agent backed by a local vLLM gpt-oss-120b server.

    Differences from the stock template (v3):
      * the frame is rendered as compact hex rows instead of Python list
        literals, which cuts the prompt from ~12.4k to ~4.3k characters
        (about 3x).  Cheaper prefill means more actions fit in the game
        clock, and the layout stays positionally readable.
      * a short action memory ("ACTION3 -> grid 0: NO CHANGE") is fed back
        to the model so it can avoid loops and repeated no-op moves.
      * a screen-diff summary tells the model exactly what its last action
        changed, and where.
    """

    # Default parameters, no need to change
    MODEL = AGENT_MODEL_NAME
    MODEL_REQUIRES_TOOLS = False

    DO_OBSERVATION = False
    OBSERVATION_MAX_TOKENS = 512

    # Maximum number of actions to take per game.
    MAX_ACTIONS = 60
    # Maximum previous messages to keep in context for the current game state.
    MESSAGE_LIMIT = 2
    # How often to retry structured action selection if the model fails to respond
    # in the specified format.
    ACTION_RETRY_LIMIT = 2
    # Maximum tokens per action selection. Lowered from 8192: the JSON answer is
    # tiny and every extra reasoning token is dead time inside the game clock.
    ACTION_MAX_TOKENS = 4096
    # How long to wait for the vllm-server to respond per completion.
    LLM_REQUEST_TIMEOUT_S = 120

    # How many past (action -> outcome) lines to show the model.
    ACTION_LOG_LIMIT = 8

    # Time-limit per ARC-AGI game in seconds.
    GAME_TIME_LIMIT_S = 60 * 20
    # Time-limit by which the agent has to return its first action.
    FIRST_ACTION_DEADLINE_S = 14 * 60

    ACTION_NAMES = [
        GameAction.RESET.name,
        GameAction.ACTION1.name,
        GameAction.ACTION2.name,
        GameAction.ACTION3.name,
        GameAction.ACTION4.name,
        GameAction.ACTION5.name,
        GameAction.ACTION6.name,
        GameAction.ACTION7.name,
    ]
    DEFAULT_ACTION_NAMES = ACTION_NAMES[1:]
    AGENT_VERSION = MY_AGENT_VERSION

    _vllm_started = False
    NOTEBOOK_START_TIME_S = 0
    NOTEBOOK_START_TIME_PATH = Path("/kaggle/working/notebook_start_time_s.txt")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize cheaply so the first RESET action is never blocked by vLLM."""
        print(f"MyAgent version: {self.AGENT_VERSION}")
        configure_openai_env()
        super().__init__(*args, **kwargs)
        self._deadline_hit = False
        self._game_started_monotonic = 0.0
        self._prev_grids: list[list[list[int]]] | None = None
        self._last_action_name: str | None = None
        self._action_log: list[str] = []

    # ------------------ time bookkeeping ------------------

    @property
    def startup_elapsed_s(self) -> float:
        start_time_s = float(self.NOTEBOOK_START_TIME_S)
        if start_time_s <= 0 and self.NOTEBOOK_START_TIME_PATH.exists():
            try:
                start_time_s = float(self.NOTEBOOK_START_TIME_PATH.read_text().strip())
            except (OSError, ValueError):
                start_time_s = 0.0
        if start_time_s <= 0:
            return self.game_elapsed_s
        return max(0.0, time.time() - start_time_s)

    @property
    def game_elapsed_s(self) -> float:
        if self._game_started_monotonic:
            return time.monotonic() - self._game_started_monotonic
        if self.timer:
            return time.time() - self.timer
        return 0.0

    @property
    def game_time_remaining_s(self) -> float | None:
        limit = max(0, int(self.GAME_TIME_LIMIT_S))
        if limit <= 0:
            return None
        return max(0.0, limit - self.game_elapsed_s)

    def _time_limit_expired(self) -> bool:
        remaining = self.game_time_remaining_s
        return remaining is not None and remaining <= 0

    def _mark_deadline_hit(self, reason: str) -> None:
        if not self._deadline_hit:
            logging.info(
                "%s; stopping %s after %s actions and %.2f seconds",
                reason,
                self.game_id,
                self.action_counter,
                self.game_elapsed_s,
            )
        self._deadline_hit = True

    def _ensure_vllm_ready(self) -> bool:
        if self.__class__._vllm_started:
            configure_openai_env()
            return True

        remaining = self.game_time_remaining_s
        if remaining is not None and remaining <= 0:
            self._mark_deadline_hit("Per-game time limit reached before vLLM startup")
            return False

        timeout_s = int(remaining) if remaining is not None else VLLM_STARTUP_TIMEOUT_SEC
        timeout_s = max(1, timeout_s)
        try:
            start_vllm_server(timeout_s=timeout_s)
        except TimeoutError:
            self._mark_deadline_hit("Per-game time limit reached while waiting for vLLM")
            return False
        except Exception as e:
            logging.exception("vLLM startup failed: %s", e)
            return False

        self.__class__._vllm_started = True
        configure_openai_env()
        return True

    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        """Use the parent game-state check, plus a wall-clock stop before new work."""
        if not self._game_started_monotonic:
            self._game_started_monotonic = time.monotonic()
        if super().is_done(frames, latest_frame):
            return True
        if self._deadline_hit:
            return True
        if self.action_counter == 0:
            return False
        if self._time_limit_expired():
            self._mark_deadline_hit("Per-game time limit reached before next action")
            return True
        return False

    # ------------------ action memory ------------------

    def _observe(self, latest_frame: FrameData) -> None:
        """Record what the previous action actually did to the screen."""
        try:
            current = [
                [list(row) for row in grid]
                for grid in (getattr(latest_frame, "frame", None) or [])
            ]
            if self._prev_grids is None:
                effect = "game start"
            else:
                effect = self._diff_grids(self._prev_grids, current)
            self._prev_grids = current
            self._action_log.append(f"{self._last_action_name or 'RESET'} -> {effect}")
            self._action_log = self._action_log[-self.ACTION_LOG_LIMIT :]
        except Exception:
            logging.exception("Failed to update action memory")

    def _remember(self, action: GameAction) -> None:
        self._last_action_name = action.name

    def _diff_grids(
        self,
        prev: list[list[list[int]]],
        cur: list[list[list[int]]],
    ) -> str:
        """Human-readable summary of what changed between two frames."""
        if not cur:
            return "no frame data"
        if len(prev) != len(cur):
            return f"grid count changed {len(prev)}->{len(cur)}"

        parts: list[str] = []
        for index, (before, after) in enumerate(zip(prev, cur)):
            changed = 0
            min_x = min_y = 1 << 30
            max_x = max_y = -1
            for y in range(min(len(before), len(after))):
                row_a, row_b = before[y], after[y]
                for x in range(min(len(row_a), len(row_b))):
                    if row_a[x] != row_b[x]:
                        changed += 1
                        if x < min_x:
                            min_x = x
                        if x > max_x:
                            max_x = x
                        if y < min_y:
                            min_y = y
                        if y > max_y:
                            max_y = y
            if changed:
                parts.append(
                    f"grid {index}: {changed} cells changed "
                    f"(x {min_x}-{max_x}, y {min_y}-{max_y})"
                )
            else:
                parts.append(f"grid {index}: NO CHANGE")
        return "; ".join(parts)

    # ------------------ prompts ------------------

    def _render_grid_hex(self, grid: list[list[int]]) -> str:
        """Render one 64x64 grid as 64 hex rows, prefixed with the row index."""
        lines = []
        for y, row in enumerate(grid):
            lines.append(f"{y:02d} " + "".join(_HEX[int(v) & 0xF] for v in row))
        return "\n".join(lines)

    def _render_frames_compact(self, latest_frame: FrameData) -> str:
        grids = getattr(latest_frame, "frame", None) or []
        if not grids:
            return "(no grid data)"
        blocks = []
        for index, grid in enumerate(grids):
            blocks.append(f"--- Grid {index} ---")
            blocks.append(self._render_grid_hex(grid))
        return "\n".join(blocks)

    def build_state_prompt(self, latest_frame: FrameData) -> str:
        """Render the current state: score, action memory and the hex frame."""
        log_text = "\n".join(f"- {entry}" for entry in self._action_log) or "- (none yet)"
        available = ", ".join(self._available_action_names(latest_frame))
        return textwrap.dedent(
            f"""
            # RESULT OF PREVIOUS ACTION
            State: {latest_frame.state.name}
            Levels completed (score): {latest_frame.levels_completed}

            # RECENT ACTIONS (oldest first)
            Each line is: action -> what changed on screen.
            "NO CHANGE" means the action had no effect, so repeating it is useless.
            {log_text}

            # CURRENT FRAME
            64x64 grids, cell values 0-15 shown as one hex digit each
            (0-9 = 0-9, a = 10, b = 11, ... f = 15).
            Row 00 is the top row, column 0 is the leftmost character.
            {self._render_frames_compact(latest_frame)}

            # AVAILABLE ACTIONS
            {available}
            """
        ).strip()

    def build_user_prompt(
        self,
        latest_frame: FrameData,
        available_action_names: list[str] | None = None,
    ) -> str:
        """Build the action-selection instruction shown after the current frame."""
        available_action_names = available_action_names or self.DEFAULT_ACTION_NAMES
        return textwrap.dedent(
            f"""
            # CONTEXT
            You are playing an ARC-AGI-3 dynamic game. Your objective is to WIN
            and avoid GAME_OVER while minimizing the number of actions.

            Each action produces one Frame. A Frame contains one or more Grids.
            Each Grid is a matrix with dimensions INT<0,63> by INT<0,63> and
            cell values INT<0,15> (hex encoded above).

            # AVAILABLE ACTIONS THIS TURN
            {', '.join(available_action_names)}
            ACTION6 is a click action and should include integer x and y
            coordinates from 0 to 63.

            # STRATEGY HINTS
            - Compare with the previous frame: an action whose row says
              "NO CHANGE" did nothing at all. Do not send it again.
            - Prefer a small, purposeful experiment over random poking.
            - Try to identify the player, walls, doors, keys and any counter.

            # RESPONSE FORMAT
            Return only one JSON object. Do not include prose, markdown, code
            fences, or commentary.

            Valid examples:
            {{"action":"ACTION1"}}
            {{"action":"ACTION6","x":0,"y":0}}
            """
        ).strip()

    # ------------------ action selection ------------------

    def choose_action(
        self,
        frames: list[FrameData],
        latest_frame: FrameData,
    ) -> GameAction:
        """Choose the next ARC action using guided JSON for simplicity."""
        self._quiet_openai_logging()
        self._observe(latest_frame)
        available_action_names = self._available_action_names(latest_frame)

        if self.action_counter == 0:
            first_action_deadline_s = max(1, int(self.FIRST_ACTION_DEADLINE_S))
            startup_elapsed_s = self.startup_elapsed_s
            if startup_elapsed_s > first_action_deadline_s:
                logging.warning(
                    f"First action for {self.game_id} is being selected after {startup_elapsed_s:.2f} seconds since notebook start, past the {first_action_deadline_s:.2f} second target",
                )
            action = self._initial_reset_action()
            self._remember(action)
            return action

        if self._time_limit_expired():
            self._mark_deadline_hit("Per-game time limit reached before LLM action")
            action = self._fallback_action(available_action_names)
            self._remember(action)
            return action

        if not self._ensure_vllm_ready():
            action = self._fallback_action(available_action_names)
            self._remember(action)
            return action

        try:
            action = self._choose_llm_action(latest_frame, available_action_names)
        except Exception as e:
            logging.exception(
                "Action selection failed; falling back to %s: %s",
                available_action_names[0],
                e,
            )
            action = self._fallback_action(available_action_names)

        self._remember(action)
        return action

    def _fallback_action(self, available_action_names: list[str]) -> GameAction:
        action_name = (
            available_action_names[0] if available_action_names else GameAction.ACTION5.name
        )
        if action_name == GameAction.RESET.name:
            action_name = GameAction.ACTION5.name
        return self._game_action_from_data({"action": action_name})

    def _choose_llm_action(
        self,
        latest_frame: FrameData,
        available_action_names: list[str],
    ) -> GameAction:
        client = OpenAIClient(
            base_url=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            timeout=max(1, int(self.LLM_REQUEST_TIMEOUT_S)),
        )
        self._prepare_action_messages(latest_frame, available_action_names)

        if self.DO_OBSERVATION:
            self._request_observation(client)
            self._append_action_prompt(latest_frame, available_action_names)

        data = self._request_json_action(client, available_action_names)
        data = self._normalize_action_data(data, available_action_names)
        self._record_assistant_action(data)
        return self._game_action_from_data(data)

    def _initial_reset_action(self) -> GameAction:
        data = {"action": GameAction.RESET.name}
        self.messages = [{"role": "assistant", "content": json.dumps(data)}]
        return GameAction.RESET

    def take_action(self, action: GameAction) -> FrameData | None:
        """Submit an action without letting gateway/runtime errors kill the worker thread."""
        try:
            return super().take_action(action)
        except Exception as e:
            logging.exception(
                "Action submission failed for %s; stopping game: %s", action.name, e
            )
            self._mark_deadline_hit("Action submission failed")
            return None

    def _prepare_action_messages(
        self,
        latest_frame: FrameData,
        available_action_names: list[str],
    ) -> None:
        self.messages = [
            {"role": "user", "content": self.build_state_prompt(latest_frame)}
        ]
        self._append_action_prompt(latest_frame, available_action_names)

    def _append_action_prompt(
        self,
        latest_frame: FrameData,
        available_action_names: list[str],
    ) -> None:
        self.messages.append(
            {
                "role": "user",
                "content": self.build_user_prompt(
                    latest_frame,
                    available_action_names=available_action_names,
                ),
            }
        )
        self.messages = self.messages[-self.MESSAGE_LIMIT :]

    def _request_json_action(
        self,
        client: OpenAIClient,
        available_action_names: list[str],
    ) -> dict[str, Any]:
        """Ask vLLM for one JSON action constrained to currently available actions."""
        retry_prompt = self._retry_prompt(available_action_names)
        schema = self._action_json_schema(available_action_names)

        for attempt in range(self.ACTION_RETRY_LIMIT + 1):
            request_messages = (
                self.messages if attempt == 0 else [*self.messages, retry_prompt]
            )
            response = self._create_action_completion(
                client, request_messages, schema, attempt
            )
            if response is None:
                continue

            choices = getattr(response, "choices", None) or []
            if not choices:
                logging.warning(
                    "vLLM JSON action response had no choices on attempt %s/%s",
                    attempt + 1,
                    self.ACTION_RETRY_LIMIT + 1,
                )
                continue

            message = choices[0].message
            data = self._parse_json_response(self._message_content(message))
            if data is not None:
                return data

            logging.warning(
                "JSON action response was invalid on attempt %s/%s; message=%r",
                attempt + 1,
                self.ACTION_RETRY_LIMIT + 1,
                message.model_dump(),
            )

        return {"action": GameAction.ACTION5.name}

    def _create_action_completion(
        self,
        client: OpenAIClient,
        request_messages: list[dict[str, Any]],
        schema: dict[str, Any],
        attempt: int,
    ) -> Any | None:
        """Submit one guided-JSON chat completion and account for token usage."""
        try:
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=request_messages,
                max_tokens=self.ACTION_MAX_TOKENS,
                extra_body={"guided_json": schema},
            )
        except openai.OpenAIError as e:
            logging.warning(
                f"vLLM JSON action request failed on attempt {attempt + 1}/{self.ACTION_RETRY_LIMIT + 1}: {e}",
            )
            return None
        except Exception as e:
            logging.exception(
                "Unexpected vLLM JSON action request failure on attempt %s/%s: %s",
                attempt + 1,
                self.ACTION_RETRY_LIMIT + 1,
                e,
            )
            return None

        usage = getattr(response, "usage", None)
        if usage and getattr(usage, "total_tokens", None):
            self.track_tokens(usage.total_tokens)
        return response

    def _request_observation(self, client: OpenAIClient) -> None:
        """Optionally add a short natural-language observation before action choice."""
        try:
            response = client.chat.completions.create(
                model=self.MODEL,
                messages=self.messages,
                max_tokens=self.OBSERVATION_MAX_TOKENS,
            )
        except openai.OpenAIError as e:
            logging.warning("vLLM observation request failed: %s", e)
            return
        except Exception as e:
            logging.exception("Unexpected vLLM observation request failure: %s", e)
            return

        choices = getattr(response, "choices", None) or []
        if not choices:
            logging.warning("vLLM observation response had no choices")
            return

        content = self._message_content(choices[0].message)
        usage = getattr(response, "usage", None)
        if usage and getattr(usage, "total_tokens", None):
            self.track_tokens(usage.total_tokens, content)
        self.messages.append({"role": "assistant", "content": content})
        self.messages = self.messages[-self.MESSAGE_LIMIT :]

    def _available_action_names(self, latest_frame: FrameData) -> list[str]:
        """Map environment-provided action ids to names, excluding RESET after turn zero."""
        names: list[str] = []
        for action_id in getattr(latest_frame, "available_actions", None) or []:
            try:
                name = GameAction(action_id).name
            except ValueError:
                continue
            if name != GameAction.RESET.name and name in self.ACTION_NAMES:
                names.append(name)
        return names or list(self.DEFAULT_ACTION_NAMES)

    def _action_json_schema(self, available_action_names: list[str]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": available_action_names},
                "x": {"type": "integer", "minimum": 0, "maximum": 63},
                "y": {"type": "integer", "minimum": 0, "maximum": 63},
            },
            "required": ["action"],
            "additionalProperties": False,
        }

    def _retry_prompt(self, available_action_names: list[str]) -> dict[str, str]:
        return {
            "role": "user",
            "content": (
                "Your previous response was not valid action JSON. Return only one JSON object now. "
                f"The action must be one of: {', '.join(available_action_names)}. "
                "Do not include prose, markdown, code fences, or commentary."
            ),
        }

    def _normalize_action_data(
        self,
        data: dict[str, Any],
        available_action_names: list[str],
    ) -> dict[str, Any]:
        """Coerce model JSON into a valid action allowed by the current frame."""
        action_name = str(data.get("action", GameAction.ACTION5.name)).upper()
        if action_name not in available_action_names:
            logging.warning(
                "Invalid or unavailable ARC action %r; defaulting to %s",
                action_name,
                available_action_names[0],
            )
            action_name = available_action_names[0]

        normalized = {"action": action_name}
        if action_name == GameAction.ACTION6.name:
            normalized["x"] = self._clamp_coordinate(data.get("x", 0))
            normalized["y"] = self._clamp_coordinate(data.get("y", 0))
        return normalized

    def _record_assistant_action(self, data: dict[str, Any]) -> None:
        self.messages = [
            *self.messages[-2:],
            {"role": "assistant", "content": json.dumps(data)},
        ]

    def _game_action_from_data(self, data: dict[str, Any]) -> GameAction:
        """Convert normalized JSON into the ARC engine GameAction object."""
        action_name = str(data.get("action", GameAction.ACTION5.name)).upper()
        action = GameAction.from_name(action_name)
        if action_name == GameAction.ACTION6.name:
            action.set_data(
                {
                    "x": self._clamp_coordinate(data.get("x", 0)),
                    "y": self._clamp_coordinate(data.get("y", 0)),
                }
            )
        else:
            action.set_data({})
        return action

    def _message_content(self, message: Any) -> str:
        """Read vLLM output from the Pydantic v2 message payload."""
        data = message.model_dump()
        for key in ["content", "reasoning_content", "reasoning"]:
            value = data.get(key)
            if value:
                return str(value)
        return ""

    def _parse_json_response(self, content: str) -> dict[str, Any] | None:
        """Parse strict JSON first, then recover a single JSON object from text."""
        if not content:
            return None

        data = self._loads_json_object(content)
        if data is not None:
            return data

        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not match:
            return None
        return self._loads_json_object(match.group(0))

    def _loads_json_object(self, content: str) -> dict[str, Any] | None:
        """Load one JSON object, returning None for invalid JSON or non-objects."""
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None

    def _clamp_coordinate(self, value: Any) -> int:
        try:
            coordinate = int(value)
        except (TypeError, ValueError):
            coordinate = 0
        return min(63, max(0, coordinate))

    def _quiet_openai_logging(self) -> None:
        logging.getLogger("openai").setLevel(logging.CRITICAL)
        logging.getLogger("httpx").setLevel(logging.CRITICAL)

    def cleanup(self, *args: Any, **kwargs: Any) -> None:
        """Record concise JSON-mode metadata, then delegate to ARC cleanup."""
        if self._cleanup and hasattr(self, "recorder") and not self.is_playback:
            self.recorder.record(
                {
                    "agent_version": self.AGENT_VERSION,
                    "llm_user_prompt": self.build_user_prompt(
                        self.frames[-1],
                        available_action_names=self.DEFAULT_ACTION_NAMES,
                    ),
                    "llm_response_format": "guided_json_action",
                    "llm_available_actions": self.DEFAULT_ACTION_NAMES,
                    "llm_frame_encoding": "hex_rows_with_row_index",
                }
            )
        super().cleanup(*args, **kwargs)

"""Offline sanity test for gptoss_my_agent_cell.py.

The notebook cell cannot import its real dependencies here (no arcengine /
openai / torch on this machine), so we stub them, exec the cell body and
exercise the pure helper methods that build the prompt.
"""

import sys
import types
from enum import IntEnum

SRC = "gptoss_my_agent_cell.py"


def _install_stubs():
    # --- openai ---
    openai_mod = types.ModuleType("openai")

    class OpenAIError(Exception):
        pass

    class _Completions:
        def create(self, **kwargs):
            raise NotImplementedError

    class _Chat:
        completions = _Completions()

    class OpenAI:
        def __init__(self, *a, **k):
            self.chat = _Chat()

    openai_mod.OpenAIError = OpenAIError
    openai_mod.OpenAI = OpenAI
    sys.modules["openai"] = openai_mod

    # --- PIL / PIL.ImageDraw ---
    pil = types.ModuleType("pil_stub")
    image_draw = types.ModuleType("PIL.ImageDraw")
    image_draw.ImageDraw = object
    pil_image = types.ModuleType("PIL")
    pil_image.ImageDraw = image_draw
    sys.modules["PIL"] = pil_image
    sys.modules["PIL.ImageDraw"] = image_draw

    # --- arcengine ---
    class GameAction(IntEnum):
        RESET = 0
        ACTION1 = 1
        ACTION2 = 2
        ACTION3 = 3
        ACTION4 = 4
        ACTION5 = 5
        ACTION6 = 6
        ACTION7 = 7

        @classmethod
        def from_name(cls, name):
            return cls[name]

        def set_data(self, data):
            self._data = data

    arcengine = types.ModuleType("arcengine")

    class FrameData:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    arcengine.GameAction = GameAction
    arcengine.FrameData = FrameData
    sys.modules["arcengine"] = arcengine

    # --- agents.templates.llm_agents.LLM ---
    agents = types.ModuleType("agents")
    templates = types.ModuleType("agents.templates")
    llm_agents = types.ModuleType("agents.templates.llm_agents")

    class LLM:
        MAX_ACTIONS = 80
        DEFAULT_ACTION_NAMES = []

        def __init__(self):
            self.messages = []
            self.token_counter = 0
            self.frames = []

        def track_tokens(self, *a, **k):
            pass

        def cleanup(self, *a, **k):
            pass

    llm_agents.LLM = LLM
    sys.modules["agents"] = agents
    sys.modules["agents.templates"] = templates
    sys.modules["agents.templates.llm_agents"] = llm_agents

    return GameAction, FrameData


def main():
    with open(SRC, encoding="utf-8") as f:
        body = f.read()

    assert body.startswith("%%writefile"), "cell must start with %%writefile"
    code = "\n".join(body.splitlines()[1:])

    GameAction, _FrameData = _install_stubs()
    namespace = {"__name__": "cellmod"}
    exec(compile(code, SRC, "exec"), namespace)  # noqa: S102

    MyAgent = namespace["MyAgent"]
    agent = object.__new__(MyAgent)
    agent._prev_grids = None
    agent._last_action_name = None
    agent._action_log = []

    # ---- hex rendering ----
    grid = [[0] * 64 for _ in range(64)]
    grid[0][0] = 15
    grid[0][63] = 10
    rendered = agent._render_grid_hex(grid)
    lines = rendered.split("\n")
    assert len(lines) == 64, f"expected 64 rows, got {len(lines)}"
    assert lines[0] == "00 f" + "0" * 62 + "a", lines[0][:20]
    assert lines[1] == "01 " + "0" * 64
    print("hex render: OK")
    print("  row 00 ->", lines[0][:16], "...")

    # ---- diffing ----
    prev = [[[0] * 64 for _ in range(64)]]
    cur = [[[0] * 64 for _ in range(64)]]
    assert agent._diff_grids(prev, cur) == "grid 0: NO CHANGE"

    cur[0][5][7] = 3
    cur[0][5][9] = 4
    cur[0][6][8] = 4
    diff = agent._diff_grids(prev, cur)
    assert diff == "grid 0: 3 cells changed (x 7-9, y 5-6)", diff
    print("diff:", diff)

    # ---- memory loop ----
    class _State:
        name = "PLAYING"

    def frame(grids, score=0, actions=None):
        return types.SimpleNamespace(
            state=_State, levels_completed=score, frame=grids, available_actions=actions
        )

    agent._observe(frame([[ [0]*64 for _ in range(64) ]]))
    assert agent._action_log == ["RESET -> game start"], agent._action_log

    agent._remember(GameAction.ACTION3)
    changed = [[[0] * 64 for _ in range(64)]]
    changed[0][10][20] = 1
    agent._observe(frame(changed))
    assert agent._action_log[-1].startswith("ACTION3 -> grid 0: 1 cells changed"), agent._action_log
    print("memory:", agent._action_log)

    # ---- available actions / schema ----
    names = agent._available_action_names(frame([[[]]], actions=[1, 2, 3, 4, 5, 6, 0]))
    assert names == ["ACTION1", "ACTION2", "ACTION3", "ACTION4", "ACTION5", "ACTION6"], names
    schema = agent._action_json_schema(names)
    assert schema["properties"]["action"]["enum"] == names
    assert schema["required"] == ["action"]
    print("available actions:", names)

    # ---- json parsing ----
    assert agent._parse_json_response('{"action":"ACTION1"}') == {"action": "ACTION1"}
    assert agent._parse_json_response('noise {"action":"ACTION6","x":70,"y":-5} tail') == {
        "action": "ACTION6",
        "x": 70,
        "y": -5,
    }
    assert agent._parse_json_response("not json at all") is None
    assert agent._clamp_coordinate(70) == 63
    assert agent._clamp_coordinate(-5) == 0
    assert agent._clamp_coordinate("abc") == 0
    print("json parsing: OK")

    # ---- normalization rejects unavailable actions ----
    norm = agent._normalize_action_data({"action": "RESET"}, ["ACTION1", "ACTION2"])
    assert norm == {"action": "ACTION1"}, norm
    norm = agent._normalize_action_data({"action": "ACTION6", "x": "12", "y": 99}, ["ACTION6"])
    assert norm == {"action": "ACTION6", "x": 12, "y": 63}, norm
    print("normalization: OK")

    # ---- state prompt builds ----
    prompt = agent.build_state_prompt(frame(changed, score=1, actions=[1, 2, 6]))
    assert "RECENT ACTIONS" in prompt and "ACTION3 ->" in prompt
    assert "hex" in prompt.lower() or "hex" in prompt
    assert "NO CHANGE" in prompt
    user = agent.build_user_prompt(frame(changed), ["ACTION1", "ACTION6"])
    assert "ACTION6" in user and "JSON" in user
    print("prompts: OK (%d chars state, %d chars user)" % (len(prompt), len(user)))

    # ---- size comparison: hex vs list-literal ----
    big = str(grid)
    print("prompt size: hex=%d chars vs list-literal=%d chars (%.1fx smaller)"
          % (len(rendered), len(big), len(big) / len(rendered)))

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()

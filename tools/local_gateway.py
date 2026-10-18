"""Local stand-in for the competition gateway.

Serves GET /api/games from the public environment_files so main.py can run
fully offline (no internet, no API key).

Usage (from the cloned ARC-AGI-3-Agents directory, which owns .env):

    cd ARC-AGI-3-Agents
    python ../tools/local_gateway.py

Then, in a second shell:

    cd ARC-AGI-3-Agents
    MPLBACKEND=agg python main.py --agent myagent
"""

from arc_agi import Arcade, OperationMode

arcade = Arcade(operation_mode=OperationMode.OFFLINE)
print(f"gateway: {len(arcade.available_environments)} local games", flush=True)
arcade.listen_and_serve(host="127.0.0.1", port=8001)

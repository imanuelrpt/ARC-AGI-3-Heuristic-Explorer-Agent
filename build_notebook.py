"""Build arc3_baseline.ipynb from the starter notebook + my_agent.py.

Only the %%writefile cell is replaced; install/runner/dummy cells stay verbatim
from the official starter so they keep matching the competition harness. A
license markdown cell is (re)inserted at the top so the published notebook is
self-documenting for the ARC Prize open-source requirement.
"""

import json

LICENSE_CELL = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# ARC Prize 2026 — ARC-AGI-3 agent\n",
        "\n",
        "Author: Imanuel Roderich\n",
        "\n",
        "Licensed under **CC-BY-4.0** (see `LICENSE`). Open-sourced to satisfy the\n",
        "ARC Prize 2026 eligibility rules.\n",
    ],
}

with open("arc3-sample-submission-stochastic-goose.ipynb", encoding="utf-8") as f:
    nb = json.load(f)

with open("my_agent.py", encoding="utf-8") as f:
    agent_lines = f.read().splitlines(keepends=True)

assert "".join(nb["cells"][1]["source"]).startswith("%%writefile"), "unexpected starter layout"
nb["cells"][1]["source"] = ["%%writefile /kaggle/working/my_agent.py\n"] + agent_lines

# Drop a previously inserted license cell so rebuilds stay idempotent.
nb["cells"] = [
    c
    for c in nb["cells"]
    if not (
        c["cell_type"] == "markdown"
        and "".join(c["source"]).startswith("# ARC Prize 2026")
    )
]
nb["cells"].insert(0, LICENSE_CELL)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

with open("arc3_baseline.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("wrote arc3_baseline.ipynb with", len(nb["cells"]), "cells")

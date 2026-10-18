"""Build arc3_gptoss_improved.ipynb from the official GPT-OSS template.

Replaces the template's `%%writefile .../my_agent.py` cell with our improved
agent (gptoss_my_agent_cell.py).  Everything else (vLLM bootstrap, framework
copy, .env, runner) is left byte-identical to the ARC Prize template so the
Kaggle plumbing keeps working.
"""

import json

TEMPLATE = "template_gptoss.ipynb"
AGENT_CELL = "gptoss_my_agent_cell.py"
OUT = "arc3_gptoss_improved.ipynb"


def main() -> None:
    with open(TEMPLATE, encoding="utf-8") as f:
        nb = json.load(f)
    with open(AGENT_CELL, encoding="utf-8") as f:
        agent_source = f.read().splitlines(keepends=True)

    replaced = 0
    for cell in nb["cells"]:
        source = "".join(cell.get("source", []))
        if source.startswith("%%writefile /kaggle/working/my_agent.py"):
            cell["source"] = agent_source
            replaced += 1

    assert replaced == 1, f"expected exactly one agent cell, replaced {replaced}"

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            cell["outputs"] = []
            cell["execution_count"] = None

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"wrote {OUT} with {len(nb['cells'])} cells (agent cell replaced)")


if __name__ == "__main__":
    main()

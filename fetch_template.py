import json, urllib.request

url = "https://www.kaggle.com/api/v1/kernels/pull/gregkamradt/arc-agi-3-gpt-oss-120b"
with urllib.request.urlopen(url, timeout=60) as r:
    data = json.load(r)

meta = data.get("metadata", {})
src = data["blob"]["source"]
nb = json.loads(src) if isinstance(src, str) else src

print("TITLE:", meta.get("title"))
print("datasets:", meta.get("kernelDataSources"))
print("models:", meta.get("modelDataSources"))
print("cells:", len(nb["cells"]))
print()
for i, c in enumerate(nb["cells"]):
    print("=" * 30, "CELL", i, c["cell_type"], "=" * 30)
    print("".join(c["source"]))
    print()

with open("template_gptoss.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

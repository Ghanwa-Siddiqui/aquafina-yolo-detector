"""Check source syntax, notebook cleanliness and forbidden executable imports."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check():
    checked = 0
    for directory in ("src", "configs", "scripts", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                imports = ([a.name for a in node.names] if isinstance(node, ast.Import)
                           else [node.module or ""] if isinstance(node, ast.ImportFrom) else [])
                assert not any(n.split(".")[0] == "ultralytics" for n in imports), path
            checked += 1
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        doc = json.loads(path.read_text(encoding="utf-8"))
        assert doc["nbformat"] == 4 and doc["cells"], path
        ids = set()
        for cell in doc["cells"]:
            assert cell["id"] not in ids
            ids.add(cell["id"])
            if cell["cell_type"] == "code":
                assert cell["outputs"] == [] and cell["execution_count"] is None, path
                ast.parse("".join(cell["source"]), filename=str(path))
        checked += 1
    print(f"Checked {checked} source/notebook files; syntax and clean-output checks passed")


if __name__ == "__main__":
    check()

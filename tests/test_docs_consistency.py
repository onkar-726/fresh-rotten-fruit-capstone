"""The documentation must match the code: every script / option / file it mentions has to exist."""
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


@lru_cache(maxsize=None)
def help_text(script: str) -> str:
    r = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True, timeout=120,
                       env={"TF_CPP_MIN_LOG_LEVEL": "3", "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert r.returncode == 0, r.stderr[-500:]
    return r.stdout


def commands():
    """(doc, script, [flags]) for every `python scripts/x.py ...` / `python app/app.py ...` found in the docs."""
    for doc in DOCS:
        for line in doc.read_text().splitlines():
            # several commands may share one line: each owns the text up to the next "python"
            for m in re.finditer(r"python3? ((?:scripts|app)/[\w]+\.py)((?:(?!python3? ).)*)", line):
                flags = re.findall(r"(?<![\w-])(--[a-z][a-z0-9-]*)", m.group(2).split("#")[0])
                yield doc.name, m.group(1), flags


def test_every_documented_script_and_flag_exists():
    seen = set()
    for doc, script, flags in commands():
        assert (ROOT / script).exists(), f"{doc}: {script} does not exist"
        for flag in flags:
            assert flag in help_text(script), f"{doc}: {script} has no option {flag}"
            seen.add((script, flag))
    assert len(seen) > 12          # the docs really do contain commands


def test_every_script_is_documented_in_the_readme():
    readme = (ROOT / "README.md").read_text()
    for p in (ROOT / "scripts").iterdir():
        if p.suffix in (".py", ".sh") and not p.name.startswith("_"):
            assert p.stem in readme, f"scripts/{p.name} is not mentioned in README.md"


def test_documented_files_exist():
    text = "\n".join(d.read_text() for d in DOCS)
    for rel in set(re.findall(r"(?:configs|notebooks|slurm|docs)/[\w./-]+\.(?:json|ipynb|sbatch|md)", text)):
        assert (ROOT / rel).exists(), f"documented file missing: {rel}"


def test_notebooks_are_clean_and_use_existing_scripts():
    import nbformat

    for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = nbformat.read(nb_path, as_version=4)
        nbformat.validate(nb)
        assert all(not c.get("outputs") for c in nb.cells if c.cell_type == "code"), f"{nb_path.name} has stored outputs"
        src = "\n".join(c.source for c in nb.cells)
        for rel in re.findall(r"scripts/[\w]+\.(?:py|sh)", src):
            assert (ROOT / rel).exists(), f"{nb_path.name} mentions missing {rel}"
        assert "drive.mount" not in src and "MyDrive" not in src          # local-only storage


def test_makefile_targets_use_existing_scripts():
    for rel in re.findall(r"(?:scripts|app|tools)/[\w]+\.py", (ROOT / "Makefile").read_text()):
        if rel != "tools/build_manual.py":
            assert (ROOT / rel).exists(), rel

"""Every script must start and print --help (catches broken imports and renamed options)."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(p.name for p in (ROOT / "scripts").glob("*.py") if not p.name.startswith("_"))


@pytest.mark.parametrize("script", SCRIPTS + ["../app/app.py"])
def test_help(script):
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"], capture_output=True, text=True,
                       timeout=120, env={"TF_CPP_MIN_LOG_LEVEL": "3", "PATH": "/usr/bin:/bin:/usr/local/bin"})
    assert r.returncode == 0, r.stderr[-1500:]
    assert "usage" in r.stdout.lower()

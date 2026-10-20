import re
import subprocess
import sys
from pathlib import Path
import pytest
from src.common.utils import get_root_dir

ROOT_DIR = get_root_dir()

def test_readme_auto_markers_and_clean_presentation():
    readme_path = ROOT_DIR / "README.md"
    assert readme_path.exists()
    content = readme_path.read_text(encoding="utf-8")

    start_marker = "<!-- AUTO-METRICS-START -->"
    end_marker = "<!-- AUTO-METRICS-END -->"

    assert start_marker in content, f"README.md must contain {start_marker}"
    assert end_marker in content, f"README.md must contain {end_marker}"

    # Verify that outside the auto markers, there are no hardcoded metric tables or claims
    parts = content.split(start_marker)
    pre_block = parts[0]
    post_block = parts[1].split(end_marker)[1]
    outside_content = pre_block + "\n" + post_block

    # Assert no hardcoded metric table pipes with specific metric numbers
    # (such as $| \*\*FD001\*\* | ... | 14.28 |)
    assert not re.search(r"\|\s*\*\*FD001\*\*\s*\|[^|]+\|[^|]+\|\s*\d+\.\d+", outside_content), (
        "Outside AUTO-METRICS block, README.md must contain NO hardcoded performance metrics"
    )

def test_check_readme_matches_rendered():
    render_script = ROOT_DIR / "scripts" / "render_readme.py"
    assert render_script.exists(), "scripts/render_readme.py must exist"

    cmd = [sys.executable, str(render_script), "--check"]
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    assert res.returncode == 0, f"check-readme failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

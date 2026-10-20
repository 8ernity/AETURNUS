import json
import re
from pathlib import Path
import pytest
from src.common.utils import get_root_dir

ROOT_DIR = get_root_dir()

def test_no_unjustified_numeric_literals_in_web_presentation():
    web_src = ROOT_DIR / "web" / "src"
    allowlist_path = ROOT_DIR / "configs" / "presentation_allowlist.json"
    
    assert allowlist_path.exists(), "configs/presentation_allowlist.json must exist"
    with open(allowlist_path, "r", encoding="utf-8") as f:
        allowlist_data = json.load(f)

    # Allowlist format: list of dicts with {"file": "...", "pattern": "...", "justification": "..."}
    allowlist_entries = allowlist_data.get("entries", allowlist_data if isinstance(allowlist_data, list) else [])
    for entry in allowlist_entries:
        assert entry.get("justification", "").strip(), f"Allowlist entry {entry} missing non-empty justification"

    # Regexes for presentation numeric literals:
    # 1. $\d+ (currency literal like $250K, $1200)
    # 2. \d+\.\d+% (percentage literal like 86.5%, 95.0%)
    # 3. "\d+ of \d+" or '\d+ of \d+' or text \d+ of \d+
    regex_dollar = re.compile(r"\$\d+(?:,\d+)*(?:\.\d+)?(?:[kKmMbB])?")
    regex_pct = re.compile(r"\b\d+\.\d+%")
    regex_of = re.compile(r"\b\d+\s+of\s+\d+\b", re.IGNORECASE)

    violations = []

    for tsx_file in web_src.rglob("*.tsx"):
        rel_path = str(tsx_file.relative_to(ROOT_DIR)).replace("\\", "/")
        lines = tsx_file.read_text(encoding="utf-8").splitlines()

        for line_num, line in enumerate(lines, start=1):
            # Check comment lines or import lines
            stripped = line.strip()
            if stripped.startswith("//") or stripped.startswith("/*") or stripped.startswith("*"):
                continue

            matches = []
            for m in regex_dollar.finditer(line):
                matches.append(m.group(0))
            for m in regex_pct.finditer(line):
                matches.append(m.group(0))
            for m in regex_of.finditer(line):
                matches.append(m.group(0))

            for m in matches:
                # Check if matched by allowlist
                is_allowed = False
                for entry in allowlist_entries:
                    e_file = entry.get("file", "").replace("\\", "/")
                    if e_file in rel_path:
                        e_pattern = entry.get("pattern", entry.get("literal", ""))
                        if e_pattern and (e_pattern in m or re.search(re.escape(e_pattern), m)):
                            is_allowed = True
                            break
                        if entry.get("exact_line") and entry.get("exact_line") in line:
                            is_allowed = True
                            break

                if not is_allowed:
                    violations.append(
                        f"{rel_path}:{line_num} Found numeric literal '{m}' in line: {line.strip()}"
                    )

    assert not violations, (
        "Found hardcoded presentation numeric literals in web/src/**/*.tsx without allowlist justification:\n"
        + "\n".join(violations)
    )

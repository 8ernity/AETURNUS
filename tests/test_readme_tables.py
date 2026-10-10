import pytest
from pathlib import Path
from src.common.utils import get_root_dir
from scripts.generate_readme_tables import update_readme

root_dir = get_root_dir()

def test_generate_readme_tables_and_no_unimplemented_claims():
    update_readme()
    readme_path = root_dir / "README.md"
    assert readme_path.exists()

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Verify no fake claims
    assert "Hampel" not in content, "README must not claim Hampel"
    assert "BiGRU" not in content, "README must not claim BiGRU"
    assert "zero hardcoded numbers" not in content, "README must not claim 'zero hardcoded numbers'"

    # Verify tables present
    assert "| Dataset Subset | Operating Conditions |" in content
    assert "| Subset | RMSE ($mean \\pm std$) |" in content
    assert "FD001" in content
    assert "FD002" in content
    assert "FD003" in content
    assert "FD004" in content

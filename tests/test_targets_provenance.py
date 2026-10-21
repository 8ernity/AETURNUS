import json
import pytest
import yaml
from fastapi.testclient import TestClient
from src.api.server import app
from src.common.utils import get_root_dir, load_yaml
from scripts.evaluate_all_models import evaluate_subset
from scripts.render_readme import update_or_check_readme

client = TestClient(app)

def test_change_target_in_yaml_propagates_without_code_edits():
    """
    Test: change a target in YAML -> rerun evaluate -> README/API/UI change;
    no other file edits.
    """
    root_dir = get_root_dir()
    targets_file = root_dir / "configs" / "targets.yaml"
    manifest_file = root_dir / "artifacts" / "run_manifest.json"
    readme_file = root_dir / "README.md"

    targets_orig = targets_file.read_text(encoding="utf-8")
    manifest_orig = manifest_file.read_text(encoding="utf-8")
    readme_orig = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

    try:
        # Load targets
        targets_data = yaml.safe_load(targets_orig)
        # Flip FD001 target to an impossibly low value (e.g. 1.0) so target_met is guaranteed False
        targets_data["subsets"]["FD001"]["target_rmse"]["value"] = 1.0
        with open(targets_file, "w", encoding="utf-8") as f:
            yaml.dump(targets_data, f)

        # Rerun evaluate for FD001
        res = evaluate_subset("FD001", root_dir)
        assert res["target_met"] is False
        assert res["target_rmse"] == 1.0

        # Update manifest file with this evaluation
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        manifest_data["subsets"]["FD001"] = res
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

        # 1. API endpoint /api/v1/models/benchmark must reflect target_met == False
        resp = client.get("/api/v1/models/benchmark")
        assert resp.status_code == 200
        benchmarks = resp.json()["benchmarks"]
        fd001_bench = next(b for b in benchmarks if b["subset"] == "FD001")
        assert fd001_bench["target_met"] is False
        assert fd001_bench["target_rmse"] == 1.0

        # 2. README render must reflect target status
        success = update_or_check_readme(check_only=False)
        assert success is True
        readme_text = readme_file.read_text(encoding="utf-8")
        assert "TARGET MISSED" in readme_text

    finally:
        # Restore files to pristine state
        targets_file.write_text(targets_orig, encoding="utf-8")
        manifest_file.write_text(manifest_orig, encoding="utf-8")
        if readme_orig:
            readme_file.write_text(readme_orig, encoding="utf-8")

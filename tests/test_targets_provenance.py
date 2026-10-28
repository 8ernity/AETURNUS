import json
import pytest
import yaml
from fastapi.testclient import TestClient
from src.api.server import app
from src.common.utils import get_root_dir, load_yaml
from scripts.evaluate_all_models import evaluate_subset
from scripts.render_readme import update_or_check_readme

from tests.auth_helpers import ADMIN_HEADERS

client = TestClient(app, headers=ADMIN_HEADERS)

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
    active_bundle = root_dir / "artifacts" / "models" / "model_bundle_FD001.joblib"
    versioned_bundle = root_dir / "artifacts" / "models" / "model_bundle_FD001_v1.0.joblib"
    active_orig = active_bundle.read_bytes() if active_bundle.exists() else None
    versioned_orig = versioned_bundle.read_bytes() if versioned_bundle.exists() else None

    try:
        # Load targets
        targets_data = yaml.safe_load(targets_orig)
        original_target = float(
            targets_data["subsets"]["FD001"]["target_rmse"]["value"]
        )
        # Use an impossibly low target so target_met is guaranteed False, while
        # still exercising a real config change even if the shipped target is 1.0.
        test_target = 1.0 if original_target != 1.0 else 14.0
        targets_data["subsets"]["FD001"]["target_rmse"]["value"] = test_target
        with open(targets_file, "w", encoding="utf-8") as f:
            yaml.dump(targets_data, f)

        # Rerun evaluate for FD001 without promoting over the tracked serving bundle.
        res = evaluate_subset("FD001", root_dir, persist_active=False)
        assert res["target_met"] is False
        assert res["target_rmse"] == test_target

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
        assert fd001_bench["target_rmse"] == test_target

        # 2. README render must reflect target status
        success = update_or_check_readme(check_only=False)
        assert success is True
        readme_text = readme_file.read_text(encoding="utf-8")
        assert "TARGET MISSED" in readme_text

    finally:
        # Restore files to pristine state
        targets_file.write_text(targets_orig, encoding="utf-8")
        manifest_file.write_text(manifest_orig, encoding="utf-8")
        if active_orig is not None:
            active_bundle.write_bytes(active_orig)
        if versioned_orig is not None:
            versioned_bundle.write_bytes(versioned_orig)
        if readme_orig:
            readme_file.write_text(readme_orig, encoding="utf-8")

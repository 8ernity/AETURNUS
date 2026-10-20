import yaml
from pathlib import Path
import pytest
from src.common.utils import get_root_dir

ROOT_DIR = get_root_dir()

VALID_KINDS = {"policy", "assumption", "tuned_on_validation"}

def _check_provenance(node, path_str, violations):
    # Metadata keys or non-parameter structural keys that don't represent values
    ignored_keys = {"rules_version", "disclaimer", "subsets", "default_subset", "columns", "multi_regime_subsets", "notify_rules", "key_drift_sensors"}

    if isinstance(node, dict):
        # Check if this dict itself is a value-bearing object
        if "value" in node:
            # Must carry kind
            kind = node.get("kind")
            if not kind:
                violations.append(f"{path_str}: Missing 'kind' attribute (expected one of {VALID_KINDS})")
            elif kind not in VALID_KINDS:
                violations.append(f"{path_str}: Invalid 'kind' '{kind}', expected one of {VALID_KINDS}")
            
            source = node.get("source")
            if not source or not str(source).strip():
                violations.append(f"{path_str}: Missing or empty 'source' attribute")
                
            if kind == "tuned_on_validation":
                run_id = node.get("run_id")
                if not run_id or not str(run_id).strip():
                    violations.append(f"{path_str}: 'tuned_on_validation' must specify 'run_id'")
            return

        # Otherwise recurse into subkeys
        for k, v in node.items():
            if k in ignored_keys:
                continue
            sub_path = f"{path_str}.{k}" if path_str else k
            if isinstance(v, (int, float, bool, str)) and not isinstance(v, bool) and str(v).replace('.', '', 1).isdigit():
                # Raw numeric literal without kind wrapper
                violations.append(f"{sub_path}: Raw value '{v}' without 'kind' wrapper dict")
            else:
                _check_provenance(v, sub_path, violations)
                
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _check_provenance(item, f"{path_str}[{idx}]", violations)

def test_all_configs_have_provenance():
    configs_dir = ROOT_DIR / "configs"
    yaml_files = list(configs_dir.glob("*.yaml"))
    assert yaml_files, "No YAML files found in configs/"

    violations = []
    for yf in yaml_files:
        with open(yf, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data:
            _check_provenance(data, yf.name, violations)

    assert not violations, (
        f"Config values missing required provenance ('kind', 'source', and 'run_id' if tuned):\n"
        + "\n".join(violations)
    )

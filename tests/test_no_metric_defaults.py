import ast
import json
from pathlib import Path

from src.common.utils import get_root_dir

SCANNED_PACKAGES = ("src/api", "src/mlops", "src/economics", "src/agent", "src/eval")
ALLOWLIST_PATH = "configs/metric_defaults_allowlist.json"


def _numeric_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def test_no_numeric_get_defaults_for_metrics():
    root = get_root_dir()
    allowlist = json.loads((root / ALLOWLIST_PATH).read_text(encoding="utf-8"))
    allowed_keys = set(allowlist["allowed_keys"])

    violations = []
    for package in SCANNED_PACKAGES:
        for path in (root / package).rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) >= 2
                    and _numeric_constant(node.args[1])
                ):
                    continue
                key_node = node.args[0]
                key = key_node.value if isinstance(key_node, ast.Constant) else None
                if key in allowed_keys:
                    continue
                violations.append(
                    f"{path.relative_to(root)}:{node.lineno}: "
                    f".get({ast.unparse(key_node)}, {node.args[1].value!r})"
                )

    assert not violations, (
        "Metric-bearing .get() calls must not supply numeric defaults. "
        "Derive values from data/config/bundles or add an allowlisted identifier key "
        f"with justification in {ALLOWLIST_PATH}:\n" + "\n".join(violations)
    )

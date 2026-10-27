import ast
import json
import re
from pathlib import Path

from src.common.utils import get_root_dir

SCANNED_PACKAGES = ("src/api", "src/mlops", "src/economics", "src/agent", "src/eval")
ALLOWLIST_PATH = "configs/metric_defaults_allowlist.json"
TRIAGE_POLICY_SOURCE = "src/agent/triage_agent.py"

SEVERITY_OR_RUL_NAME = re.compile(r"(?:^|_)(?:rul|severity)(?:_|$)|rul$|severity$", re.IGNORECASE)


def _numeric_constant(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    )


def _identifier_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _references_policy_name(node: ast.AST) -> bool:
    return any(
        SEVERITY_OR_RUL_NAME.search(_identifier_name(child))
        for child in ast.walk(node)
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


def test_no_numeric_rul_or_severity_comparisons_outside_triage_agent():
    """Policy thresholds belong in agent_rules.yaml, not inline comparisons."""
    root = get_root_dir()
    violations = []

    comparison_ops = (ast.Lt, ast.LtE, ast.Gt, ast.GtE)
    for path in (root / "src").rglob("*.py"):
        relative = path.relative_to(root).as_posix()
        if relative == TRIAGE_POLICY_SOURCE:
            continue

        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            if not any(isinstance(op, comparison_ops) for op in node.ops):
                continue

            operands = [node.left, *node.comparators]
            for index, operand in enumerate(operands):
                if not _numeric_constant(operand):
                    continue
                others = operands[:index] + operands[index + 1:]
                if any(_references_policy_name(other) for other in others):
                    violations.append(
                        f"{relative}:{node.lineno}: inline numeric policy comparison "
                        f"`{ast.unparse(node)}`"
                    )
                    break

    assert not violations, (
        "Numeric RUL/severity thresholds must be loaded from configs/agent_rules.yaml "
        "and evaluated by src/agent/triage_agent.py, not duplicated inline:\n"
        + "\n".join(violations)
    )

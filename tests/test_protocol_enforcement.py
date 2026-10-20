import re
import ast
from pathlib import Path
import pytest
from src.common.utils import get_root_dir

ROOT_DIR = get_root_dir()

def test_protocol_module_exists_and_exports():
    protocol_path = ROOT_DIR / "src" / "eval" / "protocol.py"
    assert protocol_path.exists(), "src/eval/protocol.py must exist as the single evaluation protocol"
    
    from src.eval import protocol
    assert hasattr(protocol, "sample_truncation"), "protocol must export sample_truncation"
    assert hasattr(protocol, "cap_rul"), "protocol must export cap_rul"
    assert hasattr(protocol, "calculate_nasa_score"), "protocol must export calculate_nasa_score"
    assert hasattr(protocol, "calculate_metrics"), "protocol must export calculate_metrics"
    assert hasattr(protocol, "get_protocol_id"), "protocol must export get_protocol_id"

def test_no_other_module_reimplements_metrics_or_truncation():
    src_dir = ROOT_DIR / "src"
    protocol_file = (src_dir / "eval" / "protocol.py").resolve()
    
    # Forbidden patterns outside protocol.py:
    # 1. Defining metric calculation functions
    forbidden_def_patterns = [
        re.compile(r"def\s+calculate_nasa_score\s*\("),
        re.compile(r"def\s+calculate_metrics\s*\("),
        re.compile(r"def\s+nasa_score\s*\("),
        re.compile(r"def\s+sample_truncation\s*\("),
    ]
    
    # 2. Inlined NASA score formula: exp(-d/13) or exp(d/10)
    nasa_inline_pattern = re.compile(r"(?:np\.exp|math\.exp)\s*\([^)]*(?:/\s*13|/\s*10)")

    violations = []
    for py_file in src_dir.rglob("*.py"):
        if py_file.resolve() == protocol_file:
            continue
        # Allow __pycache__ ignore
        if "__pycache__" in py_file.parts:
            continue
            
        content = py_file.read_text(encoding="utf-8")
        rel_path = str(py_file.relative_to(ROOT_DIR))
        
        for pat in forbidden_def_patterns:
            if pat.search(content):
                violations.append(f"{rel_path}: Defines forbidden metric/truncation function matching {pat.pattern}")
                
        if nasa_inline_pattern.search(content):
            violations.append(f"{rel_path}: Re-implements inline NASA score asymmetric formula")
            
    assert not violations, "Found duplicate metric/truncation implementations:\n" + "\n".join(violations)

def test_consumers_import_from_protocol():
    consumers = [
        ROOT_DIR / "src" / "models" / "model_service.py",
        ROOT_DIR / "src" / "economics" / "cost_simulator.py",
        ROOT_DIR / "scripts" / "evaluate_all_models.py",
    ]
    for path in consumers:
        assert path.exists(), f"Consumer file missing: {path}"
        content = path.read_text(encoding="utf-8")
        assert "from src.eval.protocol import" in content or "import src.eval.protocol" in content, (
            f"{path.relative_to(ROOT_DIR)} must import from src.eval.protocol"
        )

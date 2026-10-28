import sys
import json
import argparse
import hashlib
import re
import subprocess
from collections import Counter
from functools import lru_cache
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.utils import get_root_dir, load_yaml

START_MARKER = "<!-- AUTO-METRICS-START -->"
END_MARKER = "<!-- AUTO-METRICS-END -->"

PROMOTION_START_MARKER = "<!-- AUTO-PROMOTION-GATE-START -->"
PROMOTION_END_MARKER = "<!-- AUTO-PROMOTION-GATE-END -->"

TEST_BADGE_START_MARKER = "<!-- AUTO-TEST-BADGE-START -->"
TEST_BADGE_END_MARKER = "<!-- AUTO-TEST-BADGE-END -->"

TEST_SUMMARY_START_MARKER = "<!-- AUTO-TEST-SUMMARY-START -->"
TEST_SUMMARY_END_MARKER = "<!-- AUTO-TEST-SUMMARY-END -->"

TEST_TABLE_START_MARKER = "<!-- AUTO-TEST-TABLE-START -->"
TEST_TABLE_END_MARKER = "<!-- AUTO-TEST-TABLE-END -->"

from src.ingest.loader import CMAPSSLoader

def render_metrics_block(manifest: dict) -> str:
    lines = [
        START_MARKER,
        "### 🏆 C-MAPSS Empirical Benchmark Performance (FD001–FD004)",
        "",
        "> [!NOTE]",
        "> **Canonical Run Provenance**<br>",
        f"> **Run ID**: `{manifest.get('run_id', 'N/A')}` | **Git SHA**: `{manifest.get('git_sha', 'N/A')[:10]}` | **Config Hash**: `{manifest.get('config_hash', 'N/A')[:10]}`<br>",
        f"> **Target Metric**: `{manifest.get('target_metric', {}).get('value', 'N/A')}` | **Coverage Metric**: `{manifest.get('coverage_metric', {}).get('value', 'N/A')}` | **Coverage Target**: `{manifest.get('target_coverage', {}).get('value', 'N/A')}` | **RUL Cap**: `{next((s.get('metrics', {}).get('rul_cap') for s in manifest.get('subsets', {}).values() if s.get('metrics', {}).get('rul_cap') is not None), 'N/A')}`<br>",
        "> *All figures are derived directly from model bundles and authentic C-MAPSS evaluation parquets at run time. Censored predictions cover any true RUL at or above p10 and are never rendered as closed intervals.*",
        "",
        "| Dataset Subset | Operating Conditions | Fault Modes | Test Engines | Champion Model | Test RMSE (Raw Truth) | Test RMSE (Capped Truth) | True RUL Above Cap | RMSE (True RUL $\\le$ Cap) | Test MAE | NASA Asymmetric Score | PR-AUC (Fail $\\le 30$) | Coverage All (Censored) | Coverage (True RUL $\\le$ Cap) | Coverage Status | RMSE Target Status |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]

    subsets = manifest.get("subsets", {})
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        s_data = subsets.get(sub, {})
        m = s_data.get("metrics", {})
        
        # Read operating conditions and fault modes strictly from loader metadata
        loader = CMAPSSLoader(subset=sub)
        cond = loader.operating_conditions
        faults = loader.fault_modes
        n_eng = s_data["n_engines"] if "n_engines" in s_data else len(loader.load_raw_data()[2])
        algo = s_data["algorithm"].upper()
        
        raw_rmse = f"**{m['rmse_vs_raw_truth']:.2f}**" if m.get("rmse_vs_raw_truth") is not None else "N/A"
        capped_rmse = f"{m['rmse_vs_capped_truth']:.2f}" if m.get("rmse_vs_capped_truth") is not None else "N/A"
        cap_share = f"{m['share_true_rul_above_cap'] * 100:.1f}%" if m.get("share_true_rul_above_cap") is not None else "N/A"
        le_cap_rmse = f"{m['rmse_on_true_rul_le_cap']:.2f}" if m.get("rmse_on_true_rul_le_cap") is not None else "N/A"
        mae = f"{m['mae']:.2f}" if "mae" in m else "N/A"
        nasa = f"**{m['nasa_score']:,.1f}**" if "nasa_score" in m else "N/A"
        pr_auc = f"{m['pr_auc']:.3f}" if "pr_auc" in m else "N/A"
        cov_all_value = m.get("coverage_all", m.get("interval_coverage"))
        cov_all = f"{cov_all_value * 100:.1f}%" if cov_all_value is not None else "N/A"
        cov_cap_value = m.get("coverage_le_cap")
        cov_cap = f"{cov_cap_value * 100:.1f}%" if cov_cap_value is not None else "N/A"

        # Read target_met strictly from manifest; never infer or default it here.
        target_met = s_data["target_met"]
        if target_met is True:
            status = '<span style="color:#22c55e;font-weight:bold;">TARGET REACHED</span>'
        elif target_met is False:
            status = '<span style="color:#ef4444;font-weight:bold;">TARGET MISSED</span>'
        else:
            status = '<span style="color:#94a3b8;font-weight:bold;">NO DATA</span>'

        coverage_met = s_data.get("target_coverage_met")
        if coverage_met is True:
            coverage_status = '<span style="color:#22c55e;font-weight:bold;">COVERAGE MET</span>'
        elif coverage_met is False:
            coverage_status = '<span style="color:#ef4444;font-weight:bold;">COVERAGE MISSED</span>'
        else:
            coverage_status = '<span style="color:#94a3b8;font-weight:bold;">NO DATA</span>'

        lines.append(
            f"| **{sub}** | {cond} | {faults} | {n_eng} | **{algo}** | {raw_rmse} | "
            f"{capped_rmse} | {cap_share} | {le_cap_rmse} | {mae} | {nasa} | {pr_auc} | "
            f"{cov_all} | {cov_cap} | {coverage_status} | {status} |"
        )

    lines.extend([
        "",
        "### 🧪 RUL-Bucket Error Breakdown (Raw vs Capped Truth)",
        "",
        "Target metric and cap are declared in `configs/targets.yaml` and `configs/pipeline.yaml`; this table is rendered from the same manifest.",
        "",
        "| Subset | RUL Bucket | Test Engines | RMSE (Raw Truth) | RMSE (Capped Truth) | True RUL Above Cap | RMSE (True RUL $\\le$ Cap) |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ])

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        s_data = subsets.get(sub, {})
        for bucket in s_data.get("metrics", {}).get("rul_bucket_metrics", []):
            bucket_raw = f"{bucket['rmse_vs_raw_truth']:.2f}" if bucket.get("rmse_vs_raw_truth") is not None else "N/A"
            bucket_capped = f"{bucket['rmse_vs_capped_truth']:.2f}" if bucket.get("rmse_vs_capped_truth") is not None else "N/A"
            bucket_share = f"{bucket['share_true_rul_above_cap'] * 100:.1f}%" if bucket.get("share_true_rul_above_cap") is not None else "N/A"
            bucket_le_cap = f"{bucket['rmse_on_true_rul_le_cap']:.2f}" if bucket.get("rmse_on_true_rul_le_cap") is not None else "N/A"
            lines.append(
                f"| **{sub}** | {bucket['bucket']} | {bucket['sample_count']} | "
                f"{bucket_raw} | {bucket_capped} | {bucket_share} | {bucket_le_cap} |"
            )

    lines.extend([
        "",
        "### 🔬 5-Fold GroupKFold Cross-Validation (Censored, Persistent Engines)",
        "",
        "Validation targets are censored with the same random-truncation protocol as the official test set; the CV protocol ID below hashes the mode and draw count actually used.",
        "",
        "| Subset | RMSE ($mean \\pm std$) | NASA Score ($mean \\pm std$) | Algorithm | CV Protocol ID |",
        "|:---:|:---:|:---:|:---:|:---:|"
    ])

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        s_data = subsets.get(sub, {})
        cv = s_data.get("cv", {})
        algo = s_data["algorithm"]
        proto = s_data["cv_protocol_id"]
        cv_rmse = f"${cv['mean_rmse']:.2f} \\pm {cv['std_rmse']:.2f}$" if ("mean_rmse" in cv and "std_rmse" in cv) else "N/A"
        cv_nasa = f"${cv['mean_nasa']:,.1f} \\pm {cv['std_nasa']:.1f}$" if ("mean_nasa" in cv and "std_nasa" in cv) else "N/A"
        lines.append(f"| **{sub}** | {cv_rmse} | {cv_nasa} | `{algo}` | `{proto}` |")

    lines.extend([
        "",
        END_MARKER
    ])

    return "\n".join(lines)

def _format_criterion_requirement(key: str, value) -> str:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    if isinstance(value, (int, float)):
        if "min" in key:
            comparator = ">="
        elif "max" in key:
            comparator = "<="
        else:
            comparator = "="
        if key.endswith("_pct"):
            suffix = "%"
        elif key.endswith("_pp"):
            suffix = " percentage points"
        else:
            suffix = ""
        return f"{comparator} {value}{suffix}"
    return str(value)


def render_promotion_gate_block() -> str:
    config = load_yaml("configs/monitoring.yaml", raw=True)
    criteria = config.get("promotion_gate", {})
    lines = [
        PROMOTION_START_MARKER,
        "### Promotion Gate Criteria",
        "",
        "Rendered from [`configs/monitoring.yaml`](configs/monitoring.yaml) at documentation-build time. "
        f"The gate evaluates **{len(criteria)} configured criteria**; no threshold is typed into this README.",
        "",
        "| Criterion | Configured Requirement | Provenance |",
        "|---|---|---|",
    ]
    for key, node in criteria.items():
        if not isinstance(node, dict) or "value" not in node:
            continue
        label = key.replace("_", " ").strip().title()
        requirement = _format_criterion_requirement(key, node["value"]).replace("|", "\\|")
        source = str(node.get("source", "Configured starting value")).replace("|", "\\|")
        lines.append(f"| **{label}** | `{requirement}` | {source} |")
    lines.append(PROMOTION_END_MARKER)
    return "\n".join(lines)


def _test_files_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT_DIR / "tests").glob("test_*.py")):
        stat = path.stat()
        digest.update(path.name.encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


@lru_cache(maxsize=1)
def collect_test_counts() -> tuple[int, dict[str, int]]:
    """Return collected-test counts per module from pytest's own collector.

    Collection imports the whole application and takes tens of seconds, so the
    result is cached against test-file sizes and mtimes. The cache is still
    derived exclusively from ``pytest --collect-only`` output; it prevents each
    README check from re-importing the entire ML stack.
    """
    cache_path = ROOT_DIR / "artifacts" / "reports" / "test_collection.json"
    fingerprint = _test_files_fingerprint()
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint and cached.get("counts"):
                counts = {str(k): int(v) for k, v in cached["counts"].items()}
                return int(cached["total"]), dict(sorted(counts.items()))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", "tests"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + "\n" + (result.stderr or "")
    counts: Counter[str] = Counter()
    for line in output.splitlines():
        match = re.match(r"^(tests[\\/][^:]+)::", line.strip())
        if match:
            counts[match.group(1).replace("\\", "/")] += 1
    if not counts:
        raise RuntimeError(
            "pytest --collect-only produced no test IDs; cannot render README test counts.\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    payload = {
        "fingerprint": fingerprint,
        "total": sum(counts.values()),
        "counts": dict(sorted(counts.items())),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload["total"], payload["counts"]


def render_test_badge_block() -> str:
    total, _ = collect_test_counts()
    return "\n".join(
        [
            TEST_BADGE_START_MARKER,
            f"[![Tests](https://img.shields.io/badge/Tests-{total}%20Collected-brightgreen.svg?style=flat&logo=pytest)](https://docs.pytest.org/)",
            TEST_BADGE_END_MARKER,
        ]
    )


def render_test_summary_block() -> str:
    total, modules = collect_test_counts()
    return "\n".join(
        [
            TEST_SUMMARY_START_MARKER,
            f"The platform includes a rigorous **{total}-test Pytest suite across {len(modules)} test modules** and an autonomous **CI smoke test**:",
            TEST_SUMMARY_END_MARKER,
        ]
    )


def _existing_test_descriptions(content: str) -> dict[str, str]:
    descriptions: dict[str, str] = {}
    old_pattern = re.compile(
        r"^\|\s*`(tests/[^`]+)`\s*\|\s*(.*?)\s*\|\s*\d+\s*\|$"
    )
    new_pattern = re.compile(
        r"^\|\s*`(tests/[^`]+)`\s*\|\s*\d+\s*\|\s*(.*?)\s*\|$"
    )
    for line in content.splitlines():
        stripped = line.strip()
        match = old_pattern.match(stripped) or new_pattern.match(stripped)
        if match:
            descriptions[match.group(1)] = match.group(2)
    return descriptions


def render_test_table_block(content: str) -> str:
    total, modules = collect_test_counts()
    descriptions = _existing_test_descriptions(content)
    lines = [
        TEST_TABLE_START_MARKER,
        f"### Test Coverage by Module ({len(modules)} Modules, {total} Collected Tests)",
        "",
        "Counts are generated by `pytest --collect-only`; descriptions are retained from the maintained test map.",
        "",
        "| Test File | Test Count | Verified Functionality & Invariants |",
        "|---|---:|:---|",
    ]
    for module, count in sorted(modules.items()):
        description = descriptions.get(
            module,
            "Automated test module collected by the Pytest suite.",
        )
        lines.append(f"| `{module}` | {count} | {description} |")
    lines.append(TEST_TABLE_END_MARKER)
    return "\n".join(lines)


def _replace_marked_block(content: str, start: str, end: str, rendered: str) -> str:
    if start not in content or end not in content:
        raise ValueError(f"README.md missing {start} or {end}")
    before_part, rest = content.split(start, 1)
    _, after_part = rest.split(end, 1)
    return before_part + rendered + after_part


def _normalize_block(block: str) -> str:
    return "\n".join(line.rstrip() for line in block.strip().splitlines())


def update_or_check_readme(check_only: bool = False) -> bool:
    root_dir = get_root_dir()
    manifest_file = root_dir / "artifacts" / "run_manifest.json"
    if not manifest_file.exists():
        print(f"[ERROR] Run manifest {manifest_file} missing. Run 'make evaluate' first.", file=sys.stderr)
        return False

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    readme_path = root_dir / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    blocks = [
        (START_MARKER, END_MARKER, render_metrics_block(manifest)),
        (PROMOTION_START_MARKER, PROMOTION_END_MARKER, render_promotion_gate_block()),
        (TEST_BADGE_START_MARKER, TEST_BADGE_END_MARKER, render_test_badge_block()),
        (TEST_SUMMARY_START_MARKER, TEST_SUMMARY_END_MARKER, render_test_summary_block()),
        (TEST_TABLE_START_MARKER, TEST_TABLE_END_MARKER, render_test_table_block(content)),
    ]

    missing = [
        f"{start} / {end}"
        for start, end, _ in blocks
        if start not in content or end not in content
    ]
    if missing:
        print(f"[ERROR] README.md missing required markers: {', '.join(missing)}", file=sys.stderr)
        return False

    if check_only:
        for start, end, rendered in blocks:
            before_part, rest = content.split(start, 1)
            current_middle, after_part = rest.split(end, 1)
            current_block = start + current_middle + end
            if _normalize_block(current_block) != _normalize_block(rendered):
                print(
                    f"[FAIL] README block {start} differs from its config/pytest source!",
                    file=sys.stderr,
                )
                print(
                    "--- Committed in README:\n",
                    _normalize_block(current_block)[:300],
                    "...\n--- Rendered:\n",
                    _normalize_block(rendered)[:300],
                    "...",
                    file=sys.stderr,
                )
                return False
        print("[PASS] README metrics, promotion gate, and test counts match their sources exactly.")
        return True

    new_content = content
    for start, end, rendered in blocks:
        new_content = _replace_marked_block(new_content, start, end, rendered)
    readme_path.write_text(new_content, encoding="utf-8")
    print("[SUCCESS] Successfully updated README.md from manifest, monitoring config, and pytest collection.")
    return True

def main():
    parser = argparse.ArgumentParser(description="Render or check README metrics block from run_manifest.json")
    parser.add_argument("--check", action="store_true", help="Fail if committed block differs from rendered one")
    args = parser.parse_args()

    success = update_or_check_readme(check_only=args.check)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()

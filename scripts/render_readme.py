import sys
import json
import argparse
import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.utils import get_root_dir

START_MARKER = "<!-- AUTO-METRICS-START -->"
END_MARKER = "<!-- AUTO-METRICS-END -->"

from src.ingest.loader import CMAPSSLoader

def render_metrics_block(manifest: dict) -> str:
    lines = [
        START_MARKER,
        "### 🏆 C-MAPSS Empirical Benchmark Performance (FD001–FD004)",
        "",
        "> [!NOTE]",
        "> **Canonical Run Provenance**  ",
        f"> **Run ID**: `{manifest.get('run_id', 'N/A')}` | **Git SHA**: `{manifest.get('git_sha', 'N/A')[:10]}` | **Config Hash**: `{manifest.get('config_hash', 'N/A')[:10]}`  ",
        "> *All figures are derived directly from model bundles and authentic C-MAPSS evaluation parquets at run time.*",
        "",
        "| Dataset Subset | Operating Conditions | Fault Modes | Test Engines | Champion Model | Test RMSE | Test MAE | NASA Asymmetric Score | PR-AUC (Fail $\\le 30$) | Conformal Coverage | Target Status |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
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
        algo = s_data.get("algorithm", "lightgbm").upper()
        
        rmse = f"**{m['rmse']:.2f}**" if "rmse" in m else "N/A"
        mae = f"{m['mae']:.2f}" if "mae" in m else "N/A"
        nasa = f"**{m['nasa_score']:,.1f}**" if "nasa_score" in m else "N/A"
        pr_auc = f"{m['pr_auc']:.3f}" if "pr_auc" in m else "N/A"
        cov = f"{m['interval_coverage'] * 100:.1f}%" if "interval_coverage" in m else "N/A"
        
        # Read target_met strictly from manifest
        target_met = s_data.get("target_met", False)
        status = '<span style="color:#22c55e;font-weight:bold;">TARGET REACHED</span>' if target_met else '<span style="color:#ef4444;font-weight:bold;">TARGET MISSED</span>'
        
        lines.append(f"| **{sub}** | {cond} | {faults} | {n_eng} | **{algo}** | {rmse} | {mae} | {nasa} | {pr_auc} | {cov} | {status} |")

    lines.extend([
        "",
        "### 🔬 5-Fold GroupKFold Cross-Validation (Persistent Engines)",
        "",
        "| Subset | RMSE ($mean \\pm std$) | NASA Score ($mean \\pm std$) | Algorithm | Protocol ID |",
        "|:---:|:---:|:---:|:---:|:---:|"
    ])

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        s_data = subsets.get(sub, {})
        cv = s_data.get("cv", {})
        algo = s_data.get("algorithm", "lightgbm")
        proto = s_data.get("protocol_id", "canonical")
        cv_rmse = f"${cv['mean_rmse']:.2f} \\pm {cv['std_rmse']:.2f}$" if ("mean_rmse" in cv and "std_rmse" in cv) else "N/A"
        cv_nasa = f"${cv['mean_nasa']:,.1f} \\pm {cv['std_nasa']:.1f}$" if ("mean_nasa" in cv and "std_nasa" in cv) else "N/A"
        lines.append(f"| **{sub}** | {cv_rmse} | {cv_nasa} | `{algo}` | `{proto}` |")

    lines.extend([
        "",
        END_MARKER
    ])

    return "\n".join(lines)

def update_or_check_readme(check_only: bool = False) -> bool:
    root_dir = get_root_dir()
    manifest_file = root_dir / "artifacts" / "run_manifest.json"
    if not manifest_file.exists():
        print(f"[ERROR] Run manifest {manifest_file} missing. Run 'make evaluate' first.", file=sys.stderr)
        return False

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    rendered_block = render_metrics_block(manifest)
    readme_path = root_dir / "README.md"
    content = readme_path.read_text(encoding="utf-8")

    if START_MARKER not in content or END_MARKER not in content:
        print(f"[ERROR] README.md missing {START_MARKER} or {END_MARKER}", file=sys.stderr)
        return False

    # Extract existing block
    before_part, rest = content.split(START_MARKER, 1)
    current_middle, after_part = rest.split(END_MARKER, 1)
    current_block = START_MARKER + current_middle + END_MARKER

    if check_only:
        # Strict comparison
        current_norm = "\n".join(line.rstrip() for line in current_block.strip().splitlines())
        rendered_norm = "\n".join(line.rstrip() for line in rendered_block.strip().splitlines())
        if current_norm != rendered_norm:
            print("[FAIL] README committed metrics differ from artifacts/run_manifest.json rendered block!", file=sys.stderr)
            print("--- Committed in README:\n", current_norm[:300], "...\n--- Rendered from Manifest:\n", rendered_norm[:300], "...", file=sys.stderr)
            return False
        print("[PASS] README metrics match run manifest exactly.")
        return True
    else:
        new_content = before_part + rendered_block + after_part
        readme_path.write_text(new_content, encoding="utf-8")
        print("[SUCCESS] Successfully updated README.md from run_manifest.json")
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

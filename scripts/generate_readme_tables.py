import sys
import json
import re
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.common.utils import get_root_dir

def generate_subset_table(reports_dir: Path) -> str:
    subsets_meta = {
        "FD001": ("Single (Sea Level)", "HPC Degradation"),
        "FD002": ("Six Regimes (Multi)", "HPC Degradation"),
        "FD003": ("Single (Sea Level)", "HPC + Fan Degradation"),
        "FD004": ("Six Regimes (Multi)", "HPC + Fan Degradation"),
    }

    headers = [
        "| Dataset Subset | Operating Conditions | Fault Modes | Test Engines | Champion RMSE | NASA Asymmetric Score | PR-AUC (Fail $\\le 30$) | Conformal Coverage | Target Status |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|"
    ]
    rows = []

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        eval_file = reports_dir / f"evaluation_{sub}.json"
        cond, faults = subsets_meta.get(sub, ("Standard", "Degradation"))
        if eval_file.exists():
            with open(eval_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            rmse = f"**{d.get('rmse', 0.0):.2f}**"
            nasa = f"**{d.get('nasa_score', 0.0):,.1f}**"
            pr_auc = f"**{d.get('pr_auc', 0.0):.3f}**"
            cov = f"{d.get('interval_coverage', 0.80) * 100:.1f}%"
            n_engines = d.get("test_engines_count", 100)
            status_text = d.get("status", "Target Met").upper()
            if "MET" in status_text and "MISSED" not in status_text:
                status_html = '<span style="color:#22c55e;font-weight:bold;">TARGET MET</span>'
            else:
                status_html = '<span style="color:#ef4444;font-weight:bold;">TARGET MISSED</span>'
        else:
            rmse, nasa, pr_auc, cov, n_engines = "N/A", "N/A", "N/A", "N/A", "N/A"
            status_html = "PENDING"

        rows.append(f"| **{sub}** | {cond} | {faults} | {n_engines} | {rmse} | {nasa} | {pr_auc} | {cov} | {status_html} |")

    return "\n".join(headers + rows)


def generate_multi_seed_table(reports_dir: Path) -> str:
    headers = [
        "| Subset | RMSE ($mean \\pm std$) | MAE ($mean \\pm std$) | NASA Score ($mean \\pm std$) | PR-AUC ($mean \\pm std$) |",
        "|:---:|:---:|:---:|:---:|:---:|"
    ]
    rows = []

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        eval_file = reports_dir / f"evaluation_{sub}.json"
        if eval_file.exists():
            with open(eval_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            ms = d.get("multi_seed", {})
            if ms:
                rmse = f"${ms.get('rmse_mean', 0.0):.2f} \\pm {ms.get('rmse_std', 0.0):.2f}$"
                mae = f"${ms.get('mae_mean', 0.0):.2f} \\pm {ms.get('mae_std', 0.0):.2f}$"
                nasa = f"${ms.get('nasa_mean', 0.0):,.1f} \\pm {ms.get('nasa_std', 0.0):.1f}$"
                prauc = f"${ms.get('pr_auc_mean', 0.0):.3f} \\pm {ms.get('pr_auc_std', 0.0):.3f}$"
            else:
                rmse = f"${d.get('rmse', 0.0):.2f}$"
                mae = f"${d.get('mae', 0.0):.2f}$"
                nasa = f"${d.get('nasa_score', 0.0):,.1f}$"
                prauc = f"${d.get('pr_auc', 0.0):.3f}$"
        else:
            rmse, mae, nasa, prauc = "N/A", "N/A", "N/A", "N/A"

        rows.append(f"| **{sub}** | {rmse} | {mae} | {nasa} | {prauc} |")

    return "\n".join(headers + rows)


def generate_model_comparison_table(reports_dir: Path) -> str:
    headers = [
        "| Subset | Architecture | Model Name | 5-Fold CV RMSE (Train) | Test RMSE | Test MAE | Training Speed |",
        "|:---:|:---:|:---|:---:|:---:|:---:|:---:|"
    ]
    rows = []

    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        comp_file = reports_dir / f"model_comparison_{sub}.json"
        if comp_file.exists():
            with open(comp_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            champ_name = d.get("champion", "")
            for m in d.get("models", []):
                name = m.get("model_name", "Model")
                arch = m.get("architecture", "")
                cv_rmse = f"{m.get('cv_mean_rmse', 0.0):.2f}" if "cv_mean_rmse" in m else "N/A"
                rmse = f"{m.get('rmse', 0.0):.2f}"
                mae = f"{m.get('mae', 0.0):.2f}"
                time_s = m.get("train_time_sec", 0.0)
                speed = f"~{time_s:.1f}s" if time_s > 0 else "N/A"
                is_champ = (name == champ_name) or ("Champion" in name)
                if is_champ:
                    display_name = name if "Champion" in name else f"{name} (Champion)"
                    rows.append(f"| **{sub}** | **{arch}** | **{display_name}** | **{cv_rmse}** | **{rmse}** | **{mae}** | *{speed}* |")
                else:
                    rows.append(f"| {sub} | {arch} | {name} | {cv_rmse} | {rmse} | {mae} | {speed} |")

    if not rows:
        return ""
    return "\n".join(headers + rows)


def update_readme():
    from scripts.render_readme import update_or_check_readme
    return update_or_check_readme(check_only=False)

def _unused_old_update_readme():
    root_dir = get_root_dir()
    readme_path = root_dir / "README.md"
    reports_dir = root_dir / "artifacts" / "reports"

    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Remove unsupported claims
    content = content.replace("with absolute zero hardcoded numbers", "with empirical verification from artifact reports")
    content = content.replace("Hampel outlier filtering", "robust outlier clipping")
    content = content.replace("Hampel clipping", "outlier clipping")
    content = content.replace("Causal Imputer & Hampel", "Causal Imputer & Outlier Clipper")
    content = content.replace("Hampel", "Outlier Clipper")
    content = content.replace("2-Layer BiGRU + Linear", "Sequence GRU + Linear")
    content = content.replace("BiGRU", "GRU")
    content = content.replace("Zero-Falsification Guarantee", "Empirically Verified Metrics")
    content = content.replace("Zero-Falsification", "Empirical Verification")
    content = content.replace("100% deterministic", "Fully deterministic")
    content = content.replace("42%20Passed", "69%20Passed")
    content = content.replace("42-test Pytest suite", "69-test Pytest suite across 20 test modules")
    content = content.replace("42-test", "69-test")

    # 2. Update Subset Performance Matrix table
    subset_tbl = generate_subset_table(reports_dir)
    subset_pattern = r"(### Subset Performance Matrix \(FD001–FD004\)\s*\n\n[^\n]+\n\n)\| Dataset Subset \|[\s\S]*?(?=\n\n> \[!NOTE\]|\n\n### Multi-Seed)"
    content = re.sub(subset_pattern, lambda m: m.group(1) + subset_tbl, content)

    # 3. Update Multi-Seed Robustness table
    multi_seed_tbl = generate_multi_seed_table(reports_dir)
    seed_pattern = r"(### Multi-Seed Robustness Analysis\s*\n\n[^\n]+\n\n)\| Subset \|[\s\S]*?(?=\n\n### Deep Learning|\n\n---)"
    content = re.sub(seed_pattern, lambda m: m.group(1) + multi_seed_tbl, content)

    # 4. Update Deep Learning comparison table if present
    comp_tbl = generate_model_comparison_table(reports_dir)
    if comp_tbl:
        comp_pattern = r"(### Deep Learning Architecture Comparison\s*\n\n[^\n]+\n\n)\| Subset \|[\s\S]*?(?=\n\n---|\n\n## 🧠)"
        content = re.sub(comp_pattern, lambda m: m.group(1) + comp_tbl, content)

    # 5. Fix stale empirical coverage prose from actual evaluation reports
    coverages = {}
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        f_eval = reports_dir / f"evaluation_{sub}.json"
        if f_eval.exists():
            try:
                with open(f_eval, "r", encoding="utf-8") as f:
                    cov_val = json.load(f).get("interval_coverage")
                    if cov_val is not None:
                        coverages[sub] = float(cov_val) * 100.0
            except Exception:
                pass
    if coverages:
        min_cov = min(coverages.values())
        max_cov = max(coverages.values())
        c1 = f"{coverages.get('FD001', 86.0):.1f}%"
        c2 = f"{coverages.get('FD002', 62.6):.1f}%"
        c3 = f"{coverages.get('FD003', 82.0):.1f}%"
        c4 = f"{coverages.get('FD004', 60.0):.1f}%"
        cov_section = (
            f"#### 🎯 Empirical Coverage ({min_cov:.1f}%–{max_cov:.1f}%) vs 80% Nominal Target\n"
            f"On standard held-out NASA C-MAPSS test engines, the empirical conformal coverage achieved across subsets is:\n"
            f"- **FD001**: **{c1}** (Single condition, single fault — well exceeds nominal 80% target)\n"
            f"- **FD002**: **{c2}** (Six operational regimes, single fault — non-stationary flight conditions)\n"
            f"- **FD003**: **{c3}** (Single condition, dual fault — compounding HPC + Fan wear)\n"
            f"- **FD004**: **{c4}** (Six operational regimes, dual fault — maximum operating complexity)\n\n"
            f"**Why empirical coverage varies across regimes**:"
        )
        content = re.sub(
            r"#### 🎯 Empirical Coverage \([\d\.\%–\-]+\) vs 80% Nominal Target[\s\S]*?\*\*Why empirical coverage is [^\*]+:\*\*",
            cov_section,
            content
        )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)

    print("[SUCCESS] README tables and claims successfully generated and updated.")

if __name__ == "__main__":
    update_readme()

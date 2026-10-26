import json

from src.common.utils import get_root_dir
from src.eval.regime import regime_mix_gap, summarize_by_regime


def test_summarize_by_regime_and_gap():
    y = [10.0, 12.0, 40.0, 42.0, 80.0, 82.0]
    pred = [11.0, 11.0, 38.0, 44.0, 70.0, 90.0]
    regimes = [0, 0, 1, 1, 2, 2]
    summary = summarize_by_regime(y, pred, regimes)
    assert set(summary) == {"0", "1", "2"}
    assert summary["1"]["sample_count"] == 2
    assert summary["1"]["rmse"] > 0

    gap = regime_mix_gap(summary, summary)
    assert gap["status"] == "comparable"
    assert gap["chi2_pvalue"] is not None


def test_manifest_contains_regime_gap_fields():
    manifest = json.loads((get_root_dir() / "artifacts" / "run_manifest.json").read_text(encoding="utf-8"))
    assert "regime_mix_gap" in manifest
    assert "flagged_subsets" in manifest["regime_mix_gap"]
    for sub in ["FD001", "FD002", "FD003", "FD004"]:
        info = manifest["subsets"][sub]
        assert "cv_by_regime" in info
        assert "test_by_regime" in info
        assert "regime_mix_gap" in info
        assert "chi2_pvalue" in info["regime_mix_gap"]
        assert info["regime_mix_gap"]["status"] in {"comparable", "material_gap"}

"""Operating-regime diagnostics shared by CV, test evaluation, and the manifest.

The C-MAPSS subsets contain one operating regime (FD001/FD003) or six regimes
(FD002/FD004).  Comparing the regime mix seen during cross-validation with the
official test mix catches silent distribution gaps that aggregate RMSE hides.
"""

from typing import Any, Dict, Iterable, Mapping

import numpy as np
from scipy import stats


def _regime_key(value: Any) -> str:
    return str(int(value))


def summarize_by_regime(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    regimes: Iterable[Any],
) -> Dict[str, Dict[str, Any]]:
    """Returns RMSE/sample count/share for each operating regime."""
    truth = np.asarray(list(y_true), dtype=float)
    pred = np.asarray(list(y_pred), dtype=float)
    regime_arr = np.asarray([int(r) for r in regimes], dtype=int)
    if not (len(truth) == len(pred) == len(regime_arr)):
        raise ValueError("y_true, y_pred, and regimes must have equal length")

    total = len(regime_arr)
    summary: Dict[str, Dict[str, Any]] = {}
    for regime in sorted(np.unique(regime_arr).tolist()):
        mask = regime_arr == regime
        diff = pred[mask] - truth[mask]
        summary[_regime_key(regime)] = {
            "sample_count": int(np.sum(mask)),
            "share": round(float(np.sum(mask) / total), 4) if total else 0.0,
            "rmse": round(float(np.sqrt(np.mean(diff ** 2))), 4),
            "mae": round(float(np.mean(np.abs(diff))), 4),
        }
    return summary


def regime_counts(summary: Mapping[str, Mapping[str, Any]]) -> Dict[str, int]:
    """Extracts integer sample counts from a by-regime summary."""
    return {str(regime): int(values["sample_count"]) for regime, values in summary.items()}


def regime_mix_gap(
    cv_summary: Mapping[str, Mapping[str, Any]],
    test_summary: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Compares CV and test regime mixes with chi-square and KS tests.

    Chi-square is the primary categorical test.  The KS statistic is reported on
    the expanded regime-label samples as an ordinal complementary check.  When
    either split contains a single regime, both tests are undefined and the
    result is marked comparable with an explicit note.
    """
    cv_counts = regime_counts(cv_summary)
    test_counts = regime_counts(test_summary)
    labels = sorted(set(cv_counts) | set(test_counts), key=lambda x: int(x))
    cv_vector = np.asarray([cv_counts.get(label) or 0 for label in labels], dtype=int)
    test_vector = np.asarray([test_counts.get(label) or 0 for label in labels], dtype=int)
    total_cv = int(cv_vector.sum())
    total_test = int(test_vector.sum())

    cv_shares = {
        label: round(float(count / total_cv), 4) if total_cv else 0.0
        for label, count in zip(labels, cv_vector)
    }
    test_shares = {
        label: round(float(count / total_test), 4) if total_test else 0.0
        for label, count in zip(labels, test_vector)
    }
    max_share_gap = max(
        (abs(cv_shares[label] - test_shares[label]) for label in labels),
        default=0.0,
    )

    result: Dict[str, Any] = {
        "cv_regime_counts": cv_counts,
        "test_regime_counts": test_counts,
        "cv_regime_shares": cv_shares,
        "test_regime_shares": test_shares,
        "max_share_gap": round(float(max_share_gap), 4),
        "chi2_statistic": None,
        "chi2_pvalue": None,
        "ks_statistic": None,
        "ks_pvalue": None,
        "low_expected_counts": False,
        "gap_flag": False,
        "status": "comparable",
    }

    if len(labels) <= 1 or total_cv == 0 or total_test == 0:
        result["note"] = "Single-regime split; chi-square and KS tests are undefined."
        return result

    table = np.vstack([cv_vector, test_vector])
    chi2, chi2_p, _, expected = stats.chi2_contingency(table, correction=False)
    cv_labels = np.repeat(np.asarray([int(x) for x in labels]), cv_vector)
    test_labels = np.repeat(np.asarray([int(x) for x in labels]), test_vector)
    ks_stat, ks_p = stats.ks_2samp(cv_labels, test_labels)

    low_expected = bool(np.any(expected < 5))
    result.update({
        "chi2_statistic": round(float(chi2), 4),
        "chi2_pvalue": round(float(chi2_p), 6),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_pvalue": round(float(ks_p), 6),
        "low_expected_counts": low_expected,
        "gap_flag": bool(chi2_p < 0.05 or ks_p < 0.05 or max_share_gap > 0.10),
    })
    result["status"] = "material_gap" if result["gap_flag"] else "comparable"
    if low_expected:
        result["note"] = "At least one chi-square expected cell is below 5; interpret with care."
    return result

"""CV must censor validation targets with the same random-truncation protocol as the test set.

The canonical protocol is configured in configs/pipeline.yaml and enforced by
load_cv_protocol(). A last-cycle validation frame is degenerate on train engines
(every engine has RUL 0 at its final cycle), so it must be rejected both by the
protocol loader and by the distribution checks below.
"""
import numpy as np
import pandas as pd
import pytest
from scipy.stats import ks_2samp

from src.common.utils import get_root_dir
from src.eval.protocol import cap_rul, eval_frames, get_protocol_id, DEFAULT_SEED
from src.ingest.loader import CMAPSSLoader
from src.models.model_service import load_cv_protocol

ROOT_DIR = get_root_dir()
SUBSETS = ["FD001", "FD002", "FD003", "FD004"]


def _train_rul_frames(subset):
    protocol = load_cv_protocol()
    train_df = pd.read_parquet(ROOT_DIR / "data" / "processed" / subset / "train.parquet")
    frames = eval_frames(
        train_df,
        mode=protocol["mode"],
        seed=DEFAULT_SEED,
        n_draws=protocol["n_draws"],
        min_cycles=protocol["min_cycles"],
    )
    assert len(frames) == protocol["n_draws"], "eval_frames must produce one frame per draw"
    val_rul = cap_rul(np.concatenate([f["rul_raw"].to_numpy(dtype=float) for f in frames]))
    return protocol, train_df, val_rul


def _test_rul(subset):
    _, _, rul_truth = CMAPSSLoader(subset=subset).load_raw_data()
    return cap_rul(rul_truth)


@pytest.mark.parametrize("subset", SUBSETS)
def test_censored_cv_rul_matches_test_distribution(subset):
    protocol, _, val_rul = _train_rul_frames(subset)
    test_rul = _test_rul(subset)

    ks_stat = float(ks_2samp(val_rul, test_rul).statistic)
    mean_rel_gap = abs(float(val_rul.mean()) - float(test_rul.mean())) / float(test_rul.mean())

    # Tolerance is derived from the data and recorded in configs/pipeline.yaml.
    assert ks_stat < protocol["ks_threshold"], (
        f"{subset}: censored CV RUL KS statistic {ks_stat:.3f} exceeds test-derived "
        f"threshold {protocol['ks_threshold']:.3f}"
    )
    assert mean_rel_gap < protocol["mean_rel_gap"], (
        f"{subset}: censored CV mean RUL {val_rul.mean():.1f} differs from test mean "
        f"{test_rul.mean():.1f} by {mean_rel_gap:.1%}, exceeding "
        f"{protocol['mean_rel_gap']:.1%}"
    )


@pytest.mark.parametrize("subset", SUBSETS)
def test_last_cycle_validation_is_rejected_by_the_distribution_checks(subset):
    """Regression guard: the old mode='last' protocol must fail the same checks."""
    protocol, train_df, _ = _train_rul_frames(subset)
    last_frame = eval_frames(train_df, mode="last", seed=DEFAULT_SEED, n_draws=1)[0]
    last_rul = cap_rul(last_frame["rul_raw"].to_numpy(dtype=float))
    test_rul = _test_rul(subset)

    ks_stat = float(ks_2samp(last_rul, test_rul).statistic)
    mean_rel_gap = abs(float(last_rul.mean()) - float(test_rul.mean())) / float(test_rul.mean())

    assert ks_stat >= protocol["ks_threshold"] or mean_rel_gap >= protocol["mean_rel_gap"], (
        f"{subset}: mode='last' unexpectedly matched the test RUL distribution"
    )


def test_protocol_id_hashes_the_mode_and_draws_actually_used():
    protocol = load_cv_protocol()
    cv_id = get_protocol_id(
        mode=protocol["mode"], seed=DEFAULT_SEED, n_draws=protocol["n_draws"]
    )
    last_id = get_protocol_id(mode="last", seed=DEFAULT_SEED, n_draws=1)
    assert cv_id != last_id, "CV must not share the last-cycle test protocol id"
    assert protocol["n_draws"] >= 5

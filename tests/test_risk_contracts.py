import numpy as np
import pandas as pd
import pytest

from src.schema import validate_features
from src.scorecard import pd_to_score, score_to_pd
from src.stress_test import compute_expected_loss, stress_pd, run_all_scenarios
from src.model_validation import split_cohorts, evaluation_report


def test_score_monotonic_and_roundtrip():
    probabilities = np.linspace(.01, .4, 100)
    assert (np.diff(pd_to_score(probabilities)) < 0).all()
    np.testing.assert_allclose(score_to_pd(pd_to_score(probabilities)), probabilities)


@pytest.mark.parametrize('bad', [-.1, 1.1, np.nan, np.inf])
def test_invalid_probability_rejected(bad):
    with pytest.raises(ValueError):
        pd_to_score(bad)
    with pytest.raises(ValueError):
        stress_pd([bad], .3)


def test_loss_and_stress():
    p = np.array([0., .1, .5, 1.])
    np.testing.assert_array_equal(stress_pd(p, 0), p)
    assert (stress_pd(p[1:-1], .3) >= p[1:-1]).all()
    np.testing.assert_allclose(compute_expected_loss([.1], [.5], [100]), [5])
    with pytest.raises(ValueError):
        compute_expected_loss([.1], [.5], [-100])
    with pytest.raises(ValueError):
        compute_expected_loss([.1, .2], [.5], [100])


def test_zero_exposure_summary():
    _, summary = run_all_scenarios(pd.DataFrame({'PD_CALIBRATED': [.1], 'LGD': [.5], 'EAD': [0.]}))
    assert (summary.EL_shift_pct == 0).all()


def test_feature_contract():
    frame = validate_features({'b': np.nan, 'a': 2}, ['a', 'b'])
    assert list(frame.columns) == ['a', 'b']
    assert pd.isna(frame.b.iloc[0])
    for invalid in ({'a': 2}, {'a': 'two', 'b': 1}, {'a': np.inf, 'b': 1}, {'a': 1, 'b': 2, 'c': 3}):
        with pytest.raises(ValueError):
            validate_features(invalid, ['a', 'b'])


def test_cohorts_disjoint_reproducible():
    frame = pd.DataFrame({'SK_ID_CURR': np.arange(1000), 'TARGET': np.tile([0, 0, 0, 1], 250)})
    cohorts = split_cohorts(frame)
    assert [len(c) for c in cohorts] == [600, 200, 200]
    ids = [set(c.SK_ID_CURR) for c in cohorts]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for a, b in zip(cohorts, split_cohorts(frame)):
        pd.testing.assert_frame_equal(a, b)
    with pytest.raises(ValueError):
        split_cohorts(pd.concat([frame, frame.iloc[:1]]))


def test_evaluation_has_independent_metrics_and_intervals():
    y = np.tile([0, 1], 50)
    raw = np.where(y, .8, .2)
    report = evaluation_report(y, raw, raw, bootstrap_samples=10)
    assert report['raw']['roc_auc'] == 1
    assert report['raw']['brier_score'] == pytest.approx(.04)
    assert report['raw'] == report['calibrated']
    assert 'brier_score' in report['confidence_intervals_95']['calibrated']

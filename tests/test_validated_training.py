import numpy as np
import pandas as pd
import pytest

pytest.importorskip('lightgbm')
from src import config
from src.feature_engineering import one_hot
from src.validated_training import train_ensemble
from src.model_validation import split_cohorts


def test_category_vocabulary_excludes_holdout():
    frame = pd.DataFrame({'category': ['known', 'known', 'unseen']})
    encoded, _ = one_hot(frame, reference_mask=[True, True, False])
    assert 'category_unseen' not in encoded
    assert encoded.category_nan.iloc[2]


def test_ensemble_never_fits_calibration_or_test_labels(monkeypatch):
    monkeypatch.setattr(config, 'N_FOLDS', 2)
    rng = np.random.default_rng(42)
    x = rng.normal(size=500)
    frame = pd.DataFrame({'SK_ID_CURR': np.arange(500), 'TARGET': (x > 0).astype(int), 'x': x})
    development, calibration, test = split_cohorts(frame)
    models, cal_p, test_p, folds = train_ensemble(development, calibration, test, ['x'], rounds=5, threads=1)
    # Changing held-out outcomes must not change any model/prediction.
    calibration.TARGET = 1 - calibration.TARGET
    test.TARGET = 1 - test.TARGET
    other, other_cal, other_test, _ = train_ensemble(development, calibration, test, ['x'], rounds=5, threads=1)
    np.testing.assert_array_equal(cal_p, other_cal)
    np.testing.assert_array_equal(test_p, other_test)
    assert [m.model_to_string() for m in models] == [m.model_to_string() for m in other]
    assert len(folds) == 2

import numpy as np
import pandas as pd
import pytest
import lightgbm as lgb
from src.explainability import explain_fold, driver_table
from src.artifact_registry import load_validated_run


@pytest.fixture
def example():
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(200, 2)), columns=['income', 'debt'])
    y = (X.debt > X.income).astype(int)
    model = lgb.train({'objective': 'binary', 'verbose': -1, 'num_threads': 1}, lgb.Dataset(X, label=y), num_boost_round=10)
    return {'feature_names': list(X), 'fold_models': [model]}, X.iloc[:5]


def test_shap_reconstructs_fold_and_matches_input_order(example):
    bundle, X = example
    result = explain_fold(bundle, X[['debt', 'income']])
    np.testing.assert_allclose(result['base_values'] + result['values'].sum(axis=1), result['raw_margin'], atol=1e-7)
    np.testing.assert_allclose(1 / (1 + np.exp(-result['raw_margin'])), result['fold_probability'])
    table = driver_table(X.iloc[0], result['values'][0])
    assert set(table.Feature) == set(X)
    for row in table.itertuples(index=False):
        assert row[1] == X.iloc[0][row[0]]


def test_shap_rejects_wrong_schema_and_fold(example):
    bundle, X = example
    with pytest.raises(ValueError):
        explain_fold(bundle, X, fold=4)
    bundle['feature_names'] = list(reversed(bundle['feature_names']))
    with pytest.raises(ValueError, match='feature order'):
        explain_fold(bundle, X)


def test_artifact_tampering_rejected_before_unpickling(tmp_path):
    import json
    (tmp_path / 'manifest.json').write_text(json.dumps({'artifact_sha256': {'inference_bundle.pkl': 'invalid'}}))
    (tmp_path / 'inference_bundle.pkl').write_bytes(b'not a pickle')
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        load_validated_run(tmp_path)

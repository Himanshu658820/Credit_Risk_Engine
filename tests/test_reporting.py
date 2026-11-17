import numpy as np
import pandas as pd
import pytest
from src import config
from src.reporting import loss_assessment, borrower_assessment, model_card
from src.model_validation import evaluation_report


def features(revolving=False):
    return pd.Series({'SK_ID_CURR': 42, 'AMT_CREDIT': 1000., 'AMT_GOODS_PRICE': 1000.,
                      'NAME_CONTRACT_TYPE_Cash_loans': int(not revolving),
                      'NAME_CONTRACT_TYPE_Revolving_loans': int(revolving)})


def test_loss_scenarios_reconcile():
    for revolving in (False, True):
        result = loss_assessment(features(revolving), .1)
        expected_ead = 1000 * (.5 + .5 * config.CCF_REVOLVING) if revolving else 1000
        assert result['EAD'] == expected_ead
        for scenario in result['scenarios']:
            assert scenario['EL'] == pytest.approx(scenario['PD'] * scenario['LGD'] * scenario['EAD'])
        assert result['scenarios'][-1]['EL'] >= result['scenarios'][0]['EL']


def test_missing_contract_or_exposure_not_invented():
    f = features()
    f['NAME_CONTRACT_TYPE_Cash_loans'] = 0
    assert loss_assessment(f, .1) is None
    f = features()
    f['AMT_CREDIT'] = np.nan
    assert loss_assessment(f, .1) is None
    with pytest.raises(ValueError):
        loss_assessment(features(), 1.5)


def test_report_rejects_mismatched_borrower():
    with pytest.raises(ValueError, match='borrower IDs'):
        borrower_assessment('run', {}, {'SK_ID_CURR': 9}, features(), None, None, 0)


def test_card_preserves_metrics_and_caveats():
    y = np.tile([0, 1], 30)
    report = evaluation_report(y, np.where(y, .8, .2), np.where(y, .7, .3), bootstrap_samples=5)
    report.update(cohort_sizes={'development': 180, 'calibration': 60, 'test': 60}, split_seed=42, limitations=['Not temporal.'])
    manifest = {'created_utc': '2026-11-17', 'git_commit': 'abc', 'artifact_sha256': {'inference_bundle.pkl': 'hash'},
                'feature_schema_sha256': 'schema', 'packages': {'lightgbm': 'test'}}
    text = model_card('test', report, manifest)
    assert '0.040000' in text and '0.090000' in text
    assert 'Calibration did not improve' in text
    assert 'not a bureau score' in text
    assert 'Not temporal.' in text

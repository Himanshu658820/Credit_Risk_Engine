"""Deterministic, evidence-based reports for verified held-out examples."""
from datetime import datetime, timezone
import pandas as pd
from . import config
from .lgd_ead import build_lgd_ead_table
from .scorecard import pd_to_score
from .schema import bounded_values
from .stress_test import run_scenario


def _cell(value):
    if pd.isna(value):
        return 'Missing'
    return str(value).replace('|', '\\|').replace('\n', ' ').replace('\r', ' ')


def _table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(_cell(v) for v in row) + ' |' for row in rows])


def loss_assessment(features, calibrated_pd):
    """Use verified engineered fields, never join the legacy portfolio's PD."""
    required = ['AMT_CREDIT', 'AMT_GOODS_PRICE', 'NAME_CONTRACT_TYPE_Cash_loans',
                'NAME_CONTRACT_TYPE_Revolving_loans']
    if any(name not in features for name in required):
        return None
    cash, revolving = features['NAME_CONTRACT_TYPE_Cash_loans'], features['NAME_CONTRACT_TYPE_Revolving_loans']
    if pd.isna(cash) or pd.isna(revolving) or cash not in (0, 1) or revolving not in (0, 1) or cash + revolving != 1:
        return None
    if pd.isna(features['AMT_CREDIT']):
        return None
    credit = float(bounded_values(features['AMT_CREDIT'], 'Credit amount'))
    p = float(bounded_values(calibrated_pd, 'PD', maximum=1))
    raw = pd.DataFrame([{'SK_ID_CURR': int(features['SK_ID_CURR']),
                         '_RAW_AMT_CREDIT': credit,
                         '_RAW_AMT_GOODS_PRICE': features['AMT_GOODS_PRICE'],
                         '_RAW_NAME_CONTRACT_TYPE': 'Revolving loans' if revolving else 'Cash loans'}])
    risk = build_lgd_ead_table(raw).iloc[0]
    scenarios = []
    for name, params in config.STRESS_SCENARIOS.items():
        pd_s, lgd_s, el_s = run_scenario([p], [risk.LGD], [risk.EAD], name)
        scenarios.append({'scenario': name, 'PD': float(pd_s[0]), 'LGD': float(lgd_s[0]),
                          'EAD': float(risk.EAD), 'EL': float(el_s[0]), **params})
    return {'security_proxy': risk.SECURITY_CLASS, 'LGD': float(risk.LGD), 'EAD': float(risk.EAD),
            'scenarios': scenarios}


def assumptions_text():
    return (f'LGD proxies: secured {config.LGD_SECURED:.0%}, unsecured {config.LGD_UNSECURED:.0%}, '
            f'revolving {config.LGD_REVOLVING:.0%}. These are project assumptions, not fitted recovery estimates. '
            'The secured label uses a goods-price/credit ratio between 0.8 and 1.2; it is not evidence of legal collateral. '
            f'Revolving EAD assumes 50% drawn plus a {config.CCF_REVOLVING:.0%} conversion factor on the undrawn 50%; '
            'other EAD equals application credit. EAD stays constant under stress. '
            'Monetary amounts are in dataset units, with no currency conversion. '
            'Stress shifts are hypothetical and are not calibrated to macroeconomic outcomes.')


def borrower_assessment(run_name, manifest, prediction, features, drivers, explanation, index):
    borrower = int(prediction['SK_ID_CURR'])
    if borrower != int(features['SK_ID_CURR']):
        raise ValueError('Prediction and feature borrower IDs differ')
    raw = float(bounded_values(prediction['PD_RAW'], 'Raw PD', maximum=1))
    calibrated = float(bounded_values(prediction['PD_CALIBRATED'], 'Calibrated PD', maximum=1))
    loss = loss_assessment(features, calibrated)
    sections = ['# Borrower assessment',
                'Research demonstration for a saved held-out borrower. This report is not a lending decision.',
                f"Borrower dataset ID: {borrower}\n\nRun: {_cell(run_name)}\n\nGenerated UTC: {datetime.now(timezone.utc).isoformat()}\n\nModel trained UTC: {manifest['created_utc']}\n\nBundle SHA256: {manifest['artifact_sha256']['inference_bundle.pkl']}",
                '## Model estimates',
                _table(['Measure', 'Value'], [['Ensemble raw PD', f'{raw:.4%}'], ['Ensemble calibrated PD', f'{calibrated:.4%}'],
                       ['Custom model score (300–850)', f'{float(pd_to_score(calibrated)):.0f}']]),
                'The score is derived from calibrated PD; it is not an official bureau score. No approval threshold or risk grade has been validated.',
                '## Fold explanation',
                f"Fold {explanation['fold']} only; raw log-odds. Fold probability: {explanation['fold_probability'][index]:.4%}. "
                'These contributions do not decompose the ensemble probability or calibrated PD. They reflect associations, not causes.',
                _table(['Feature', 'Observed value', 'SHAP contribution', 'Direction'],
                       [[r[0], r[1], f'{r[2]:+.6f}', r[3]] for r in drivers.head(5).itertuples(index=False, name=None)]),
                f"Expected margin {explanation['base_values'][index]:.6f} + all feature contributions = {explanation['raw_margin'][index]:.6f}. Only the five largest drivers are listed above.",
                '## Assumption based loss scenarios']
    if loss is None:
        sections.append('LGD, EAD and expected loss are unavailable because required exposure or contract inputs are missing or ambiguous. No values have been invented.')
    else:
        sections += [f"Security proxy: {loss['security_proxy']}. Baseline LGD: {loss['LGD']:.0%}. EAD: {loss['EAD']:,.2f} dataset units.",
                     _table(['Scenario', 'PD', 'LGD', 'EAD', 'Expected loss', 'PD logit shift', 'LGD multiplier'],
                            [[s['scenario'], f"{s['PD']:.4%}", f"{s['LGD']:.2%}", f"{s['EAD']:,.2f}", f"{s['EL']:,.2f}", s['pd_logit_shift'], s['lgd_multiplier']] for s in loss['scenarios']]),
                     'Expected loss = calibrated or stressed PD × assumed LGD × assumed EAD. Scenario amounts use dataset units.']
    sections += [assumptions_text(), '## Limits and human review',
                 'Review input completeness and contract assumptions before interpreting the estimates. Check the model card for held-out performance; a probability is not a certainty about this borrower. '
                 'The dataset is historical, with no external or temporal validation in this run. Fairness and subgroup validation are not established. '
                 'Confirm the applicable prediction horizon before any operational interpretation. This demo is not a production underwriting or regulatory-compliance system.']
    return '\n\n'.join(sections) + '\n'


def model_card(run_name, report, manifest):
    metric_rows = [[key, f"{report['raw'][key]:.6f}", f"{report['calibrated'][key]:.6f}"] for key in report['raw']]
    worse = all(report['calibrated'][key] >= report['raw'][key] for key in ['brier_score', 'log_loss'])
    sections = ['# Credit Risk Engine model card',
                '## Purpose and intended use',
                'A portfolio research and interview demonstration of default prediction, calibration and model explanation using Home Credit Default Risk. Not intended for automated lending decisions, production underwriting or regulatory compliance.',
                '## Model identity',
                f"Run: {_cell(run_name)}\n\nTrained UTC: {manifest['created_utc']}\n\nTraining source commit: {manifest['git_commit']}\n\nTraining tree had local changes: {manifest.get('working_tree_dirty', False)}\n\nBundle SHA256: {manifest['artifact_sha256']['inference_bundle.pkl']}\n\nFeature schema SHA256: {manifest['feature_schema_sha256']}",
                'The training source hashes and dataset file hashes are recorded in the run manifest. The commit alone does not identify local training changes.',
                '## Data and evaluation design',
                'Five LightGBM folds are fitted only within the development cohort, with early stopping on development validation rows. Their mean probability is fitted to an isotonic calibrator using a separate calibration cohort. The test cohort is used for final reporting, not threshold, model or risk-band selection.',
                _table(['Cohort', 'Borrowers'], [[k, f'{v:,}'] for k, v in report['cohort_sizes'].items()]),
                f"Split seed: {report['split_seed']}. Test defaults: {report['defaults']:,} ({report['default_rate']:.2%}). "
                'This is a random borrower split within a dataset used previously for model development, not new external evidence. The target is the supplied binary TARGET; a standardized forward prediction horizon has not been established by this implementation.',
                '## Held-out performance', _table(['Metric', 'Raw ensemble', 'Calibrated ensemble'], metric_rows),
                'Higher ROC-AUC and average precision are better; lower Brier score and log loss are better. Average precision is non-interpolated, not trapezoidal PR area.',
                ('Calibration did not improve Brier score or log loss on this test cohort. Both outputs are retained; no model is selected using this test comparison.' if worse else 'Calibration effects differ by metric; both raw and calibrated results are retained.'),
                '## Uncertainty',
                _table(['Model', 'Metric', '95% lower', '95% upper'], [[kind, metric, f'{bounds[0]:.6f}', f'{bounds[1]:.6f}'] for kind, intervals in report['confidence_intervals_95'].items() for metric, bounds in intervals.items()]),
                f"Bootstrap replicates: {report['bootstrap']['replicates']}. {report['bootstrap']['scope']}",
                '## Explanations and inputs',
                'Native Tree SHAP explains one selected fold in raw log-odds. It does not decompose the ensemble probability or calibrated PD and does not establish causation. The dashboard uses 100 saved held-out examples; their importance chart is not a portfolio-wide estimate. Inputs must be fully engineered numeric features in the recorded schema. Missing history values are supported; infinite inputs are rejected.',
                '## Loss assumptions', assumptions_text(),
                'The reporting layer uses the current project loss assumptions, separately from the trained PD bundle. Each borrower assessment records its scenario parameters. The 300–850 score is a custom log-odds mapping, not a bureau score.',
                '## Limitations and governance',
                '\n'.join('- ' + item for item in report['limitations']),
                '- Subgroup fairness, temporal drift and live outcome monitoring have not been validated.\n- No approval policy or risk-band thresholds have been established.\n- Artifact hashes detect mismatches; only trusted local pickle files may be loaded.\n- Any material model change requires a new validation plan; repeatedly optimizing against this test cohort would compromise its independence.',
                '## Reproduction',
                'Run `python -m src.validated_training` with the Home Credit CSVs in `DATA_SET/`; then `streamlit run dashboard/app.py`. A new run directory is created. Run `python -m pytest tests -q` for numerical and contract tests.',
                _table(['Training package', 'Version'], list(manifest['packages'].items()))]
    return '\n\n'.join(sections) + '\n'

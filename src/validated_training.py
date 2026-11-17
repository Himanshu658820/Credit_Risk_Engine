"""Train a development-only ensemble, calibrate separately, evaluate once."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import subprocess
from datetime import datetime, timezone

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from . import config
from .calibration import fit_calibrator, apply_calibration
from .feature_engineering import build_features, sanitize_columns
from .model_validation import split_cohorts, evaluation_report
from .schema import validate_features


def train_ensemble(development, calibration, test, feature_names, rounds=None, threads=4):
    """Only development labels reach LightGBM, including early stopping."""
    X = validate_features(development[feature_names], feature_names)
    y = development.TARGET
    predict_frames = [validate_features(frame[feature_names], feature_names) for frame in (calibration, test)]
    predictions = [np.zeros(len(frame)) for frame in predict_frames]
    models, fold_metrics = [], []
    splitter = StratifiedKFold(config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    params = dict(config.LGB_PARAMS, num_threads=threads, deterministic=True, force_col_wise=True)
    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y), 1):
        print(f'Training development fold {fold}/{config.N_FOLDS}', flush=True)
        model = lgb.train(params, lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx]),
                          num_boost_round=rounds or config.NUM_BOOST_ROUND,
                          valid_sets=[lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])],
                          callbacks=[lgb.early_stopping(config.EARLY_STOPPING_ROUNDS), lgb.log_evaluation(200)])
        auc = roc_auc_score(y.iloc[val_idx], model.predict(X.iloc[val_idx]))
        fold_metrics.append({'fold': fold, 'auc': float(auc), 'best_iteration': model.best_iteration})
        model.free_dataset()
        models.append(model)
        for output, frame in zip(predictions, predict_frames):
            output += model.predict(frame, num_iteration=model.best_iteration) / config.N_FOLDS
    return models, predictions[0], predictions[1], fold_metrics


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def run(output, threads=4):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    applications = pd.read_csv(Path(config.DATA_DIR) / 'application_train.csv', usecols=['SK_ID_CURR', 'TARGET'])
    cohorts = split_cohorts(applications, config.RANDOM_SEED)
    names = ['development', 'calibration', 'test']
    assignments = pd.concat([frame.assign(cohort=name) for frame, name in zip(cohorts, names)])
    assignments.to_csv(output / 'split_assignments.csv', index=False)
    print('Cohorts:', dict(zip(names, map(len, cohorts))), flush=True)
    print('Building features with development-only category vocabularies...', flush=True)
    features = sanitize_columns(build_features(set(cohorts[0].SK_ID_CURR)))
    feature_names = [c for c in features if c not in ('SK_ID_CURR', 'TARGET', 'is_train') and not c.startswith('_RAW_')]
    # Fixed preprocessing policy: undefined ratios are missing, never infinity.
    features[feature_names] = features[feature_names].replace([np.inf, -np.inf], np.nan)
    indexed = features.set_index('SK_ID_CURR', drop=False)
    development, calibration, test = [indexed.loc[frame.SK_ID_CURR].copy() for frame in cohorts]
    del features, indexed
    gc.collect()
    models, calibration_raw, test_raw, fold_metrics = train_ensemble(
        development, calibration, test, feature_names, threads=threads)
    calibrator = fit_calibrator(pd.DataFrame({'TARGET': calibration.TARGET.to_numpy(), 'PD_RAW': calibration_raw}))
    test_calibrated = apply_calibration(test_raw, calibrator)
    bundle = {'fold_models': models, 'feature_names': feature_names, 'calibrator': calibrator,
              'split_seed': config.RANDOM_SEED, 'format_version': 1,
              'preprocessing': 'Engineered numeric features; undefined ratios become NaN; development-only category vocabulary'}
    with open(output / 'inference_bundle.pkl', 'wb') as target:
        pickle.dump(bundle, target)
    predictions = test[['SK_ID_CURR', 'TARGET']].copy()
    predictions['PD_RAW'] = test_raw
    predictions['PD_CALIBRATED'] = test_calibrated
    predictions.to_csv(output / 'test_predictions.csv', index=False)
    test[['SK_ID_CURR'] + feature_names].head(100).to_pickle(output / 'example_features.pkl')
    print('Evaluating untouched test predictions and bootstrap intervals...', flush=True)
    report = evaluation_report(test.TARGET.to_numpy(), test_raw, test_calibrated)
    report.update(split_seed=config.RANDOM_SEED, cohort_sizes=dict(zip(names, map(len, cohorts))),
                  development_folds=fold_metrics,
                  limitations=['Random borrower split, not temporal validation.',
                               'Calibration is not guaranteed to improve every metric.',
                               'Previously used dataset and model design: holdout is independent of this run, not a new external dataset.',
                               'LGD/EAD remain assumptions; not evaluated by this PD report.'])
    (output / 'metrics.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for kind, curve in report['reliability'].items():
        ax.plot(curve['mean_prediction'], curve['observed_rate'], marker='o', label=kind)
    ax.plot([0, 1], [0, 1], '--', color='grey')
    ax.set(xlabel='Mean predicted default probability', ylabel='Observed default rate', title='Untouched test cohort reliability')
    ax.legend()
    fig.savefig(output / 'calibration.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('Recording dataset and artifact fingerprints...', flush=True)
    manifest = {'created_utc': datetime.now(timezone.utc).isoformat(),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'working_tree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
                'source_sha256': {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')},
                'dataset_sha256': {p.name: sha256(p) for p in Path(config.DATA_DIR).glob('*.csv')},
                'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'lightgbm', 'scikit-learn']},
                'feature_schema_sha256': hashlib.sha256(json.dumps(feature_names).encode()).hexdigest(),
                'threads': threads, 'seed': config.RANDOM_SEED,
                'artifact_sha256': {p.name: sha256(p) for p in output.iterdir() if p.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({'raw': report['raw'], 'calibrated': report['calibrated']}, indent=2), flush=True)
    print(f'Completed: {output.resolve()}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default=str(Path(config.ARTIFACTS_DIR) / 'validated' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')))
    parser.add_argument('--threads', type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads must be positive')
    run(args.output, args.threads)

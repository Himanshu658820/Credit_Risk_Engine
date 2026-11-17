"""Read locally trusted validated runs and reject mismatched artifacts."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .schema import validate_features


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_validated_run(directory):
    root = Path(directory)
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    required = ['inference_bundle.pkl', 'example_features.pkl', 'metrics.json', 'test_predictions.csv']
    for name in required:
        expected = manifest['artifact_sha256'].get(name)
        if expected is None or fingerprint(root / name) != expected:
            raise ValueError(f'Artifact fingerprint mismatch: {name}')
    # Hashes detect accidental inconsistency; they do not authenticate untrusted pickles.
    with (root / 'inference_bundle.pkl').open('rb') as source:
        bundle = pickle.load(source)
    if bundle.get('format_version') != 1:
        raise ValueError('Unsupported bundle version')
    schema_hash = hashlib.sha256(json.dumps(bundle['feature_names']).encode()).hexdigest()
    if schema_hash != manifest['feature_schema_sha256']:
        raise ValueError('Feature schema mismatch')
    samples = pd.read_pickle(root / 'example_features.pkl').reset_index(drop=True)
    X = validate_features(samples.drop(columns='SK_ID_CURR'), bundle['feature_names'])
    predictions = pd.read_csv(root / 'test_predictions.csv').set_index('SK_ID_CURR')
    if samples.SK_ID_CURR.duplicated().any() or predictions.index.duplicated().any():
        raise ValueError('Duplicate borrower IDs in artifacts')
    saved = predictions.loc[samples.SK_ID_CURR]
    raw = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in bundle['fold_models']], axis=0)
    calibrated = bundle['calibrator'].predict(raw)
    if not np.allclose(raw, saved.PD_RAW, atol=1e-10, rtol=0) or not np.allclose(calibrated, saved.PD_CALIBRATED, atol=1e-10, rtol=0):
        raise ValueError('Saved examples do not reproduce the evaluated predictions')
    return bundle, samples, saved.reset_index(), json.loads((root / 'metrics.json').read_text()), manifest

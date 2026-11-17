"""Native LightGBM Tree SHAP for an explicitly identified fold's raw margin."""
import numpy as np
import pandas as pd
from .schema import validate_features


def explain_fold(bundle, features, fold=0):
    names = bundle['feature_names']
    X = validate_features(features, names)
    if not 0 <= fold < len(bundle['fold_models']):
        raise ValueError('Invalid fold index')
    model = bundle['fold_models'][fold]
    if model.feature_name() != names:
        raise ValueError('Model feature order does not match the bundle')
    contributions = np.asarray(model.predict(X, pred_contrib=True, num_iteration=model.best_iteration))
    raw = np.asarray(model.predict(X, raw_score=True, num_iteration=model.best_iteration))
    if contributions.shape != (len(X), len(names) + 1) or not np.isfinite(contributions).all():
        raise ValueError('Invalid explanation shape or values')
    if not np.allclose(contributions.sum(axis=1), raw, atol=1e-7, rtol=1e-7):
        raise ValueError('SHAP contributions do not reconstruct the explained model output')
    return {'values': contributions[:, :-1], 'base_values': contributions[:, -1],
            'raw_margin': raw, 'fold': fold + 1,
            'fold_probability': model.predict(X, num_iteration=model.best_iteration)}


def driver_table(features, values):
    table = pd.DataFrame({'Feature': features.index, 'Observed value': features.to_numpy(),
                          'SHAP contribution': values})
    table['Direction'] = np.where(table['SHAP contribution'] > 0, 'Raises fold risk',
                                  np.where(table['SHAP contribution'] < 0, 'Lowers fold risk', 'No contribution'))
    return table.iloc[np.argsort(-np.abs(values))].reset_index(drop=True)

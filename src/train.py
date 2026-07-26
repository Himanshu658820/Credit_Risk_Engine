"""
Trains the LightGBM default-prediction model with 5-fold stratified CV,
runs diagnostics, and exports the artifacts every downstream module needs:

  artifacts/oof_predictions.csv  -> SK_ID_CURR, TARGET, PD_RAW, plus the
                                     _RAW_* columns needed by lgd_ead.py
  artifacts/lgb_model.pkl        -> {fold_models, feature_names}
  artifacts/submission.csv       -> Kaggle-format test predictions

Run directly:  python -m src.train   (from the project root)
"""

import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from sklearn.calibration import calibration_curve

from . import config
from .feature_engineering import build_features, sanitize_columns


def train_model(df):
    df = sanitize_columns(df)
    train_df = df[df["is_train"] == 1].copy()
    test_df = df[df["is_train"] == 0].copy()

    # _RAW_* columns carry raw strings/duplicated values for lgd_ead.py —
    # they must never be fed into LightGBM as training features.
    drop_cols = ["SK_ID_CURR", "TARGET", "is_train"]
    raw_cols = [c for c in train_df.columns if c.startswith("_RAW_")]
    feats = [c for c in train_df.columns if c not in drop_cols and c not in raw_cols]

    folds = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED
    )
    oof_preds = np.zeros(len(train_df))
    test_preds = np.zeros(len(test_df))
    fold_models = []
    fold_aucs = []
    feature_importance = pd.DataFrame()

    X = train_df[feats]
    y = train_df["TARGET"]
    X_test = test_df[feats]

    for fold, (trn_idx, val_idx) in enumerate(folds.split(X, y)):
        print(f"\n===== Fold {fold + 1}/{config.N_FOLDS} =====")
        X_trn, y_trn = X.iloc[trn_idx], y.iloc[trn_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

        dtrain = lgb.Dataset(X_trn, label=y_trn)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            config.LGB_PARAMS,
            dtrain,
            num_boost_round=config.NUM_BOOST_ROUND,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
            callbacks=[
                lgb.early_stopping(config.EARLY_STOPPING_ROUNDS, verbose=True),
                lgb.log_evaluation(200),
            ],
        )

        oof_preds[val_idx] = model.predict(X_val, num_iteration=model.best_iteration)
        test_preds += (
            model.predict(X_test, num_iteration=model.best_iteration) / config.N_FOLDS
        )

        fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
        fold_aucs.append(fold_auc)
        print(f"Fold {fold + 1} AUC: {fold_auc:.5f}")

        fi = pd.DataFrame(
            {
                "feature": feats,
                "importance": model.feature_importance(importance_type="gain"),
                "fold": fold + 1,
            }
        )
        feature_importance = pd.concat([feature_importance, fi], axis=0)
        fold_models.append(model)

    overall_auc = roc_auc_score(y, oof_preds)
    print(f"\n=== Overall OOF AUC: {overall_auc:.5f} ===")
    print(
        f"Fold AUCs: {[round(a, 5) for a in fold_aucs]} | "
        f"mean={np.mean(fold_aucs):.5f} std={np.std(fold_aucs):.5f}"
    )

    return {
        "oof_preds": oof_preds,
        "y_true": y.values,
        "test_preds": test_preds,
        "feature_importance": feature_importance,
        "fold_aucs": fold_aucs,
        "overall_auc": overall_auc,
        "models": fold_models,
        "feats": feats,
        "train_ids": train_df["SK_ID_CURR"].values,
        "test_ids": test_df["SK_ID_CURR"].values,
        "train_raw": (
            train_df[["SK_ID_CURR"] + raw_cols]
            if raw_cols
            else train_df[["SK_ID_CURR"]]
        ),
    }


def diagnose_model(results, top_n=30, save_path=None):
    """
    Produces a full diagnostic report:
      1. Fold-wise AUC bar chart (stability check)
      2. ROC curve (OOF)
      3. Feature importance (top N, averaged across folds)
      4. Prediction distribution by class
      5. Calibration curve (raw, pre-Isotonic)
      6. Confusion matrix at optimal (Youden's J) threshold
    """
    oof = results["oof_preds"]
    y_true = results["y_true"]
    fi = results["feature_importance"]
    fold_aucs = results["fold_aucs"]
    overall_auc = results["overall_auc"]

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    ax = axes[0, 0]
    ax.bar(range(1, len(fold_aucs) + 1), fold_aucs, color="steelblue")
    ax.axhline(
        overall_auc, color="red", linestyle="--", label=f"OOF AUC={overall_auc:.4f}"
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("AUC")
    ax.set_title("Fold-wise AUC stability")
    ax.legend()

    ax = axes[0, 1]
    fpr, tpr, thresholds = roc_curve(y_true, oof)
    ax.plot(fpr, tpr, label=f"AUC = {overall_auc:.4f}", color="darkorange")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (OOF)")
    ax.legend()

    ax = axes[0, 2]
    mean_fi = (
        fi.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .head(top_n)
    )
    sns.barplot(x=mean_fi.values, y=mean_fi.index, ax=ax, color="seagreen")
    ax.set_title(f"Top {top_n} Feature Importance (gain, avg over folds)")
    ax.set_xlabel("Importance")

    ax = axes[1, 0]
    sns.kdeplot(oof[y_true == 0], label="TARGET=0", ax=ax, fill=True)
    sns.kdeplot(oof[y_true == 1], label="TARGET=1", ax=ax, fill=True)
    ax.set_title("Predicted probability distribution by class")
    ax.set_xlabel("Predicted probability")
    ax.legend()

    ax = axes[1, 1]
    prob_true, prob_pred = calibration_curve(
        y_true, oof, n_bins=15, strategy="quantile"
    )
    ax.plot(prob_pred, prob_true, marker="o", label="Model (pre-calibration)")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (raw)")
    ax.legend()

    ax = axes[1, 2]
    youden_idx = np.argmax(tpr - fpr)
    best_thresh = thresholds[youden_idx]
    y_pred_opt = (oof >= best_thresh).astype(int)
    cm = confusion_matrix(y_true, y_pred_opt)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Pred 0", "Pred 1"],
        yticklabels=["True 0", "True 1"],
    )
    ax.set_title(f"Confusion Matrix @ optimal threshold={best_thresh:.3f}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Diagnostics figure saved to {save_path}")
    plt.close(fig)

    print("\n" + "=" * 60)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"Overall OOF AUC        : {overall_auc:.5f}")
    print(
        f"Fold AUC mean / std    : {np.mean(fold_aucs):.5f} / {np.std(fold_aucs):.5f}"
    )
    print(f"Optimal threshold (J)  : {best_thresh:.4f}")
    print(f"Class balance (TARGET) : {np.mean(y_true):.4f} positive rate")
    print("\nTop 15 features by importance:")
    print(mean_fi.head(15).to_string())

    return {
        "mean_feature_importance": mean_fi,
        "best_threshold": best_thresh,
        "confusion_matrix": cm,
    }


def export_artifacts(results, df):
    """Save OOF predictions, model bundle, and submission file."""
    raw_cols = [c for c in results["train_raw"].columns if c != "SK_ID_CURR"]
    oof_df = results["train_raw"].copy()
    oof_df["TARGET"] = results["y_true"]
    oof_df["PD_RAW"] = results["oof_preds"]
    oof_df.to_csv(config.OOF_PREDICTIONS_PATH, index=False)
    print(f"Saved OOF predictions: {config.OOF_PREDICTIONS_PATH}  shape={oof_df.shape}")

    model_bundle = {"fold_models": results["models"], "feature_names": results["feats"]}
    with open(config.MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f)
    print(
        f"Saved model bundle ({len(results['models'])} fold models): {config.MODEL_PATH}"
    )

    submission = pd.DataFrame(
        {"SK_ID_CURR": results["test_ids"], "TARGET": results["test_preds"]}
    )
    submission.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Saved submission: {config.SUBMISSION_PATH}")

    computed_auc = roc_auc_score(oof_df["TARGET"], oof_df["PD_RAW"])
    print(
        f"\nSanity check — recomputed AUC from saved file: {computed_auc:.5f} "
        f"(should match {results['overall_auc']:.5f})"
    )


if __name__ == "__main__":
    print("Building feature table...")
    df = build_features()

    print("\nTraining LightGBM with 5-fold CV...")
    results = train_model(df)

    print("\nRunning diagnostics...")
    diagnose_model(results, save_path=config.ARTIFACTS_DIR + "/model_diagnostics.png")

    print("\nExporting artifacts...")
    export_artifacts(results, df)

"""XGBoost fit/predict with no PyTorch import (runs in its own process on macOS)."""
import numpy as np


def fit_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray, seed: int, device: str = "cpu"):
    import xgboost as xgb

    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.08, subsample=0.8,
        colsample_bytree=0.8, eval_metric="auc", random_state=seed, n_jobs=-1,
        tree_method="hist", device=device,
    )
    clf.fit(X_tr, y_tr)
    return clf.predict_proba(X_te)[:, 1]

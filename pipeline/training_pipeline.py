import json
import joblib
import numpy as np
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


def load_model_config(config_path):
    with open(config_path, "r") as f:
        full_config = yaml.safe_load(f)
    return full_config.get("training", {})


def train_model(X_train, y_train, X_val, y_val, config):
    model = XGBClassifier(
        n_estimators=config.get("n_estimators", 100),
        learning_rate=config.get("learning_rate", 0.1),
        max_depth=config.get("max_depth", 5),
        random_state=42,
        eval_metric=config.get("eval_metric", "logloss"),
        early_stopping_rounds=config.get("early_stopping_rounds", 10),
        use_label_encoder=False,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    card = f"""### ZenAM Model Card

- Model: XGBoost
- Estimators: {model.n_estimators}
- Max Depth: {model.max_depth}
- Early Stopping: {model.early_stopping_rounds}
- Score (Val): {model.score(X_val, y_val):.4f}
"""

    return model, card


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True)
    return report


def save_model_assets(model, card, metrics, model_path, card_path, metrics_path):
    joblib.dump(model, model_path)
    with open(card_path, "w") as f:
        f.write(card)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

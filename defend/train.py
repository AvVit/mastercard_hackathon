import json
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score, precision_score, recall_score
import lightgbm as lgb

def train():
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'generate', 'data', 'train.jsonl')
    df = pd.read_json(data_path, lines=True)

    features = ['turn_count', 'amount', 'rebuff_heuristic_score', 'rebuff_similarity_score', 'rebuff_llm_score']
    X = df[features]
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        num_leaves=31,
        class_weight='balanced',
        random_state=42
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    auc   = roc_auc_score(y_test, probs)
    f1    = f1_score(y_test, preds)
    prec  = precision_score(y_test, preds)
    rec   = recall_score(y_test, preds)

    print("=" * 50)
    print("FRAUD DETECTOR — TRAINING RESULTS")
    print("=" * 50)
    print(f"ROC-AUC  : {auc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print("-" * 50)
    print(classification_report(y_test, preds, target_names=['Legitimate', 'Fraud']))

    # Save model
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'fraud_detector.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)

    # Save metrics JSON for README
    metrics = {"roc_auc": round(auc, 4), "f1": round(f1, 4), "precision": round(prec, 4), "recall": round(rec, 4)}
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nModel saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")

if __name__ == '__main__':
    train()

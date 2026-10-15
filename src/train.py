import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             f1_score, precision_score, recall_score)
import joblib

from preprocess import load_data, split_data, get_preprocessor, RANDOM_STATE


def get_models(balanced, pos_weight):
    cw = "balanced" if balanced else None
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight=cw),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1,
            class_weight=cw, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, learning_rate=0.1, max_depth=6,
            eval_metric="logloss", random_state=RANDOM_STATE,
            scale_pos_weight=pos_weight if balanced else 1),
    }


def evaluate(model, X, y):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "PR-AUC": average_precision_score(y, proba),
        "ROC-AUC": roc_auc_score(y, proba),
        "F1": f1_score(y, pred),
        "Precision": precision_score(y, pred),
        "Recall": recall_score(y, pred),
    }


def main():
    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()  # ~3.2

    results, pipelines = [], {}
    for balanced in [False, True]:
        for name, clf in get_models(balanced, pos_weight).items():
            pipe = Pipeline([("prep", clone(get_preprocessor())), ("model", clf)])
            pipe.fit(X_train, y_train)
            res = {"Model": name, "Balanced": balanced, **evaluate(pipe, X_val, y_val)}
            results.append(res)
            pipelines[(name, balanced)] = pipe
            print(f"{name:12s} balanced={balanced!s:5s} PR-AUC={res['PR-AUC']:.4f} "
                  f"Recall={res['Recall']:.3f} F1={res['F1']:.3f}")

    df = pd.DataFrame(results).round(4)
    df.to_csv("outputs/baseline_results.csv", index=False)
    print("\n", df.to_string(index=False))

    best = df.sort_values("PR-AUC", ascending=False).iloc[0]
    best_pipe = pipelines[(best["Model"], best["Balanced"])]
    joblib.dump(best_pipe, "outputs/best_model_pipeline.joblib")
    print(f"\nMeilleur : {best['Model']} (balanced={best['Balanced']}), PR-AUC={best['PR-AUC']}")


if __name__ == "__main__":
    main()

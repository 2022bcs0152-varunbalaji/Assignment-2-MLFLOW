import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc

from src.preprocess import load_data, clean_data, split_features_target, get_pipeline


mlflow.set_tracking_uri("file:./mlruns")

import os
os.makedirs("mlruns", exist_ok=True)

mlflow.set_experiment("churn_prediction_real_data")

train_df, test_df = load_data(
    "../data/customer_churn_dataset-training-master.csv",
    "../data/customer_churn_dataset-testing-master.csv"
)

train_df = clean_data(train_df)
test_df = clean_data(test_df)

X_train, y_train = split_features_target(train_df)
X_test, y_test = split_features_target(test_df)

pipeline = get_pipeline(X_train)
X_train = pipeline.fit_transform(X_train)
X_test = pipeline.transform(X_test)

with mlflow.start_run():

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    f1 = f1_score(y_test, preds)
    roc = roc_auc_score(y_test, probs)

    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = auc(recall, precision)

    print(f"F1 Score      : {f1:.4f}")
    print(f"ROC-AUC       : {roc:.4f}")
    print(f"PR-AUC        : {pr_auc:.4f}")

    mlflow.log_param("model", "RandomForest")
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("roc_auc", roc)
    mlflow.log_metric("pr_auc", pr_auc)

    os.makedirs("../model", exist_ok=True)
    joblib.dump(model, "../model/model.pkl")
    joblib.dump(pipeline, "../model/pipeline.pkl")

    mlflow.sklearn.log_model(model, "model")

print("Training completed successfully")
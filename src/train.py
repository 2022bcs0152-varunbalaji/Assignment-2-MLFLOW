import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc

from src.preprocess import load_data, clean_data, split_features_target, get_pipeline

mlflow.set_tracking_uri("file:./mlruns")
os.makedirs("mlruns", exist_ok=True)

mlflow.set_experiment("churn_prediction_real_data")

train_path = "../data/customer_churn_dataset-training-master.csv"
test_path = "../data/customer_churn_dataset-testing-master.csv"

if not os.path.exists(train_path):
    print(" Dataset not found → creating dummy dataset (CI mode)")

    df = pd.DataFrame({
        "Age": [25, 40, 30, 50],
        "Gender": ["Male", "Female", "Male", "Female"],
        "Tenure": [10, 20, 15, 25],
        "Usage Frequency": [5, 10, 7, 12],
        "Support Calls": [1, 2, 0, 3],
        "Payment Delay": [5, 3, 6, 2],
        "Subscription Type": ["Basic", "Premium", "Standard", "Basic"],
        "Contract Length": ["Monthly", "Annual", "Quarterly", "Monthly"],
        "Total Spend": [200, 500, 300, 600],
        "Last Interaction": [10, 20, 15, 25],
        "Churn": [0, 1, 0, 1]
    })

    os.makedirs("../data", exist_ok=True)
    df.to_csv(train_path, index=False)
    df.to_csv(test_path, index=False)

train_df, test_df = load_data(train_path, test_path)

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

print(" Training completed successfully")
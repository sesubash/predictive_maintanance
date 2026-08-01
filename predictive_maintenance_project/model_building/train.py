import os

import joblib
import mlflow
import pandas as pd
import xgboost as xgb
from dotenv import load_dotenv
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

load_dotenv()

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("engine-predictive-maintenance-experiment")

PROJECT_DIR = "predictive_maintenance_project"
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = os.path.join(PROJECT_DIR, "model_building")

# 4.1 Load the train and test data from the local project data folder
Xtrain = pd.read_csv(os.path.join(DATA_DIR, "Xtrain.csv"))
Xtest = pd.read_csv(os.path.join(DATA_DIR, "Xtest.csv"))
ytrain = pd.read_csv(os.path.join(DATA_DIR, "ytrain.csv")).squeeze()
ytest = pd.read_csv(os.path.join(DATA_DIR, "ytest.csv")).squeeze()
print(f"Train and test data loaded from {DATA_DIR}")

# 4.2 Define a model and parameters
scale_pos_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    random_state=42,
)
model_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("xgbclassifier", xgb_model),
])
param_grid = {
    "xgbclassifier__n_estimators": [50, 100],
    "xgbclassifier__max_depth": [3, 4, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
    "xgbclassifier__colsample_bytree": [0.6, 0.8],
    "xgbclassifier__reg_lambda": [0.4, 1.0],
}

# 4.3 Tune the model with the defined parameters
grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring="f1")
grid_search.fit(Xtrain, ytrain)
best_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)
print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")

# 4.4 Log all the tuned parameters
with mlflow.start_run(run_name="production_training_run"):
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_test_f1", results["mean_test_score"][i])
            mlflow.log_metric("std_test_f1", results["std_test_score"][i])

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)
    print("All tuned parameter combinations logged to MLflow.")

    # 4.5 Evaluate the model performance
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    print("\n=== TRAIN SET METRICS ===")
    print(f"Accuracy:  {train_report['accuracy']:.4f}")
    print(f"Precision: {train_report['1']['precision']:.4f}")
    print(f"Recall:    {train_report['1']['recall']:.4f}")
    print(f"F1-Score:  {train_report['1']['f1-score']:.4f}")
    print("\n=== Train Classification Report ===")
    print(classification_report(ytrain, y_pred_train))

    print("\n=== TEST SET METRICS ===")
    print(f"Accuracy:  {test_report['accuracy']:.4f}")
    print(f"Precision: {test_report['1']['precision']:.4f}")
    print(f"Recall:    {test_report['1']['recall']:.4f}")
    print(f"F1-Score:  {test_report['1']['f1-score']:.4f}")
    print("\n=== Test Classification Report ===")
    print(classification_report(ytest, y_pred_test))

    performance_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Train": [
            train_report["accuracy"],
            train_report["1"]["precision"],
            train_report["1"]["recall"],
            train_report["1"]["f1-score"],
        ],
        "Test": [
            test_report["accuracy"],
            test_report["1"]["precision"],
            test_report["1"]["recall"],
            test_report["1"]["f1-score"],
        ],
    })
    performance_df["Gap (Train - Test)"] = performance_df["Train"] - performance_df["Test"]
    print("\n=== MODEL PERFORMANCE SUMMARY ===")
    print(performance_df.round(4).to_string(index=False))

    mlflow.log_metrics({
        "train_accuracy": train_report["accuracy"],
        "train_precision": train_report["1"]["precision"],
        "train_recall": train_report["1"]["recall"],
        "train_f1": train_report["1"]["f1-score"],
        "test_accuracy": test_report["accuracy"],
        "test_precision": test_report["1"]["precision"],
        "test_recall": test_report["1"]["recall"],
        "test_f1": test_report["1"]["f1-score"],
    })

    train_f1 = train_report["1"]["f1-score"]
    test_f1 = test_report["1"]["f1-score"]
    print("\n=== PERFORMANCE INTERPRETATION ===")
    print(f"- Test F1-Score: {test_f1:.4f}")
    print(f"- Test Recall: {test_report['1']['recall']:.4f}")
    print(f"- Test Precision: {test_report['1']['precision']:.4f}")
    print(f"- Train-test F1 gap: {train_f1 - test_f1:.4f}")

    # 4.6 Save the best tuned model as a versioned artifact
    model_path = os.path.join(MODEL_DIR, "best_engine_maintenance_model_v1.joblib")
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"\nBest model saved locally as {model_path}")

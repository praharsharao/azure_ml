import sys
import importlib.metadata

# --- ENVIRONMENT DEBUG INFO ---
print("\n" + "="*30)
print("ENVIRONMENT DEBUG INFO")
print("="*30)
print(f"Python version: {sys.version}")

# 1. Check specific package versions safely
packages_to_check = ['setuptools', 'mlflow', 'mlflow-skinny', 'azureml-mlflow']
for pkg in packages_to_check:
    try:
        version = importlib.metadata.version(pkg)
        print(f"{pkg} version: {version}")
    except importlib.metadata.PackageNotFoundError:
        print(f"{pkg} is NOT INSTALLED")

# 2. Test if the infamous pkg_resources is finally available
try:
    import pkg_resources
    print(f"pkg_resources successfully imported! (setuptools version: {pkg_resources.__version__})")
except ModuleNotFoundError:
    print("CRITICAL ERROR: pkg_resources is STILL MISSING!")
print("="*30 + "\n")
# ------------------------------

import argparse
import pandas as pd
import numpy as np
import os
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, default='insurance.csv')
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {args.input_data}...")
    df = pd.read_csv(args.input_data)

    # 2. Features and Target
    drop_cols = ['customer_id', 'employer_id', 'churn_probability', 'retention_score', 'churn_flag']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    X = df.drop(existing_drop_cols, axis=1)
    y = df['churn_flag']

    # 3. Preprocessing
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_features),
            ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Model Definitions
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=(y.value_counts()[0]/y.value_counts()[1]), eval_metric='logloss')
    }

    best_model = None
    best_f1 = 0
    best_name = ""
    os.makedirs('outputs', exist_ok=True)

    # 5. The Training Loop (Creates 3 Child Runs)
    print("\n--- Starting Model Comparisons ---")
    for name, model in models.items():
        # Start a nested run for each model so they show up as separate trials
        with mlflow.start_run(run_name=name, nested=True):
            clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            clf.fit(X_train, y_train)
            score = f1_score(y_test, clf.predict(X_test))
            
            print(f"Finished {name} | F1: {score:.4f}")

            # Log metrics and params
            mlflow.log_param("model_type", name)
            mlflow.log_metric("f1_score", score)

            # SAVE MODEL AS ARTIFACT (Bypasses the 404 registry error)
            model_name_clean = name.replace(' ', '_').lower()
            local_path = f"outputs/{model_name_clean}.pkl"
            joblib.dump(clf, local_path)
            mlflow.log_artifact(local_path, artifact_path="model_trials")

            if score > best_f1:
                best_f1 = score
                best_model = clf
                best_name = name

    # 6. Final Results
    print(f"\nWINNER: {best_name} (F1: {best_f1:.4f})")
    joblib.dump(best_model, 'outputs/best_model.pkl')
    mlflow.log_artifact('outputs/best_model.pkl', artifact_path="final_model")

if __name__ == "__main__":
    main()
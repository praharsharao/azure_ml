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

    # 5. The Training Loop
    print("\n--- Starting Model Comparisons ---")
    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True):
            clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            clf.fit(X_train, y_train)
            score = f1_score(y_test, clf.predict(X_test))
            
            print(f"Finished {name} | F1: {score:.4f}")

            mlflow.log_param("model_type", name)
            mlflow.log_metric("f1_score", score)

            model_name_clean = name.replace(' ', '_').lower()
            local_path = f"outputs/{model_name_clean}.pkl"
            joblib.dump(clf, local_path)
            mlflow.log_artifact(local_path, artifact_path="model_trials")

            if score > best_f1:
                best_f1 = score
                best_model = clf
                best_name = name

    # 6. Final Results & AUTOMATIC VERSIONING
    print(f"\nWINNER: {best_name} (F1: {best_f1:.4f})")
    
    # This specifically tells Azure to register the model and bump the version!
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="insurance-churn-prediction-model"
    )
    print("✅ Model successfully registered to Azure ML!")

if __name__ == "__main__":
    main()

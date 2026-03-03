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
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, default='insurance.csv')
    args = parser.parse_args()

    # 1. Load Data
    print(f"Loading data from {args.input_data}...")
    df = pd.read_csv(args.input_data)

    # 2. Preprocessing setup
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

    # 3. Training Loop
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=(y.value_counts()[0]/y.value_counts()[1]), eval_metric='logloss')
    }

    best_model = None
    best_f1 = 0
    best_name = ""

    os.makedirs('outputs', exist_ok=True)

    for name, model in models.items():
        with mlflow.start_run(run_name=name, nested=True):
            clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            clf.fit(X_train, y_train)
            score = f1_score(y_test, clf.predict(X_test))
            
            # Log metrics and params
            mlflow.log_param("model_name", name)
            mlflow.log_metric("test_f1_score", score)

            # SAVE LOCALLY FIRST TO BYPASS 404 ERROR
            temp_path = f"outputs/{name.replace(' ', '_')}.pkl"
            joblib.dump(clf, temp_path)
            
            # UPLOAD AS ARTIFACT (This bypasses the /logged-models API)
            mlflow.log_artifact(temp_path, artifact_path="model_files")

            if score > best_f1:
                best_f1 = score
                best_model = clf
                best_name = name

    # 4. Final winner registration
    print(f"WINNER: {best_name} (F1: {best_f1:.4f})")
    joblib.dump(best_model, 'outputs/model.pkl')
    mlflow.log_artifact('outputs/model.pkl', artifact_path="best_model")

if __name__ == "__main__":
    main()
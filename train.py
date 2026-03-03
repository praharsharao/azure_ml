import argparse
import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
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

    # --- REMOVED: mlflow.set_experiment ---
    # Azure ML handles the experiment context automatically based on your 
    # submission script. Manual settings here cause ID mismatch errors.

    # 1. Load Data
    print(f"Loading data from {args.input_data}...")
    try:
        df = pd.read_csv(args.input_data)
    except FileNotFoundError:
        print(f"Error: The file '{args.input_data}' was not found.")
        return

    # 2. Define Features and Target
    drop_cols = ['customer_id', 'employer_id', 'churn_probability', 'retention_score', 'churn_flag']
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(existing_drop_cols, axis=1)
    y = df['churn_flag']

    # 3. Feature Selection
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    print(f"Numerical Features: {len(numeric_features)}")
    print(f"Categorical Features: {len(categorical_features)}")

    # 4. Preprocessing Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 6. Models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(
            scale_pos_weight=(y.value_counts()[0]/y.value_counts()[1]), 
            eval_metric='logloss'
        )
    }

    best_model = None
    best_f1 = 0
    best_name = ""

    print("\n--- Training & Evaluating Models with MLflow ---")
    
    # 7. Training Loop
    for name, model in models.items():
        # IMPORTANT: Use nested=True to create child runs under the Azure ML parent job
        with mlflow.start_run(run_name=name, nested=True):
            
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            score = f1_score(y_test, y_pred)
            print(f"{name:<20} | F1-Score: {score:.4f}")

            # MLflow Logging
            mlflow.log_param("model_name", name)
            mlflow.log_metric("test_f1_score", score)

            # Log the entire pipeline so preprocessing is bundled with the model
            mlflow.sklearn.log_model(clf, artifact_path="model")

            if score > best_f1:
                best_f1 = score
                best_model = clf
                best_name = name

    # 8. Final Results
    print(f"\n WINNER: {best_name} (F1: {best_f1:.4f})")
    
    # 9. Local Artifacts (Azure ML will collect anything in the 'outputs' folder)
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(best_model, 'outputs/best_insurance_model.pkl')

if __name__ == "__main__":
    main()
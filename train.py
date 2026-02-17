import argparse
import pandas as pd
import numpy as np
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from xgboost import XGBClassifier

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", type=str, default='insurance.csv')
    args = parser.parse_args()

    # --- MLflow Setup ---
    # Set the experiment name. If it doesn't exist, it will be created.
    mlflow.set_experiment("Insurance_Churn_Prediction")

    # 1. Load Data
    print(f"Loading data from {args.input_data}...")
    try:
        df = pd.read_csv(args.input_data)
    except FileNotFoundError:
        print(f"Error: The file '{args.input_data}' was not found.")
        return

    # 2. Define Features and Target
    # We drop ID columns and 'leakage' columns (probability/scores likely derived from the target)
    drop_cols = ['customer_id', 'employer_id', 'churn_probability', 'retention_score', 'churn_flag']
    
    # Check which columns actually exist before dropping
    existing_drop_cols = [c for c in drop_cols if c in df.columns]
    
    X = df.drop(existing_drop_cols, axis=1)
    y = df['churn_flag']

    # 3. Automatic Feature Selection
    # Identify numerical and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    print(f"Numerical Features: {len(numeric_features)}")
    print(f"Categorical Features: {len(categorical_features)}")

    # 4. Create Preprocessing Pipeline
    # This pipeline handles missing values and encoding automatically
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 6. Define Models to Compare
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
        "XGBoost": XGBClassifier(scale_pos_weight=(y.value_counts()[0]/y.value_counts()[1]), eval_metric='logloss', use_label_encoder=False)
    }

    best_model = None
    best_f1 = 0
    best_name = ""

    print("\n--- Training & Evaluating Models with MLflow ---")
    
    # 7. Train and Evaluate Loop
    for name, model in models.items():
        # --- MLflow Run Start ---
        # We start a new run for each model to compare them individually
        with mlflow.start_run(run_name=name):
            
            # Create a full pipeline: Preprocessor + Model
            clf = Pipeline(steps=[('preprocessor', preprocessor),
                                  ('classifier', model)])
            
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            
            # Calculate F1 score
            score = f1_score(y_test, y_pred)
            print(f"{name:<20} | F1-Score: {score:.4f}")

            # --- MLflow Logging ---
            # 1. Log the algorithm name as a parameter
            mlflow.log_param("model_name", name)
            
            # 2. Log key hyperparameters (optional, but good for tracking)
            if hasattr(model, 'n_estimators'):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, 'C'):
                mlflow.log_param("C", model.C)

            # 3. Log the metric (This is what you compare in the dashboard)
            mlflow.log_metric("test_f1_score", score)

            # 4. Log the model artifact (Saves the pickle file to MLflow)
            # We log the *entire pipeline* so preprocessing is saved with the model
            mlflow.sklearn.log_model(clf, artifact_path="model")

            # Update best model tracking logic (local)
            if score > best_f1:
                best_f1 = score
                best_model = clf
                best_name = name

    # 8. Final Results
    print(f"\n WINNER: {best_name} (F1: {best_f1:.4f})")
    print("\nFinal Classification Report:")
    print(classification_report(y_test, best_model.predict(X_test)))

    # 9. Save Best Model Locally (Optional, since MLflow already saved it)
    os.makedirs('outputs', exist_ok=True)
    model_path = 'outputs/best_insurance_model.pkl'
    joblib.dump(best_model, model_path)
    print(f" Best model saved locally to {model_path}")

if __name__ == "__main__":
    main()
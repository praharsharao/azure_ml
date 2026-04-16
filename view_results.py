import pandas as pd
import os

def main():
    # Azure usually nests the output file inside a 'score' directory
    file_path = "./final_predictions/score/batch_predictions.csv"
    
    # Fallback in case it's in the root of the download folder
    if not os.path.exists(file_path):
        file_path = "./final_predictions/batch_predictions.csv"
        
    if not os.path.exists(file_path):
        print(" Error: Could not find the predictions file. Please run 'ls -R ./final_predictions' to check the exact path.")
        return

    print(" Predictions successfully loaded!\n")
    
    # Read the output CSV (Batch outputs often drop the header row)
    df = pd.read_csv(file_path, header=None)
    
    print(" First 5 Predictions:")
    print(df.head())
    print("\n" + "-"*30 + "\n")
    
    # Grab the last column, which typically holds the model's prediction output
    pred_col = df.columns[-1]
    print(" Overall Churn Distribution:")
    print(df[pred_col].value_counts().to_string())

if __name__ == "__main__":
    main()
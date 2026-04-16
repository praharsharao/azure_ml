import pandas as pd

# Load your original clean data
df = pd.read_csv("insurance.csv")

# Print the actual column names so you know what is available
print("Your available columns are:", df.columns.tolist())

# Ruin the data! (Making everyone 40 years older guarantees statistical drift)
df['age'] = df['age'] + 40

# Save it as your dummy file
df.to_csv("dummy_drift_data.csv", index=False)
print("✅ Dummy data created successfully!")

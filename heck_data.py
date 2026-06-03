import pandas as pd

print("Checking Heart Dataset...")
heart_df = pd.read_csv('data/heart.csv', header=None)
print(f"Number of rows: {len(heart_df)}")
print(f"Number of columns: {len(heart_df.columns)}")
print(f"First 3 rows:\n{heart_df.head(3)}")

if len(heart_df) < 100:
    print("\n❌ ERROR: Heart dataset has too few rows!")
    print("Please download again from:")
    print("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data")
else:
    print("\n✅ Heart dataset looks good!")

print("\n" + "="*50)
print("Checking Diabetes Dataset...")
diabetes_df = pd.read_csv('data/diabetes.csv')
print(f"Number of rows: {len(diabetes_df)}")
print(f"Number of columns: {len(diabetes_df.columns)}")
print(f"First 3 rows:\n{diabetes_df.head(3)}")

if len(diabetes_df) < 100:
    print("\n❌ ERROR: Diabetes dataset has too few rows!")
else:
    print("\n✅ Diabetes dataset looks good!")
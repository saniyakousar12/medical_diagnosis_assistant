
import pandas as pd

# Check heart dataset
print("=" * 50)
print("HEART DATASET INSPECTION")
print("=" * 50)
heart_df = pd.read_csv('data/heart.csv')
print(f"Shape: {heart_df.shape}")
print(f"\nFirst 5 rows:")
print(heart_df.head())
print(f"\nColumn names:")
print(heart_df.columns.tolist())
print(f"\nData types:")
print(heart_df.dtypes)

print("\n" + "=" * 50)
print("DIABETES DATASET INSPECTION")
print("=" * 50)
diabetes_df = pd.read_csv('data/diabetes.csv')
print(f"Shape: {diabetes_df.shape}")
print(f"\nFirst 5 rows:")
print(diabetes_df.head())
print(f"\nColumn names:")
print(diabetes_df.columns.tolist())
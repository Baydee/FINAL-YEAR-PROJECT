import pandas as pd

# Load the original dataset
df = pd.read_csv('credit_card_fraud_10k.csv')

print("Original dataset:")
print(f"Total rows: {len(df)}")
print(f"Fraud (1): {len(df[df['is_fraud'] == 1])}")
print(f"Legitimate (0): {len(df[df['is_fraud'] == 0])}")

# Separate fraudulent and legitimate transactions
fraud = df[df['is_fraud'] == 1]
legitimate = df[df['is_fraud'] == 0]

print(f"\nFraud transactions available: {len(fraud)}")
print(f"Legitimate transactions available: {len(legitimate)}")

# For 50-50 balance, use all fraud transactions and same number of legitimate
num_fraud = len(fraud)
legitimate_balanced = legitimate.sample(n=num_fraud, random_state=42)

# Combine fraud and balanced legitimate
balanced_df = pd.concat([fraud, legitimate_balanced], ignore_index=True)

# Shuffle the dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\nBalanced dataset:")
print(f"Total rows: {len(balanced_df)}")
print(f"Fraud (1): {len(balanced_df[balanced_df['is_fraud'] == 1])}")
print(f"Legitimate (0): {len(balanced_df[balanced_df['is_fraud'] == 0])}")
print(f"Percentage fraud: {len(balanced_df[balanced_df['is_fraud'] == 1]) / len(balanced_df) * 100:.2f}%")

# Save balanced dataset
balanced_df.to_csv('credit_card_fraud_balanced.csv', index=False)
print("\n✅ Balanced dataset saved as 'credit_card_fraud_balanced.csv'")

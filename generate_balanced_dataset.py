import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

# Generate synthetic balanced dataset with 20% fraud rate
np.random.seed(42)

n_samples = 10000
fraud_rate = 0.20  # 20% fraud

# Create synthetic data
X, y = make_classification(
    n_samples=n_samples,
    n_features=8,
    n_informative=6,
    n_redundant=2,
    weights=[0.8, 0.2],  # 80% non-fraud, 20% fraud
    random_state=42,
    flip_y=0.01
)

# Create DataFrame with realistic feature names
df = pd.DataFrame({
    'transaction_id': range(1, n_samples + 1),
    'amount': np.random.uniform(10, 5000, n_samples),
    'transaction_hour': np.random.randint(0, 24, n_samples),
    'merchant_category': np.random.randint(0, 10, n_samples),
    'foreign_transaction': np.random.randint(0, 2, n_samples),
    'location_mismatch': np.random.randint(0, 2, n_samples),
    'device_trust_score': np.random.randint(0, 100, n_samples),
    'velocity_last_24h': np.random.randint(0, 20, n_samples),
    'cardholder_age': np.random.randint(18, 80, n_samples),
    'is_fraud': y
})

print("Dataset Generated Successfully!")
print(f"Total Transactions: {len(df)}")
print(f"\nFraud Distribution:")
print(df['is_fraud'].value_counts())
print(f"\nFraud Rate: {(df['is_fraud'].sum() / len(df) * 100):.2f}%")
print(f"\nDataset Shape: {df.shape}")
print(f"\nFirst 5 rows:")
print(df.head())

# Save to CSV
output_file = 'credit_card_fraud_balanced_20percent.csv'
df.to_csv(output_file, index=False)
print(f"\n✅ Saved to: {output_file}")

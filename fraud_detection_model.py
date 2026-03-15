import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_CATEGORIES_FALLBACK = ['Clothing', 'Electronics', 'Food', 'Grocery', 'Travel']
BASE_NUMERIC_FEATURES = [
	'amount',
	'transaction_hour',
	'transaction_day_of_week',
	'foreign_transaction',
	'vpn_proxy_detected',
	'location_mismatch',
	'device_trust_score',
	'velocity_last_1h',
	'velocity_last_24h',
	'cardholder_age',
]


def build_training_matrix(df):
	categories = sorted(df['merchant_category'].dropna().astype(str).str.title().unique().tolist()) or MODEL_CATEGORIES_FALLBACK
	encoded_categories = categories[1:]

	x_frame = df[BASE_NUMERIC_FEATURES].astype(float).copy()
	for category in encoded_categories:
		x_frame[f'merchant_category_{category}'] = (df['merchant_category'].astype(str).str.title() == category).astype(int)

	y_series = df['is_fraud'].astype(int)
	return x_frame, y_series


def load_dataset(path='credit_card_fraud_10k.csv'):
	df = pd.read_csv(path)
	for column, default_value in [
		('transaction_day_of_week', 0),
		('vpn_proxy_detected', 0),
		('velocity_last_1h', 0),
	]:
		if column not in df.columns:
			df[column] = default_value

	for column in BASE_NUMERIC_FEATURES:
		if column not in df.columns:
			df[column] = 0

	if 'merchant_category' not in df.columns:
		df['merchant_category'] = MODEL_CATEGORIES_FALLBACK[0]

	required = BASE_NUMERIC_FEATURES + ['merchant_category', 'is_fraud']
	return df[required].copy()


def main():
	df = load_dataset()
	print('Dataset shape:', df.shape)
	print('Class distribution:')
	print(df['is_fraud'].value_counts())

	X, y = build_training_matrix(df)

	x_train, x_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=0.2,
		random_state=42,
		stratify=y,
	)

	scaler = StandardScaler()
	x_train_scaled = scaler.fit_transform(x_train)
	x_test_scaled = scaler.transform(x_test)

	model = LGBMClassifier(
		n_estimators=200,
		learning_rate=0.05,
		num_leaves=31,
		random_state=42,
		class_weight='balanced',
	)
	model.fit(x_train_scaled, y_train)

	predictions = model.predict(x_test_scaled)
	probabilities = model.predict_proba(x_test_scaled)[:, 1]
	print('LightGBM Results:')
	print(classification_report(y_test, predictions))
	print('ROC-AUC:', roc_auc_score(y_test, probabilities))

	joblib.dump(model, 'lgbm_model.pkl')
	joblib.dump(scaler, 'scaler.pkl')
	print('Saved lgbm_model.pkl and scaler.pkl')


if __name__ == '__main__':
	main()
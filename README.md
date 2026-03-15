# Real-Time Fraud Detection System

This project now implements the missing requirements that were still described in PROJECT.txt but not present in the codebase: persistent PostgreSQL storage, geocoded user registration, user-aware transaction scoring, analyst review queues, and structured audit logging.

## What Was Added

- PostgreSQL-backed user, transaction, device, prediction, and audit-log persistence
- Runtime schema bootstrap in the Flask app so fresh and partially initialized databases are both supported
- User registration with address geocoding through OpenStreetMap Nominatim
- Transaction scoring that uses stored user profiles, transaction history, device reuse, transaction velocity, and distance from registered location
- Analyst endpoints for review queue, user transaction history, and system logs
- Updated dashboard with user registration, live scoring, review queue, high-value alerts, and selected-user history
- Correct merchant category encoding aligned to the trained model dataset
- Continuous model adaptation: each scored transaction is captured as a training sample and queued for background LightGBM retraining

## Architecture

- Flask serves the API and the analyst dashboard
- PostgreSQL stores users, transactions, devices, predictions, and logs
- LightGBM and the saved scaler provide fraud probability scoring
- Nginx serves the frontend and can proxy traffic in Docker deployments
- MinIO remains available for model/report storage in the compose stack

## Quick Start

### Docker Compose

```bash
docker-compose up -d
docker-compose up -d --force-recreate nginx
```

Verify health:

```bash
curl.exe -k -i https://localhost/api/health
```

Services:

- Dashboard: http://localhost
- API: http://localhost:5000/api
- PostgreSQL: localhost:5432
- MinIO Console: http://localhost:9001

### Local Development

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Make sure PostgreSQL is running and either:

- export or set DATABASE_URL, or
- use the default local credentials already coded in app.py

3. Start the app:

```bash
python app.py
```

## Start Tomorrow Checklist

Use this quick flow when you come back to the project.

1. Open a terminal in the project folder.

2. Start all services:

```bash
docker-compose up -d
docker-compose up -d --force-recreate nginx
```

3. Confirm containers are running:

```bash
docker ps
```

4. Open the dashboard:

- http://localhost

5. Verify API health:

```bash
curl.exe -k -i https://localhost/api/health
```

6. Sign in with analyst credentials from `.env`:

- Username: `ANALYST_USERNAME`
- Password: `ANALYST_PASSWORD`

7. If login/API fails after changes, recreate nginx:

```bash
docker-compose up -d --force-recreate nginx
```

8. If you are done for the day, stop everything:

```bash
docker-compose down
```

## Main API Endpoints

### Register User

```http
POST /api/register
Content-Type: application/json
```

```json
{
  "username": "jane_doe",
  "email": "jane@example.com",
  "phone_number": "+2348012345678",
  "address": "12 Allen Avenue, Ikeja, Lagos, Nigeria",
  "cardholder_age": 32
}
```

### List Users

```http
GET /api/users
```

### Score Transaction

```http
POST /api/predict
Content-Type: application/json
```

```json
{
  "user_id": "a1b2c3d4",
  "amount": 250.0,
  "currency": "NGN",
  "merchant_name": "Acme Electronics",
  "merchant_category": "Electronics",
  "timestamp": "2026-03-11T14:30:00Z",
  "device_fingerprint": "browser-device-hash",
  "latitude": 6.5244,
  "longitude": 3.3792,
  "ip_country": "Nigeria",
  "ip_city": "Lagos"
}
```

Example response:

```json
{
  "success": true,
  "transaction_id": "6f1d0f9f8a6d4e9b",
  "decision": "REVIEW",
  "risk_level": "MEDIUM",
  "fraud_probability": 0.58,
  "processing_time_ms": 41.72,
  "device_id": "38f62ad1be9cc0a2",
  "risk_factors": [
    "Low device trust score",
    "High-value transaction alert"
  ],
  "distance_from_home_km": 18.6,
  "velocity_last_24h": 4,
  "known_device": false
}
```

### Dashboard Summary

```http
GET /api/dashboard-data
```

### Review Queue

```http
GET /api/review-queue
```

### User Transaction History

```http
GET /api/users/<user_id>/transactions
```

### Audit Logs

```http
GET /api/logs
```

### Health Check

```http
GET /api/health
```

## Data Model

### users

Stores profile information, geocoded address data, registration metadata, age, and cumulative transaction count.

### transactions

Stores each scored transaction with merchant details, location fields, decision, risk level, and model confidence.

### devices

Stores known device fingerprints per user and updates last-seen time and transaction count.

### fraud_predictions

Stores the fraud label, confidence score, and JSON risk-factor list for each transaction.

### system_logs

Stores structured audit events for registration and scoring activity.

## Notes

- Merchant categories are derived from the training dataset and currently align to: Clothing, Electronics, Food, Grocery, Travel.
- The backend bootstraps missing tables and indexes on startup.
- Address geocoding depends on external OpenStreetMap Nominatim availability. If geocoding fails, registration still succeeds without coordinates.
- The dashboard polls for updated dashboard data every 5 seconds.

## Validation

The rewritten backend was compile-checked with:

```bash
python -m py_compile app.py
```

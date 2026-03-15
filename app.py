from csv import DictReader
import base64
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from functools import wraps
import hashlib
import json
import logging
import math
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
import bcrypt
import joblib
import jwt
from minio import Minio
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import pool
from psycopg2.extras import Json, RealDictCursor
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from dotenv import load_dotenv

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_CATEGORIES_FALLBACK = ['Clothing', 'Electronics', 'Food', 'Grocery', 'Travel']
DB_POOL = None
HIGH_AMOUNT_THRESHOLD = float(os.getenv('HIGH_AMOUNT_THRESHOLD', '500'))
DEFAULT_TRANSACTION_LATITUDE = float(os.getenv('DEFAULT_TRANSACTION_LATITUDE', '6.5244'))
DEFAULT_TRANSACTION_LONGITUDE = float(os.getenv('DEFAULT_TRANSACTION_LONGITUDE', '3.3792'))
DEFAULT_TRANSACTION_CITY = os.getenv('DEFAULT_TRANSACTION_CITY', 'Lagos')
DEFAULT_TRANSACTION_STATE = os.getenv('DEFAULT_TRANSACTION_STATE', 'Lagos State')
DEFAULT_TRANSACTION_COUNTRY = os.getenv('DEFAULT_TRANSACTION_COUNTRY', 'Nigeria')
MODEL_PRIMARY_PATH = os.getenv('MODEL_PRIMARY_PATH', 'lgbm_model.pkl')
MODEL_FALLBACK_PATH = os.getenv('MODEL_FALLBACK_PATH', 'xgb_model.pkl')
ONLINE_RETRAIN_ENABLED = os.getenv('ONLINE_RETRAIN_ENABLED', 'true').strip().lower() == 'true'
ONLINE_RETRAIN_LOOKBACK = int(os.getenv('ONLINE_RETRAIN_LOOKBACK', '5000'))
ONLINE_RETRAIN_MIN_SAMPLES = int(os.getenv('ONLINE_RETRAIN_MIN_SAMPLES', '100'))
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'change-me-in-production')
JWT_EXPIRY_HOURS = int(os.getenv('JWT_EXPIRY_HOURS', '8'))
ANALYST_API_KEY = os.getenv('ANALYST_API_KEY', '')
ANALYST_USERNAME = os.getenv('ANALYST_USERNAME', 'analyst')
ANALYST_PASSWORD_HASH = os.getenv('ANALYST_PASSWORD_HASH', '')
ANALYST_PASSWORD = os.getenv('ANALYST_PASSWORD', '')
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', '').strip()
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY', '').strip()
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY', '').strip()
MINIO_BUCKET = os.getenv('MINIO_BUCKET', 'fraud-models').strip() or 'fraud-models'
MINIO_SECURE = os.getenv('MINIO_SECURE', 'false').strip().lower() == 'true'
DATA_ENCRYPTION_KEY = os.getenv('DATA_ENCRYPTION_KEY', '').strip()
MINIO_CLIENT = None

SENSITIVE_FIELDS = {
    'email',
    'phone_number',
    'registered_address',
    'registration_ip',
    'registration_user_agent',
    'ip_address',
    'user_agent',
}

TRANSFER_MERCHANT_CATEGORY = 'Travel'
FAKE_BANK_NAMES = [
    'Atlas Trust Bank',
    'Harborline Bank',
    'CrestPoint Microfinance',
    'Northfield Savings',
    'Summit Union Bank',
    'Blue Oak Bank',
]
FAKE_FIRST_NAMES = ['Ada', 'David', 'Chiamaka', 'Samuel', 'Lara', 'Emeka', 'Zainab', 'Tobi', 'Nora', 'Ife']
FAKE_LAST_NAMES = ['Adebayo', 'Okafor', 'Balogun', 'Daniels', 'Umeh', 'Hassan', 'Ibrahim', 'Akinola', 'Eze', 'Nwosu']

MODEL_LOCK = threading.RLock()
RETRAIN_LOCK = threading.Lock()
RETRAIN_PENDING = False
RETRAIN_IN_PROGRESS = False


class RequestError(Exception):
    def __init__(self, payload, status_code):
        super().__init__(payload)
        self.payload = payload
        self.status_code = status_code


def _load_aes_key():
    if not DATA_ENCRYPTION_KEY:
        logger.warning('DATA_ENCRYPTION_KEY is not set. Sensitive fields will be stored in plaintext.')
        return None

    try:
        decoded = base64.urlsafe_b64decode(DATA_ENCRYPTION_KEY)
    except Exception:
        decoded = DATA_ENCRYPTION_KEY.encode('utf-8')

    if len(decoded) != 32:
        logger.warning('DATA_ENCRYPTION_KEY must resolve to exactly 32 bytes for AES-256. Encryption disabled.')
        return None
    return decoded


AES_KEY = _load_aes_key()


def build_database_url():
    if os.getenv('DATABASE_URL'):
        return os.getenv('DATABASE_URL')

    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'fraud_detection_db')
    db_user = os.getenv('DB_USER', 'fraud_admin')
    db_password = os.getenv('DB_PASSWORD', 'fraud_secure_password_123')
    return f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'


DATABASE_URL = build_database_url()


SCHEMA_STATEMENTS = [
    '''
    CREATE TABLE IF NOT EXISTS users (
        user_id VARCHAR(50) PRIMARY KEY,
        username VARCHAR(100) NOT NULL UNIQUE,
        email VARCHAR(100) NOT NULL UNIQUE,
        phone_number VARCHAR(20),
        registered_address TEXT,
        registered_country VARCHAR(50),
        registered_city VARCHAR(50),
        registered_latitude DECIMAL(10, 8),
        registered_longitude DECIMAL(11, 8),
        cardholder_age INTEGER,
        registration_ip VARCHAR(45),
        registration_user_agent TEXT,
        account_creation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        total_transactions INTEGER DEFAULT 0
    )
    ''',
    '''
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id VARCHAR(50) PRIMARY KEY,
        user_id VARCHAR(50) NOT NULL,
        amount DECIMAL(10, 2) NOT NULL,
        currency VARCHAR(3) DEFAULT 'NGN',
        merchant_name VARCHAR(255),
        merchant_category VARCHAR(50),
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        latitude DECIMAL(10, 8),
        longitude DECIMAL(11, 8),
        ip_address VARCHAR(45),
        ip_country VARCHAR(50),
        ip_city VARCHAR(50),
        device_fingerprint VARCHAR(255),
        user_agent TEXT,
        prediction VARCHAR(20),
        confidence_score DECIMAL(5, 4),
        risk_level VARCHAR(20),
        status VARCHAR(20),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''',
    '''
    CREATE TABLE IF NOT EXISTS devices (
        device_id VARCHAR(50) PRIMARY KEY,
        user_id VARCHAR(50) NOT NULL,
        device_fingerprint VARCHAR(255) NOT NULL,
        device_type VARCHAR(50),
        browser VARCHAR(100),
        operating_system VARCHAR(100),
        first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        total_transactions INTEGER DEFAULT 0,
        UNIQUE(user_id, device_fingerprint),
        FOREIGN KEY (user_id) REFERENCES users(user_id)
    )
    ''',
    '''
    CREATE TABLE IF NOT EXISTS fraud_predictions (
        prediction_id SERIAL PRIMARY KEY,
        transaction_id VARCHAR(50) UNIQUE,
        prediction VARCHAR(20),
        confidence_score DECIMAL(5, 4),
        risk_factors JSONB,
        feature_context JSONB,
        training_label SMALLINT,
        prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
    )
    ''',
    '''
    CREATE TABLE IF NOT EXISTS system_logs (
        log_id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        log_level VARCHAR(20),
        message TEXT,
        transaction_id VARCHAR(50),
        user_id VARCHAR(50),
        metadata JSONB
    )
    ''',
    'ALTER TABLE users ADD COLUMN IF NOT EXISTS cardholder_age INTEGER',
    'ALTER TABLE users ALTER COLUMN email TYPE TEXT',
    'ALTER TABLE users ALTER COLUMN phone_number TYPE TEXT',
    'ALTER TABLE users ALTER COLUMN registration_ip TYPE TEXT',
    'ALTER TABLE transactions ALTER COLUMN ip_address TYPE TEXT',
    "ALTER TABLE transactions ALTER COLUMN currency SET DEFAULT 'NGN'",
    'ALTER TABLE users ADD COLUMN IF NOT EXISTS email_hash VARCHAR(64)',
    'CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email_hash ON users(email_hash)',
    'ALTER TABLE system_logs ADD COLUMN IF NOT EXISTS transaction_id VARCHAR(50)',
    'ALTER TABLE system_logs ADD COLUMN IF NOT EXISTS user_id VARCHAR(50)',
    'ALTER TABLE system_logs ADD COLUMN IF NOT EXISTS metadata JSONB',
    'ALTER TABLE fraud_predictions ADD COLUMN IF NOT EXISTS feature_context JSONB',
    'ALTER TABLE fraud_predictions ADD COLUMN IF NOT EXISTS training_label SMALLINT',
    'CREATE INDEX IF NOT EXISTS idx_transactions_user_id ON transactions(user_id)',
    'CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_transactions_risk_level ON transactions(risk_level)',
    'CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status)',
    'CREATE INDEX IF NOT EXISTS idx_devices_user_id ON devices(user_id)',
    'CREATE UNIQUE INDEX IF NOT EXISTS idx_devices_user_fingerprint ON devices(user_id, device_fingerprint)',
    'CREATE INDEX IF NOT EXISTS idx_system_logs_timestamp ON system_logs(timestamp)',
    'CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON fraud_predictions(prediction_timestamp)'
]


try:
    model_path = MODEL_PRIMARY_PATH if os.path.exists(MODEL_PRIMARY_PATH) else MODEL_FALLBACK_PATH
    model = joblib.load(model_path)
    scaler = joblib.load('scaler.pkl')
    MODEL_LOADED = True
    logger.info('Model artifacts loaded successfully from %s', model_path)
except Exception as exc:
    model = None
    scaler = None
    MODEL_LOADED = False
    logger.exception('Failed to load model artifacts: %s', exc)


def load_model_categories():
    dataset_path = os.path.join(os.path.dirname(__file__), 'credit_card_fraud_10k.csv')
    if not os.path.exists(dataset_path):
        return MODEL_CATEGORIES_FALLBACK

    categories = set()
    with open(dataset_path, 'r', encoding='utf-8') as dataset_file:
        reader = DictReader(dataset_file)
        for row in reader:
            category = (row.get('merchant_category') or '').strip()
            if category:
                categories.add(category)

    return sorted(categories) or MODEL_CATEGORIES_FALLBACK


MODEL_CATEGORIES = load_model_categories()
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
    'cardholder_age'
]
ENCODED_CATEGORIES = MODEL_CATEGORIES[1:]


def _base_dataset_training_frame():
    dataset_path = os.path.join(os.path.dirname(__file__), 'credit_card_fraud_10k.csv')
    if not os.path.exists(dataset_path):
        return pd.DataFrame()

    df = pd.read_csv(dataset_path)
    if 'is_fraud' not in df.columns:
        return pd.DataFrame()

    # Keep compatibility with legacy dataset by backfilling newly engineered features.
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
        df['merchant_category'] = MODEL_CATEGORIES[0]

    return df[BASE_NUMERIC_FEATURES + ['merchant_category', 'is_fraud']].copy()


def _runtime_training_frame(connection):
    with connection.cursor(cursor_factory=RealDictCursor) as cursor:
        cursor.execute(
            '''
            SELECT feature_context, training_label
            FROM fraud_predictions
            WHERE feature_context IS NOT NULL AND training_label IS NOT NULL
            ORDER BY prediction_timestamp DESC
            LIMIT %s
            ''',
            (ONLINE_RETRAIN_LOOKBACK,)
        )
        rows = cursor.fetchall()

    records = []
    for row in rows:
        context = row.get('feature_context') or {}
        if not isinstance(context, dict):
            continue

        record = {
            'merchant_category': context.get('merchant_category') or MODEL_CATEGORIES[0],
            'is_fraud': int(row.get('training_label') or 0),
        }
        for feature_name in BASE_NUMERIC_FEATURES:
            record[feature_name] = to_float(context.get(feature_name), 0)
        records.append(record)

    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)


def _to_training_matrix(df):
    training_df = df.copy()
    for feature_name in BASE_NUMERIC_FEATURES:
        if feature_name not in training_df.columns:
            training_df[feature_name] = 0
    if 'merchant_category' not in training_df.columns:
        training_df['merchant_category'] = MODEL_CATEGORIES[0]

    X = training_df[BASE_NUMERIC_FEATURES].astype(float).copy()
    for category in ENCODED_CATEGORIES:
        X[f'merchant_category_{category}'] = (training_df['merchant_category'] == category).astype(int)

    y = training_df['is_fraud'].astype(int)
    return X, y


def _fit_lightgbm_model(training_df):
    if LGBMClassifier is None:
        raise RuntimeError('LightGBM is not available. Install lightgbm in the runtime environment.')

    if training_df.empty or 'is_fraud' not in training_df.columns:
        raise RuntimeError('No labeled training data is available for LightGBM retraining.')

    X, y = _to_training_matrix(training_df)
    if len(X) < ONLINE_RETRAIN_MIN_SAMPLES or y.nunique() < 2:
        raise RuntimeError('Not enough class-diverse data for LightGBM retraining yet.')

    from sklearn.preprocessing import StandardScaler

    local_scaler = StandardScaler()
    X_scaled = local_scaler.fit_transform(X)

    local_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=31,
        random_state=42,
        class_weight='balanced',
    )
    local_model.fit(X_scaled, y)
    return local_model, local_scaler


def retrain_model_from_history(trigger='runtime'):
    if not ONLINE_RETRAIN_ENABLED:
        return

    connection = None
    try:
        base_df = _base_dataset_training_frame()
        connection = get_db_connection()
        runtime_df = _runtime_training_frame(connection)

        if runtime_df.empty:
            combined_df = base_df
        elif base_df.empty:
            combined_df = runtime_df
        else:
            combined_df = pd.concat([base_df, runtime_df], ignore_index=True)

        if combined_df.empty:
            raise RuntimeError('No training samples available yet.')

        local_model, local_scaler = _fit_lightgbm_model(combined_df)
        with MODEL_LOCK:
            global model, scaler, MODEL_LOADED
            model = local_model
            scaler = local_scaler
            MODEL_LOADED = True
        logger.info('LightGBM model refreshed (%s). Samples=%s', trigger, len(combined_df))
    except Exception as exc:
        logger.warning('Skipping retrain (%s): %s', trigger, exc)
    finally:
        release_db_connection(connection)


def _retrain_worker():
    global RETRAIN_PENDING, RETRAIN_IN_PROGRESS
    while True:
        with RETRAIN_LOCK:
            if not RETRAIN_PENDING:
                RETRAIN_IN_PROGRESS = False
                return
            RETRAIN_PENDING = False

        retrain_model_from_history(trigger='post-transaction')


def queue_retrain():
    if not ONLINE_RETRAIN_ENABLED:
        return

    global RETRAIN_PENDING, RETRAIN_IN_PROGRESS
    with RETRAIN_LOCK:
        RETRAIN_PENDING = True
        if RETRAIN_IN_PROGRESS:
            return
        RETRAIN_IN_PROGRESS = True

    threading.Thread(target=_retrain_worker, name='lightgbm-retrainer', daemon=True).start()


EMAIL_PATTERN = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def init_db_pool():
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = pool.SimpleConnectionPool(1, 10, dsn=DATABASE_URL)
        logger.info('Database pool initialized')
        bootstrap_database()
        sync_model_artifacts_to_minio()
        retrain_model_from_history(trigger='startup')


def get_db_connection():
    init_db_pool()
    return DB_POOL.getconn()


def release_db_connection(connection):
    if DB_POOL and connection:
        DB_POOL.putconn(connection)


def bootstrap_database():
    connection = None
    try:
        connection = DB_POOL.getconn()
        connection.autocommit = True
        with connection.cursor() as cursor:
            for statement in SCHEMA_STATEMENTS:
                cursor.execute(statement)
    finally:
        if connection:
            DB_POOL.putconn(connection)


def to_float(value, default=None):
    if value in (None, ''):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def serialize_row(row):
    if row is None:
        return None

    serialized = {}
    for key, value in row.items():
        if isinstance(value, Decimal):
            serialized[key] = float(value)
        elif isinstance(value, datetime):
            serialized[key] = value.isoformat()
        else:
            serialized[key] = value

    for field in SENSITIVE_FIELDS:
        if field in serialized:
            serialized[field] = decrypt_value(serialized.get(field))

    serialized.pop('email_hash', None)
    return serialized


def email_hash(email):
    return hashlib.sha256((email or '').strip().lower().encode('utf-8')).hexdigest()


def encrypt_value(value):
    if value in (None, '') or AES_KEY is None:
        return value

    plaintext = str(value).encode('utf-8')
    nonce = secrets.token_bytes(12)
    ciphertext = AESGCM(AES_KEY).encrypt(nonce, plaintext, None)
    token = base64.urlsafe_b64encode(nonce + ciphertext).decode('utf-8')
    return f'enc:v1:{token}'


def decrypt_value(value):
    if value in (None, '') or not isinstance(value, str):
        return value
    if not value.startswith('enc:v1:') or AES_KEY is None:
        return value

    try:
        token = value.split('enc:v1:', 1)[1]
        raw = base64.urlsafe_b64decode(token)
        nonce, ciphertext = raw[:12], raw[12:]
        plaintext = AESGCM(AES_KEY).decrypt(nonce, ciphertext, None)
        return plaintext.decode('utf-8')
    except Exception:
        return value


def get_password_hash():
    if ANALYST_PASSWORD_HASH:
        return ANALYST_PASSWORD_HASH.encode('utf-8')
    if ANALYST_PASSWORD:
        return bcrypt.hashpw(ANALYST_PASSWORD.encode('utf-8'), bcrypt.gensalt(rounds=12))
    return None


def create_access_token(username):
    now = datetime.now(timezone.utc)
    payload = {
        'sub': username,
        'iat': int(now.timestamp()),
        'exp': int((now + timedelta(hours=JWT_EXPIRY_HOURS)).timestamp()),
        'scope': 'analyst',
    }
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm='HS256')


def verify_access_token(token):
    try:
        return jwt.decode(token, JWT_SECRET_KEY, algorithms=['HS256'])
    except jwt.InvalidTokenError:
        return None


def is_request_authorized():
    provided_api_key = request.headers.get('X-API-Key', '').strip()
    if ANALYST_API_KEY and provided_api_key and secrets.compare_digest(provided_api_key, ANALYST_API_KEY):
        return True

    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        token = auth_header.split(' ', 1)[1].strip()
        if verify_access_token(token):
            return True
    return False


def require_analyst_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not is_request_authorized():
            return jsonify({'success': False, 'error': 'Authentication required'}), 401
        return func(*args, **kwargs)

    return wrapper


def get_minio_client():
    global MINIO_CLIENT
    if MINIO_CLIENT is not None:
        return MINIO_CLIENT
    if not MINIO_ENDPOINT or not MINIO_ACCESS_KEY or not MINIO_SECRET_KEY:
        return None

    MINIO_CLIENT = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=MINIO_SECURE,
    )
    return MINIO_CLIENT


def sync_model_artifacts_to_minio():
    client = get_minio_client()
    if client is None:
        logger.info('MinIO credentials are not configured. Skipping object storage sync.')
        return

    try:
        if not client.bucket_exists(MINIO_BUCKET):
            client.make_bucket(MINIO_BUCKET)

        timestamp = datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')
        artifacts = ['lgbm_model.pkl', 'xgb_model.pkl', 'scaler.pkl']
        uploaded_objects = []

        for artifact in artifacts:
            if not os.path.exists(artifact):
                continue
            versioned_name = f'models/{timestamp}/{artifact}'
            latest_name = f'models/latest/{artifact}'
            client.fput_object(MINIO_BUCKET, versioned_name, artifact)
            client.fput_object(MINIO_BUCKET, latest_name, artifact)
            uploaded_objects.append(versioned_name)

        metadata = {
            'synced_at': datetime.now(timezone.utc).isoformat(),
            'model_loaded': MODEL_LOADED,
            'merchant_categories': MODEL_CATEGORIES,
            'objects': uploaded_objects,
        }
        metadata_path = 'model_metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
            json.dump(metadata, metadata_file, indent=2)
        client.fput_object(MINIO_BUCKET, f'models/{timestamp}/model_metadata.json', metadata_path)
        client.fput_object(MINIO_BUCKET, 'models/latest/model_metadata.json', metadata_path)
        os.remove(metadata_path)
    except Exception as exc:
        logger.warning('MinIO sync failed: %s', exc)


def get_client_ip():
    forwarded = request.headers.get('X-Forwarded-For')
    if forwarded:
        return forwarded.split(',')[0].strip()
    return request.remote_addr or '127.0.0.1'


def get_device_fingerprint(payload, ip_address, user_agent):
    supplied_fingerprint = (payload.get('device_fingerprint') or '').strip()
    if supplied_fingerprint:
        return supplied_fingerprint

    raw_fingerprint = f"{ip_address}|{user_agent}|{payload.get('user_id', '')}"
    return hashlib.sha256(raw_fingerprint.encode('utf-8')).hexdigest()[:32]


def normalize_timestamp(value):
    if not value:
        return datetime.now(timezone.utc)

    cleaned = value.replace('Z', '+00:00')
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalize_merchant_category(value):
    if value is None:
        return None

    if isinstance(value, int) or (isinstance(value, str) and value.isdigit()):
        index = int(value)
        if 0 <= index < len(MODEL_CATEGORIES):
            return MODEL_CATEGORIES[index]
        return None

    normalized = str(value).strip().title()
    for category in MODEL_CATEGORIES:
        if category.lower() == normalized.lower():
            return category
    return None


def geocode_address(address):
    if not address:
        return {}

    query = urllib.parse.quote(address)
    url = f'https://nominatim.openstreetmap.org/search?q={query}&format=jsonv2&limit=1&addressdetails=1'
    req = urllib.request.Request(url, headers={'User-Agent': 'fraud-detection-academic-project/1.0'})

    try:
        with urllib.request.urlopen(req, timeout=6) as response:
            payload = json.loads(response.read().decode('utf-8'))
            if not payload:
                return {}
            item = payload[0]
            address_parts = item.get('address', {})
            city_value = (
                address_parts.get('city')
                or address_parts.get('town')
                or address_parts.get('village')
                or address_parts.get('municipality')
                or address_parts.get('county')
                or address_parts.get('state_district')
                or address_parts.get('state')
            )

            if not city_value:
                # Fallback to display_name tokens when city-like fields are unavailable.
                display_name = item.get('display_name') or ''
                parts = [part.strip() for part in display_name.split(',') if part.strip()]
                if len(parts) >= 2:
                    city_value = parts[-2]

            return {
                'latitude': float(item['lat']),
                'longitude': float(item['lon']),
                'city': city_value,
                'country': address_parts.get('country'),
                'display_name': item.get('display_name')
            }
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError) as exc:
        logger.warning('Geocoding failed for address %s: %s', address, exc)
        return {}


def reverse_geocode_coordinates(latitude, longitude):
    lat = to_float(latitude)
    lon = to_float(longitude)
    if lat is None or lon is None:
        return {}

    url = (
        'https://nominatim.openstreetmap.org/reverse'
        f'?lat={lat}&lon={lon}&format=jsonv2&addressdetails=1'
    )
    req = urllib.request.Request(url, headers={'User-Agent': 'fraud-detection-academic-project/1.0'})

    try:
        with urllib.request.urlopen(req, timeout=6) as response:
            payload = json.loads(response.read().decode('utf-8'))
            address_parts = payload.get('address', {})
            city_value = (
                address_parts.get('city')
                or address_parts.get('town')
                or address_parts.get('village')
                or address_parts.get('municipality')
                or address_parts.get('county')
            )
            return {
                'city': city_value,
                'state': address_parts.get('state') or address_parts.get('region'),
                'country': address_parts.get('country'),
            }
    except Exception as exc:
        logger.warning('Reverse geocoding failed for %s,%s: %s', lat, lon, exc)
        return {}


def infer_city_from_address(address):
    if not address:
        return None

    parts = [part.strip() for part in str(address).split(',') if part.strip()]
    if len(parts) >= 2:
        return parts[-2]
    return None


def haversine_km(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None

    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    return radius_km * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))


def derive_browser(user_agent):
    lowered = user_agent.lower()
    if 'edg' in lowered:
        return 'Edge'
    if 'chrome' in lowered and 'edg' not in lowered:
        return 'Chrome'
    if 'firefox' in lowered:
        return 'Firefox'
    if 'safari' in lowered and 'chrome' not in lowered:
        return 'Safari'
    return 'Unknown'


def derive_os(user_agent):
    lowered = user_agent.lower()
    if 'windows' in lowered:
        return 'Windows'
    if 'android' in lowered:
        return 'Android'
    if 'iphone' in lowered or 'ios' in lowered:
        return 'iOS'
    if 'mac os' in lowered or 'macintosh' in lowered:
        return 'macOS'
    if 'linux' in lowered:
        return 'Linux'
    return 'Unknown'


def derive_device_type(user_agent):
    lowered = user_agent.lower()
    if 'mobile' in lowered or 'android' in lowered or 'iphone' in lowered:
        return 'mobile'
    if 'tablet' in lowered or 'ipad' in lowered:
        return 'tablet'
    return 'desktop'


def validate_registration_input(payload):
    errors = []
    required_fields = ['username', 'email', 'phone_number', 'address']

    for field in required_fields:
        if not str(payload.get(field, '')).strip():
            errors.append(f'Missing required field: {field}')

    if payload.get('email') and not EMAIL_PATTERN.match(payload['email']):
        errors.append('Email format is invalid')

    cardholder_age = payload.get('cardholder_age')
    if cardholder_age not in (None, ''):
        try:
            age = int(cardholder_age)
            if age < 18 or age > 100:
                errors.append('cardholder_age must be between 18 and 100')
        except (TypeError, ValueError):
            errors.append('cardholder_age must be an integer')

    return errors


def validate_transaction_input(payload):
    errors = []
    required_fields = ['user_id', 'amount', 'merchant_name', 'merchant_category']

    for field in required_fields:
        if payload.get(field) in (None, ''):
            errors.append(f'Missing required field: {field}')

    amount = to_float(payload.get('amount'))
    if amount is None or amount <= 0:
        errors.append('Amount must be a positive number')

    merchant_category = normalize_merchant_category(payload.get('merchant_category'))
    if merchant_category is None:
        errors.append(f"merchant_category must be one of: {', '.join(MODEL_CATEGORIES)}")

    if payload.get('timestamp'):
        try:
            normalize_timestamp(payload['timestamp'])
        except ValueError:
            errors.append('timestamp must be valid ISO 8601')

    latitude = payload.get('latitude')
    longitude = payload.get('longitude')
    if latitude not in (None, '') and to_float(latitude) is None:
        errors.append('latitude must be numeric')
    if longitude not in (None, '') and to_float(longitude) is None:
        errors.append('longitude must be numeric')

    return errors


def fetch_user(cursor, user_id):
    cursor.execute('SELECT * FROM users WHERE user_id = %s', (user_id,))
    return cursor.fetchone()


def fetch_user_stats(cursor, user_id, transaction_time):
    last_1h = transaction_time - timedelta(hours=1)
    last_24h = transaction_time - timedelta(hours=24)

    cursor.execute(
        '''
                SELECT
                        (SELECT COUNT(*) FROM transactions WHERE user_id = %s AND timestamp >= %s) AS tx_count_last_1h,
                        (SELECT COUNT(*) FROM transactions WHERE user_id = %s AND timestamp >= %s) AS tx_count_last_24h,
                        (
                                SELECT COALESCE(AVG(amount), 0)
                                FROM transactions
                                WHERE user_id = %s
                                    AND timestamp >= %s
                                    AND status = 'APPROVE'
                                    AND prediction = 'LEGITIMATE'
                        ) AS avg_amount
        ''',
        (
                        user_id,
            last_1h.replace(tzinfo=None),
                        user_id,
            last_24h.replace(tzinfo=None),
            user_id,
            last_24h.replace(tzinfo=None),
        )
    )
    aggregate = cursor.fetchone()

    cursor.execute(
        '''
        SELECT latitude, longitude, timestamp, amount
        FROM transactions
        WHERE user_id = %s AND latitude IS NOT NULL AND longitude IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT 1
        ''',
        (user_id,)
    )
    previous_transaction = cursor.fetchone()

    return {
        'velocity_last_1h': int(aggregate['tx_count_last_1h']) if aggregate else 0,
        'velocity_last_24h': int(aggregate['tx_count_last_24h']) if aggregate else 0,
        'average_amount_last_24h': float(aggregate['avg_amount']) if aggregate else 0.0,
        'previous_transaction': previous_transaction,
    }


def lookup_ip_risk(ip_address):
    if not ip_address or ip_address in ('127.0.0.1', '::1'):
        return {
            'is_proxy_or_vpn': False,
            'provider': None,
            'ip_country': None,
            'ip_state': None,
            'ip_city': None,
            'ip_latitude': None,
            'ip_longitude': None,
        }

    url = f'http://ip-api.com/json/{urllib.parse.quote(ip_address)}?fields=status,proxy,hosting,isp,country,regionName,city,lat,lon,message'
    try:
        with urllib.request.urlopen(url, timeout=4) as response:
            payload = json.loads(response.read().decode('utf-8'))
            if payload.get('status') != 'success':
                return {
                    'is_proxy_or_vpn': False,
                    'provider': None,
                    'ip_country': None,
                    'ip_state': None,
                    'ip_city': None,
                    'ip_latitude': None,
                    'ip_longitude': None,
                }
            return {
                'is_proxy_or_vpn': bool(payload.get('proxy') or payload.get('hosting')),
                'provider': payload.get('isp'),
                'ip_country': payload.get('country'),
                'ip_state': payload.get('regionName'),
                'ip_city': payload.get('city'),
                'ip_latitude': to_float(payload.get('lat')),
                'ip_longitude': to_float(payload.get('lon')),
            }
    except Exception:
        return {
            'is_proxy_or_vpn': False,
            'provider': None,
            'ip_country': None,
            'ip_state': None,
            'ip_city': None,
            'ip_latitude': None,
            'ip_longitude': None,
        }


def resolve_transaction_location(payload, user, ip_risk):
    ip_country = (ip_risk.get('ip_country') or '').strip()
    ip_state = (ip_risk.get('ip_state') or '').strip()
    ip_city = (ip_risk.get('ip_city') or '').strip()
    profile_country = (user.get('registered_country') or '').strip()
    profile_city = (user.get('registered_city') or '').strip()

    country = ''
    state = ''
    city = ''
    latitude = None
    longitude = None

    browser_requested = (str(payload.get('geolocation_source') or '').strip().lower() == 'browser')
    browser_lat = to_float(payload.get('latitude'))
    browser_lon = to_float(payload.get('longitude'))
    browser_geo = {}

    if browser_requested and browser_lat is not None and browser_lon is not None:
        latitude = browser_lat
        longitude = browser_lon
        browser_geo = reverse_geocode_coordinates(browser_lat, browser_lon)
        country = (browser_geo.get('country') or '').strip()
        state = (browser_geo.get('state') or '').strip()
        city = (browser_geo.get('city') or '').strip()

    # Fall back to IP geolocation when browser geolocation is unavailable.
    if not country:
        country = ip_country
    if not state:
        state = ip_state
    if not city:
        city = ip_city
    if latitude is None:
        latitude = to_float(ip_risk.get('ip_latitude'))
    if longitude is None:
        longitude = to_float(ip_risk.get('ip_longitude'))

    profile_used = False
    default_used = False

    if not country:
        country = profile_country
        profile_used = bool(country)
    if not city:
        city = profile_city
        profile_used = profile_used or bool(city)

    if not country:
        country = DEFAULT_TRANSACTION_COUNTRY
        default_used = True
    if not city:
        city = DEFAULT_TRANSACTION_CITY
        default_used = True
    if latitude is None:
        latitude = DEFAULT_TRANSACTION_LATITUDE
        default_used = True
    if longitude is None:
        longitude = DEFAULT_TRANSACTION_LONGITUDE
        default_used = True

    if browser_geo.get('city') and browser_geo.get('country'):
        source = 'browser_geo'
    elif ip_city and ip_country:
        source = 'ip_geo'
    elif profile_used:
        source = 'profile_fallback'
    else:
        source = 'default_fallback'

    if not state and source == 'default_fallback':
        state = DEFAULT_TRANSACTION_STATE

    if browser_requested and browser_lat is not None and browser_lon is not None:
        coords_source = 'browser_geo'
    elif to_float(ip_risk.get('ip_latitude')) is not None and to_float(ip_risk.get('ip_longitude')) is not None:
        coords_source = 'ip_geo'
    else:
        coords_source = 'default_fallback'

    return {
        'ip_country': country,
        'ip_state': state,
        'ip_city': city,
        'latitude': latitude,
        'longitude': longitude,
        'source': source,
        'coords_source': coords_source,
    }


def is_known_device(cursor, user_id, device_fingerprint):
    cursor.execute(
        'SELECT 1 FROM devices WHERE user_id = %s AND device_fingerprint = %s LIMIT 1',
        (user_id, device_fingerprint)
    )
    return cursor.fetchone() is not None


def compute_device_trust_score(known_device, foreign_transaction, location_mismatch, velocity_last_24h, impossible_travel):
    score = 90 if known_device else 45
    if foreign_transaction:
        score -= 20
    if location_mismatch:
        score -= 20
    if impossible_travel:
        score -= 20
    score -= min(velocity_last_24h * 2, 15)
    return max(0, min(100, score))


def build_feature_vector(context):
    legacy_features = [
        float(context['amount']),
        int(context['transaction_hour']),
        int(context['foreign_transaction']),
        int(context['location_mismatch']),
        int(context['device_trust_score']),
        int(context['velocity_last_24h']),
        int(context['cardholder_age']),
    ]

    advanced_features = [
        float(context['amount']),
        int(context['transaction_hour']),
        int(context['transaction_day_of_week']),
        int(context['foreign_transaction']),
        int(context['vpn_proxy_detected']),
        int(context['location_mismatch']),
        int(context['device_trust_score']),
        int(context['velocity_last_1h']),
        int(context['velocity_last_24h']),
        int(context['cardholder_age']),
    ]

    with MODEL_LOCK:
        local_scaler = scaler
    expected_total = int(getattr(local_scaler, 'n_features_in_', len(legacy_features) + len(ENCODED_CATEGORIES)))
    expected_numeric_count = expected_total - len(ENCODED_CATEGORIES)
    features = advanced_features if expected_numeric_count == len(advanced_features) else legacy_features

    for category in ENCODED_CATEGORIES:
        features.append(1 if context['merchant_category'] == category else 0)

    return np.array(features, dtype=float).reshape(1, -1)


def make_prediction(feature_vector):
    with MODEL_LOCK:
        local_model = model
        local_scaler = scaler
        loaded = MODEL_LOADED

    if not loaded or local_model is None or local_scaler is None:
        raise RuntimeError('Model artifacts are not loaded')

    scaled = local_scaler.transform(feature_vector)
    fraud_probability = float(local_model.predict_proba(scaled)[0][1])

    if fraud_probability < 0.30:
        risk_level = 'LOW'
        decision = 'APPROVE'
    elif fraud_probability < 0.70:
        risk_level = 'MEDIUM'
        decision = 'REVIEW'
    else:
        risk_level = 'HIGH'
        decision = 'BLOCK'

    return {
        'fraud_probability': fraud_probability,
        'risk_level': risk_level,
        'decision': decision,
        'prediction': 'FRAUDULENT' if fraud_probability >= 0.5 else 'LEGITIMATE'
    }


def apply_rule_adjustments(prediction, context):
    adjusted_probability = prediction['fraud_probability']
    if context['vpn_proxy_detected']:
        adjusted_probability += 0.08
    if context['velocity_last_1h'] >= 3:
        adjusted_probability += 0.05
    if context['transaction_day_of_week'] in (5, 6) and context['amount_deviation_ratio'] >= 2.5:
        adjusted_probability += 0.04

    adjusted_probability = max(0.0, min(1.0, adjusted_probability))
    if adjusted_probability < 0.30:
        risk_level = 'LOW'
        decision = 'APPROVE'
    elif adjusted_probability < 0.70:
        risk_level = 'MEDIUM'
        decision = 'REVIEW'
    else:
        risk_level = 'HIGH'
        decision = 'BLOCK'

    prediction['fraud_probability'] = adjusted_probability
    prediction['risk_level'] = risk_level
    prediction['decision'] = decision
    prediction['prediction'] = 'FRAUDULENT' if adjusted_probability >= 0.5 else 'LEGITIMATE'
    return prediction


def build_risk_factors(context):
    risk_factors = []
    if context['foreign_transaction']:
        risk_factors.append('Foreign transaction detected')
    if context['location_mismatch']:
        risk_factors.append('Transaction far from registered location')
    if context['impossible_travel']:
        risk_factors.append('Impossible travel pattern detected')
    if context['velocity_last_24h'] >= 5:
        risk_factors.append('High recent transaction velocity')
    if context['velocity_last_1h'] >= 3:
        risk_factors.append('High transaction burst in the last hour')
    if context['vpn_proxy_detected']:
        risk_factors.append('VPN or proxy network detected')
    if context['device_trust_score'] <= 40:
        risk_factors.append('Low device trust score')
    if context['amount_deviation_ratio'] >= 2.5:
        risk_factors.append('Amount deviates significantly from recent average')
    if context['amount'] >= HIGH_AMOUNT_THRESHOLD:
        risk_factors.append('High-value transaction alert')
    return risk_factors


def log_system_event(connection, level, message, transaction_id=None, user_id=None, metadata=None):
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                '''
                INSERT INTO system_logs (log_level, message, transaction_id, user_id, metadata)
                VALUES (%s, %s, %s, %s, %s)
                ''',
                (level, message, transaction_id, user_id, Json(metadata or {}))
            )
    except Exception as exc:
        logger.warning('Failed to persist log event: %s', exc)


def validate_account_number(account_number):
    normalized = ''.join(ch for ch in str(account_number or '') if ch.isdigit())
    return normalized if len(normalized) == 10 else None


def build_fake_beneficiary(account_number):
    normalized = validate_account_number(account_number)
    if not normalized:
        return None

    digest = hashlib.sha256(normalized.encode('utf-8')).hexdigest()
    first_name = FAKE_FIRST_NAMES[int(digest[0:2], 16) % len(FAKE_FIRST_NAMES)]
    last_name = FAKE_LAST_NAMES[int(digest[2:4], 16) % len(FAKE_LAST_NAMES)]
    bank_name = FAKE_BANK_NAMES[int(digest[4:6], 16) % len(FAKE_BANK_NAMES)]

    return {
        'account_number': normalized,
        'account_name': f'{first_name} {last_name}',
        'bank_name': bank_name,
    }


def run_transaction_scoring(payload, request_user_agent=None, request_ip=None):
    errors = validate_transaction_input(payload)
    if errors:
        raise RequestError({'success': False, 'errors': errors}, 400)

    started_at = time.perf_counter()
    connection = None

    try:
        connection = get_db_connection()
        user_agent = payload.get('user_agent') or request_user_agent or 'Unknown'
        ip_address = payload.get('ip_address') or request_ip or '127.0.0.1'
        device_fingerprint = get_device_fingerprint(payload, ip_address, user_agent)
        merchant_category = normalize_merchant_category(payload['merchant_category'])
        transaction_time = normalize_timestamp(payload.get('timestamp'))

        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            user = fetch_user(cursor, payload['user_id'])
            if not user:
                raise RequestError({'success': False, 'error': 'User not found'}, 404)

            stats = fetch_user_stats(cursor, payload['user_id'], transaction_time)
            known_device = is_known_device(cursor, payload['user_id'], device_fingerprint)
            ip_risk = lookup_ip_risk(ip_address)
            location = resolve_transaction_location(payload, user, ip_risk)

            registered_latitude = to_float(user.get('registered_latitude'))
            registered_longitude = to_float(user.get('registered_longitude'))
            transaction_latitude = to_float(location.get('latitude'))
            transaction_longitude = to_float(location.get('longitude'))
            distance_from_home_km = haversine_km(
                registered_latitude,
                registered_longitude,
                transaction_latitude,
                transaction_longitude,
            )

            previous_transaction = stats['previous_transaction']
            impossible_travel = False
            if previous_transaction and transaction_latitude is not None and transaction_longitude is not None:
                previous_lat = to_float(previous_transaction.get('latitude'))
                previous_lon = to_float(previous_transaction.get('longitude'))
                previous_ts = previous_transaction.get('timestamp')
                travel_distance = haversine_km(previous_lat, previous_lon, transaction_latitude, transaction_longitude)
                if travel_distance is not None and previous_ts:
                    elapsed_hours = max((transaction_time.replace(tzinfo=None) - previous_ts).total_seconds() / 3600, 0.01)
                    impossible_travel = (travel_distance / elapsed_hours) > 900

            registered_country = (user.get('registered_country') or '').strip().lower()
            supplied_country = location['ip_country'].strip().lower()
            foreign_transaction = int(bool(supplied_country and registered_country and supplied_country != registered_country))
            location_mismatch = int(bool(distance_from_home_km is not None and distance_from_home_km > 50) or impossible_travel)
            velocity_last_1h = stats['velocity_last_1h'] + 1
            velocity_last_24h = stats['velocity_last_24h'] + 1
            average_amount = stats['average_amount_last_24h']
            amount = float(payload['amount'])
            amount_deviation_ratio = (amount / average_amount) if average_amount > 0 else (2.5 if amount >= HIGH_AMOUNT_THRESHOLD else 1.0)
            cardholder_age = payload.get('cardholder_age') or user.get('cardholder_age') or 30
            transaction_day_of_week = transaction_time.weekday()
            vpn_proxy_detected = int(ip_risk.get('is_proxy_or_vpn', False))
            device_trust_score = payload.get('device_trust_score')
            if device_trust_score in (None, ''):
                device_trust_score = compute_device_trust_score(
                    known_device,
                    foreign_transaction,
                    location_mismatch,
                    velocity_last_24h,
                    impossible_travel,
                )
            else:
                device_trust_score = int(device_trust_score)

            feature_context = {
                'amount': amount,
                'transaction_hour': transaction_time.hour,
                'transaction_day_of_week': transaction_day_of_week,
                'foreign_transaction': foreign_transaction,
                'vpn_proxy_detected': vpn_proxy_detected,
                'location_mismatch': location_mismatch,
                'device_trust_score': device_trust_score,
                'velocity_last_1h': velocity_last_1h,
                'velocity_last_24h': velocity_last_24h,
                'cardholder_age': int(cardholder_age),
                'merchant_category': merchant_category,
                'impossible_travel': impossible_travel,
                'amount_deviation_ratio': amount_deviation_ratio,
            }

            prediction = apply_rule_adjustments(make_prediction(build_feature_vector(feature_context)), feature_context)
            risk_factors = build_risk_factors(feature_context)
            transaction_id = str(uuid.uuid4())[:16]
            device_id = hashlib.sha256(f'{payload["user_id"]}|{device_fingerprint}'.encode('utf-8')).hexdigest()[:16]
            ip_country = location['ip_country']
            ip_state = location['ip_state']
            ip_city = location['ip_city']

            cursor.execute(
                '''
                INSERT INTO transactions (
                    transaction_id, user_id, amount, currency, merchant_name, merchant_category,
                    timestamp, latitude, longitude, ip_address, ip_country, ip_city,
                    device_fingerprint, user_agent, prediction, confidence_score,
                    risk_level, status
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ''',
                (
                    transaction_id,
                    payload['user_id'],
                    amount,
                    'NGN',
                    payload['merchant_name'].strip(),
                    merchant_category,
                    transaction_time.replace(tzinfo=None),
                    transaction_latitude,
                    transaction_longitude,
                    encrypt_value(ip_address),
                    ip_country,
                    ip_city,
                    device_fingerprint,
                    encrypt_value(user_agent),
                    prediction['prediction'],
                    prediction['fraud_probability'],
                    prediction['risk_level'],
                    prediction['decision'],
                )
            )
            cursor.execute(
                '''
                INSERT INTO fraud_predictions (
                    transaction_id, prediction, confidence_score, risk_factors, feature_context, training_label
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                ''',
                (
                    transaction_id,
                    prediction['prediction'],
                    prediction['fraud_probability'],
                    Json(risk_factors),
                    Json({
                        'amount': feature_context['amount'],
                        'transaction_hour': feature_context['transaction_hour'],
                        'transaction_day_of_week': feature_context['transaction_day_of_week'],
                        'foreign_transaction': feature_context['foreign_transaction'],
                        'vpn_proxy_detected': feature_context['vpn_proxy_detected'],
                        'location_mismatch': feature_context['location_mismatch'],
                        'device_trust_score': feature_context['device_trust_score'],
                        'velocity_last_1h': feature_context['velocity_last_1h'],
                        'velocity_last_24h': feature_context['velocity_last_24h'],
                        'cardholder_age': feature_context['cardholder_age'],
                        'merchant_category': feature_context['merchant_category'],
                    }),
                    1 if prediction['prediction'] == 'FRAUDULENT' else 0,
                )
            )
            cursor.execute(
                '''
                INSERT INTO devices (
                    device_id, user_id, device_fingerprint, device_type, browser,
                    operating_system, last_seen, total_transactions
                )
                VALUES (%s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP, 1)
                ON CONFLICT (user_id, device_fingerprint)
                DO UPDATE SET
                    last_seen = CURRENT_TIMESTAMP,
                    total_transactions = devices.total_transactions + 1,
                    browser = EXCLUDED.browser,
                    operating_system = EXCLUDED.operating_system,
                    device_type = EXCLUDED.device_type
                ''',
                (
                    device_id,
                    payload['user_id'],
                    device_fingerprint,
                    derive_device_type(user_agent),
                    derive_browser(user_agent),
                    derive_os(user_agent),
                )
            )
            cursor.execute(
                'UPDATE users SET total_transactions = total_transactions + 1 WHERE user_id = %s',
                (payload['user_id'],)
            )
            log_system_event(
                connection,
                'INFO',
                'Transaction scored',
                transaction_id=transaction_id,
                user_id=payload['user_id'],
                metadata={
                    'decision': prediction['decision'],
                    'risk_level': prediction['risk_level'],
                    'risk_factors': risk_factors,
                    'vpn_proxy_detected': bool(vpn_proxy_detected),
                    'velocity_last_1h': velocity_last_1h,
                    'merchant_name': payload['merchant_name'],
                    'transaction_city': ip_city,
                    'transaction_state': ip_state,
                    'transaction_country': ip_country,
                    'location_source': location['source'],
                    'coordinates_source': location['coords_source'],
                },
            )

        connection.commit()
        queue_retrain()
        processing_time_ms = (time.perf_counter() - started_at) * 1000
        return {
            'success': True,
            'transaction_id': transaction_id,
            'decision': prediction['decision'],
            'risk_level': prediction['risk_level'],
            'fraud_probability': prediction['fraud_probability'],
            'processing_time_ms': round(processing_time_ms, 2),
            'device_id': device_id,
            'risk_factors': risk_factors,
            'distance_from_home_km': round(distance_from_home_km, 2) if distance_from_home_km is not None else None,
            'velocity_last_1h': velocity_last_1h,
            'velocity_last_24h': velocity_last_24h,
            'known_device': known_device,
            'transaction_city': ip_city,
            'transaction_state': ip_state,
            'transaction_country': ip_country,
            'location_source': location['source'],
            'coordinates_source': location['coords_source'],
        }
    except RequestError:
        if connection:
            connection.rollback()
        raise
    except Exception:
        if connection:
            connection.rollback()
        raise
    finally:
        release_db_connection(connection)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/customer')
def customer_app():
    return render_template('customer.html')


@app.route('/api/auth/login', methods=['POST'])
def analyst_login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = (payload.get('password') or '').strip()
    password_hash = get_password_hash()

    if not username or not password:
        return jsonify({'success': False, 'error': 'username and password are required'}), 400

    if username != ANALYST_USERNAME:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    password_ok = False
    if password_hash is not None:
        password_ok = bcrypt.checkpw(password.encode('utf-8'), password_hash)
    elif ANALYST_API_KEY:
        password_ok = secrets.compare_digest(password, ANALYST_API_KEY)

    if not password_ok:
        return jsonify({'success': False, 'error': 'Invalid credentials'}), 401

    token = create_access_token(username)
    return jsonify({
        'success': True,
        'token': token,
        'token_type': 'Bearer',
        'expires_in_hours': JWT_EXPIRY_HOURS,
    }), 200


@app.route('/api/auth/validate', methods=['GET'])
@require_analyst_auth
def validate_auth():
    return jsonify({'success': True}), 200


@app.route('/api/register', methods=['POST'])
def register_user():
    payload = request.get_json(silent=True) or {}
    errors = validate_registration_input(payload)
    if errors:
        return jsonify({'success': False, 'errors': errors}), 400

    geo = geocode_address(payload['address'])
    resolved_city = geo.get('city') or infer_city_from_address(payload.get('address'))
    user_id = str(uuid.uuid4())[:8]
    user_agent = request.headers.get('User-Agent', 'Unknown')
    normalized_email = payload['email'].strip().lower()
    email_fingerprint = email_hash(normalized_email)
    connection = None

    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                '''
                INSERT INTO users (
                    user_id, username, email, phone_number, registered_address,
                    registered_country, registered_city, registered_latitude,
                    registered_longitude, cardholder_age, registration_ip,
                    registration_user_agent, email_hash
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING *
                ''',
                (
                    user_id,
                    payload['username'].strip(),
                    encrypt_value(normalized_email),
                    encrypt_value(payload['phone_number'].strip()),
                    encrypt_value(payload['address'].strip()),
                    geo.get('country'),
                    resolved_city,
                    geo.get('latitude'),
                    geo.get('longitude'),
                    int(payload['cardholder_age']) if payload.get('cardholder_age') not in (None, '') else None,
                    encrypt_value(get_client_ip()),
                    encrypt_value(user_agent),
                    email_fingerprint,
                )
            )
            user = cursor.fetchone()
            log_system_event(connection, 'INFO', 'User registered', user_id=user_id, metadata={'geocoded': bool(geo)})
        connection.commit()
    except psycopg2.IntegrityError as exc:
        if connection:
            connection.rollback()
        message = 'Username or email already exists'
        logger.warning('Registration conflict: %s', exc)
        return jsonify({'success': False, 'error': message}), 409
    except Exception as exc:
        if connection:
            connection.rollback()
        logger.exception('Registration error: %s', exc)
        return jsonify({'success': False, 'error': 'Registration failed'}), 500
    finally:
        release_db_connection(connection)

    serialized_user = serialize_row(user)
    return jsonify({
        'success': True,
        'user_id': user_id,
        'message': 'User registered successfully',
        'user': serialized_user,
        'geocoded': bool(geo),
        'city_resolution': 'geocoder' if geo.get('city') else ('address_fallback' if resolved_city else 'unresolved')
    }), 201


@app.route('/api/users', methods=['GET'])
@require_analyst_auth
def get_users():
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                '''
                SELECT user_id, username, email, registered_city, registered_country,
                       total_transactions, account_creation_date
                FROM users
                ORDER BY account_creation_date DESC
                '''
            )
            users = [serialize_row(row) for row in cursor.fetchall()]
        return jsonify({'users': users, 'total_count': len(users)}), 200
    except Exception as exc:
        logger.exception('Get users error: %s', exc)
        return jsonify({'error': 'Failed to fetch users'}), 500
    finally:
        release_db_connection(connection)


@app.route('/api/predict', methods=['POST'])
def predict_fraud():
    payload = request.get_json(silent=True) or {}
    try:
        response = run_transaction_scoring(payload, request.headers.get('User-Agent', 'Unknown'), get_client_ip())
        return jsonify(response), 200
    except RequestError as exc:
        return jsonify(exc.payload), exc.status_code
    except Exception as exc:
        logger.exception('Prediction error: %s', exc)
        return jsonify({'success': False, 'error': 'Prediction failed'}), 500


@app.route('/api/customer/account-lookup', methods=['GET'])
def customer_account_lookup():
    account_number = request.args.get('account_number', '')
    beneficiary = build_fake_beneficiary(account_number)
    if not beneficiary:
        return jsonify({'success': False, 'error': 'Account number must be 10 digits'}), 400
    return jsonify({'success': True, **beneficiary}), 200


@app.route('/api/customer/users', methods=['GET'])
def customer_users_list():
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                '''
                SELECT user_id, username, registered_city, registered_country, total_transactions, account_creation_date
                FROM users
                ORDER BY account_creation_date DESC
                LIMIT 100
                '''
            )
            users = [serialize_row(row) for row in cursor.fetchall()]
        return jsonify({'success': True, 'users': users}), 200
    except Exception as exc:
        logger.exception('Customer list error: %s', exc)
        return jsonify({'success': False, 'error': 'Failed to load customers'}), 500
    finally:
        release_db_connection(connection)


@app.route('/api/customer/location-preview', methods=['GET'])
def customer_location_preview():
    ip_address = get_client_ip()
    ip_risk = lookup_ip_risk(ip_address)
    return jsonify({
        'success': True,
        'ip_address': ip_address,
        'city': ip_risk.get('ip_city') or DEFAULT_TRANSACTION_CITY,
        'state': ip_risk.get('ip_state') or DEFAULT_TRANSACTION_STATE,
        'country': ip_risk.get('ip_country') or DEFAULT_TRANSACTION_COUNTRY,
        'latitude': ip_risk.get('ip_latitude') or DEFAULT_TRANSACTION_LATITUDE,
        'longitude': ip_risk.get('ip_longitude') or DEFAULT_TRANSACTION_LONGITUDE,
        'location_source': 'ip' if ip_risk.get('ip_city') else 'default',
    }), 200


@app.route('/api/customer/users/<user_id>', methods=['GET'])
def customer_user_profile(user_id):
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                '''
                SELECT user_id, username, registered_city, registered_country, total_transactions, account_creation_date
                FROM users
                WHERE user_id = %s
                ''',
                (user_id,)
            )
            user = serialize_row(cursor.fetchone())
        if not user:
            return jsonify({'success': False, 'error': 'User not found'}), 404
        return jsonify({'success': True, 'user': user}), 200
    except Exception as exc:
        logger.exception('Customer profile error: %s', exc)
        return jsonify({'success': False, 'error': 'Failed to load customer profile'}), 500
    finally:
        release_db_connection(connection)


@app.route('/api/customer/users/<user_id>/transactions', methods=['GET'])
def customer_user_transactions(user_id):
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                '''
                SELECT transaction_id, amount, currency, merchant_name, merchant_category,
                       timestamp, risk_level, status, confidence_score
                FROM transactions
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT 10
                ''',
                (user_id,)
            )
            transactions = [serialize_row(row) for row in cursor.fetchall()]
        return jsonify({'success': True, 'transactions': transactions}), 200
    except Exception as exc:
        logger.exception('Customer transaction history error: %s', exc)
        return jsonify({'success': False, 'error': 'Failed to load customer transactions'}), 500
    finally:
        release_db_connection(connection)


@app.route('/api/customer/transfer', methods=['POST'])
def customer_transfer():
    payload = request.get_json(silent=True) or {}
    user_id = (payload.get('user_id') or '').strip()
    amount = payload.get('amount')
    account_number = payload.get('recipient_account_number')

    if not user_id:
        return jsonify({'success': False, 'error': 'user_id is required'}), 400

    geolocation_source = (payload.get('geolocation_source') or '').strip().lower()
    latitude = to_float(payload.get('latitude'))
    longitude = to_float(payload.get('longitude'))
    if geolocation_source != 'browser' or latitude is None or longitude is None:
        return jsonify({
            'success': False,
            'error': 'Current live location is required. Enable browser location and retry.',
            'code': 'LIVE_LOCATION_REQUIRED',
        }), 400

    beneficiary = build_fake_beneficiary(account_number)
    if not beneficiary:
        return jsonify({'success': False, 'error': 'Recipient account number must be 10 digits'}), 400

    transfer_payload = {
        'user_id': user_id,
        'amount': amount,
        'currency': 'NGN',
        'merchant_name': f"Transfer to {beneficiary['account_name']}",
        'merchant_category': TRANSFER_MERCHANT_CATEGORY,
        'timestamp': payload.get('timestamp') or datetime.now(timezone.utc).isoformat(),
        'device_fingerprint': payload.get('device_fingerprint'),
        'latitude': latitude,
        'longitude': longitude,
        'geolocation_source': geolocation_source,
        'user_agent': payload.get('user_agent') or 'Customer Banking Simulator',
    }

    try:
        result = run_transaction_scoring(transfer_payload, request.headers.get('User-Agent', 'Unknown'), get_client_ip())
        result['recipient'] = beneficiary
        result['narration'] = (payload.get('narration') or '').strip()
        result['transfer_channel'] = 'bank-transfer-simulator'
        return jsonify(result), 200
    except RequestError as exc:
        return jsonify(exc.payload), exc.status_code
    except Exception as exc:
        logger.exception('Customer transfer error: %s', exc)
        return jsonify({'success': False, 'error': 'Transfer failed'}), 500


@app.route('/api/dashboard-data', methods=['GET'])
@require_analyst_auth
def get_dashboard_data():
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                '''
                SELECT
                    COUNT(*) AS total_transactions,
                    COUNT(*) FILTER (WHERE prediction = 'FRAUDULENT') AS fraud_count,
                    COUNT(*) FILTER (WHERE status IN ('REVIEW', 'BLOCK')) AS review_queue_count,
                    COUNT(DISTINCT user_id) AS active_users,
                    COALESCE(AVG(confidence_score), 0) AS average_fraud_probability,
                    COALESCE(SUM(amount) FILTER (WHERE status = 'BLOCK'), 0) AS blocked_amount
                FROM transactions
                '''
            )
            summary = serialize_row(cursor.fetchone())

            cursor.execute(
                'SELECT risk_level, COUNT(*) AS count FROM transactions GROUP BY risk_level ORDER BY risk_level'
            )
            risk_distribution = {row['risk_level']: int(row['count']) for row in cursor.fetchall() if row['risk_level']}

            cursor.execute(
                'SELECT status, COUNT(*) AS count FROM transactions GROUP BY status ORDER BY status'
            )
            decision_distribution = {row['status']: int(row['count']) for row in cursor.fetchall() if row['status']}

            cursor.execute(
                '''
                SELECT t.transaction_id, t.user_id, u.username, t.amount, t.currency, t.merchant_name,
                       t.merchant_category, t.timestamp, t.risk_level, t.status, t.confidence_score,
                       fp.risk_factors
                FROM transactions t
                JOIN users u ON u.user_id = t.user_id
                LEFT JOIN fraud_predictions fp ON fp.transaction_id = t.transaction_id
                ORDER BY t.timestamp DESC
                LIMIT 10
                '''
            )
            recent_transactions = [serialize_row(row) for row in cursor.fetchall()]

            cursor.execute(
                '''
                SELECT t.transaction_id, t.user_id, u.username, t.amount, t.merchant_name,
                       t.timestamp, t.risk_level, t.status, fp.risk_factors
                FROM transactions t
                JOIN users u ON u.user_id = t.user_id
                LEFT JOIN fraud_predictions fp ON fp.transaction_id = t.transaction_id
                WHERE t.status IN ('REVIEW', 'BLOCK')
                ORDER BY t.timestamp DESC
                LIMIT 10
                '''
            )
            flagged_transactions = [serialize_row(row) for row in cursor.fetchall()]

            cursor.execute(
                '''
                SELECT t.transaction_id, t.user_id, u.username, t.amount, t.currency,
                       t.merchant_name, t.timestamp, t.status
                FROM transactions t
                JOIN users u ON u.user_id = t.user_id
                WHERE t.amount >= %s
                ORDER BY t.timestamp DESC
                LIMIT 10
                ''',
                (HIGH_AMOUNT_THRESHOLD,)
            )
            high_amount_alerts = [serialize_row(row) for row in cursor.fetchall()]

            cursor.execute('SELECT COUNT(*) AS total_users FROM users')
            total_users = int(cursor.fetchone()['total_users'])

        total_transactions = summary.get('total_transactions', 0) or 0
        fraud_count = summary.get('fraud_count', 0) or 0
        fraud_rate = round((fraud_count / total_transactions) * 100, 2) if total_transactions else 0

        return jsonify({
            'total_transactions': int(total_transactions),
            'fraud_count': int(fraud_count),
            'fraud_rate': fraud_rate,
            'average_fraud_probability': round(float(summary.get('average_fraud_probability', 0) or 0), 4),
            'review_queue_count': int(summary.get('review_queue_count', 0) or 0),
            'active_users': int(summary.get('active_users', 0) or 0),
            'registered_users': total_users,
            'blocked_amount': round(float(summary.get('blocked_amount', 0) or 0), 2),
            'risk_distribution': risk_distribution,
            'decision_distribution': decision_distribution,
            'recent_transactions': recent_transactions,
            'flagged_transactions': flagged_transactions,
            'high_amount_alerts': high_amount_alerts,
            'merchant_categories': MODEL_CATEGORIES,
        }), 200
    except Exception as exc:
        logger.exception('Dashboard data error: %s', exc)
        return jsonify({'error': 'Failed to load dashboard data'}), 500
    finally:
        release_db_connection(connection)


@app.route('/api/transactions', methods=['GET'])
@require_analyst_auth
def get_transactions():
    risk_level = request.args.get('risk_level')
    decision = request.args.get('decision')
    user_id = request.args.get('user_id')
    limit = min(int(request.args.get('limit', 50)), 200)

    clauses = []
    params = []
    if risk_level:
        clauses.append('t.risk_level = %s')
        params.append(risk_level.upper())
    if decision:
        clauses.append('t.status = %s')
        params.append(decision.upper())
    if user_id:
        clauses.append('t.user_id = %s')
        params.append(user_id)

    where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ''
    query = f'''
        SELECT t.transaction_id, t.user_id, u.username, t.amount, t.currency,
               t.merchant_name, t.merchant_category, t.timestamp, t.ip_country,
               t.ip_city, t.risk_level, t.status, t.confidence_score
        FROM transactions t
        JOIN users u ON u.user_id = t.user_id
        {where_clause}
        ORDER BY t.timestamp DESC
        LIMIT %s
    '''
    params.append(limit)

    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            transactions = [serialize_row(row) for row in cursor.fetchall()]
        return jsonify({'transactions': transactions, 'total_count': len(transactions)}), 200
    except Exception as exc:
        logger.exception('Get transactions error: %s', exc)
        return jsonify({'error': 'Failed to fetch transactions'}), 500
    finally:
        release_db_connection(connection)


@app.route('/api/users/<user_id>/transactions', methods=['GET'])
@require_analyst_auth
def get_user_transactions(user_id):
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                '''
                SELECT transaction_id, amount, currency, merchant_name, merchant_category,
                       timestamp, risk_level, status, confidence_score
                FROM transactions
                WHERE user_id = %s
                ORDER BY timestamp DESC
                LIMIT 25
                ''',
                (user_id,)
            )
            transactions = [serialize_row(row) for row in cursor.fetchall()]
        return jsonify({'transactions': transactions, 'total_count': len(transactions)}), 200
    except Exception as exc:
        logger.exception('User transaction history error: %s', exc)
        return jsonify({'error': 'Failed to fetch user transaction history'}), 500
    finally:
        release_db_connection(connection)


@app.route('/api/users/<user_id>', methods=['DELETE'])
@require_analyst_auth
def delete_user(user_id):
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute('SELECT user_id, username FROM users WHERE user_id = %s', (user_id,))
            existing_user = cursor.fetchone()
            if not existing_user:
                return jsonify({'success': False, 'error': 'User not found'}), 404

            cursor.execute(
                '''
                DELETE FROM fraud_predictions
                WHERE transaction_id IN (
                    SELECT transaction_id FROM transactions WHERE user_id = %s
                )
                ''',
                (user_id,)
            )
            deleted_predictions = cursor.rowcount

            cursor.execute('DELETE FROM devices WHERE user_id = %s', (user_id,))
            deleted_devices = cursor.rowcount

            cursor.execute('DELETE FROM system_logs WHERE user_id = %s', (user_id,))
            deleted_logs = cursor.rowcount

            cursor.execute('DELETE FROM transactions WHERE user_id = %s', (user_id,))
            deleted_transactions = cursor.rowcount

            cursor.execute('DELETE FROM users WHERE user_id = %s', (user_id,))

            log_system_event(
                connection,
                'INFO',
                'User deleted',
                user_id=user_id,
                metadata={
                    'username': existing_user.get('username'),
                    'deleted_transactions': deleted_transactions,
                    'deleted_predictions': deleted_predictions,
                    'deleted_devices': deleted_devices,
                    'deleted_logs': deleted_logs,
                },
            )

        connection.commit()
        return jsonify({
            'success': True,
            'message': 'User deleted successfully',
            'user_id': user_id,
            'deleted_counts': {
                'transactions': deleted_transactions,
                'predictions': deleted_predictions,
                'devices': deleted_devices,
                'logs': deleted_logs,
            }
        }), 200
    except Exception as exc:
        if connection:
            connection.rollback()
        logger.exception('Delete user error: %s', exc)
        return jsonify({'success': False, 'error': 'Failed to delete user'}), 500
    finally:
        release_db_connection(connection)


@app.route('/api/review-queue', methods=['GET'])
@require_analyst_auth
def get_review_queue():
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                '''
                SELECT t.transaction_id, t.user_id, u.username, t.amount, t.currency,
                       t.merchant_name, t.timestamp, t.risk_level, t.status,
                       fp.risk_factors
                FROM transactions t
                JOIN users u ON u.user_id = t.user_id
                LEFT JOIN fraud_predictions fp ON fp.transaction_id = t.transaction_id
                WHERE t.status IN ('REVIEW', 'BLOCK')
                ORDER BY t.timestamp DESC
                LIMIT 25
                '''
            )
            transactions = [serialize_row(row) for row in cursor.fetchall()]
        return jsonify({'transactions': transactions, 'total_count': len(transactions)}), 200
    except Exception as exc:
        logger.exception('Review queue error: %s', exc)
        return jsonify({'error': 'Failed to fetch review queue'}), 500
    finally:
        release_db_connection(connection)


@app.route('/api/logs', methods=['GET'])
@require_analyst_auth
def get_logs():
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                '''
                SELECT timestamp, log_level, message, transaction_id, user_id, metadata
                FROM system_logs
                ORDER BY timestamp DESC
                LIMIT 50
                '''
            )
            logs = [serialize_row(row) for row in cursor.fetchall()]
        return jsonify({'logs': logs, 'total_count': len(logs)}), 200
    except Exception as exc:
        logger.exception('Log fetch error: %s', exc)
        return jsonify({'error': 'Failed to fetch logs'}), 500
    finally:
        release_db_connection(connection)


@app.route('/api/device-info', methods=['GET'])
@require_analyst_auth
def get_device_info():
    ip_address = get_client_ip()
    user_agent = request.headers.get('User-Agent', 'Unknown')
    device_fingerprint = get_device_fingerprint({}, ip_address, user_agent)
    return jsonify({
        'ip_address': ip_address,
        'user_agent': user_agent,
        'device_id': hashlib.sha256(f'{ip_address}|{user_agent}'.encode('utf-8')).hexdigest()[:16],
        'device_fingerprint': device_fingerprint,
        'merchant_categories': MODEL_CATEGORIES,
    }), 200


@app.route('/api/health', methods=['GET'])
def health_check():
    database_status = 'unhealthy'
    connection = None
    try:
        connection = get_db_connection()
        with connection.cursor() as cursor:
            cursor.execute('SELECT 1')
            cursor.fetchone()
        database_status = 'healthy'
    except Exception as exc:
        logger.warning('Health check database failure: %s', exc)
    finally:
        release_db_connection(connection)

    status = 'healthy' if MODEL_LOADED and database_status == 'healthy' else 'degraded'
    return jsonify({
        'status': status,
        'database': database_status,
        'model_loaded': MODEL_LOADED,
        'merchant_categories': MODEL_CATEGORIES,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }), 200 if status == 'healthy' else 503


@app.errorhandler(404)
def not_found(_error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(_error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    init_db_pool()
    app.run(debug=True, host='0.0.0.0', port=5000)

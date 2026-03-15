-- Create Users Table
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
);

-- Create Transactions Table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50),
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
);

-- Create Devices Table
CREATE TABLE IF NOT EXISTS devices (
    device_id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50),
    device_fingerprint VARCHAR(255) NOT NULL,
    device_type VARCHAR(50),
    browser VARCHAR(100),
    operating_system VARCHAR(100),
    first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    total_transactions INTEGER DEFAULT 0,
    UNIQUE(user_id, device_fingerprint),
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Create Fraud Predictions Table
CREATE TABLE IF NOT EXISTS fraud_predictions (
    prediction_id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50),
    prediction VARCHAR(20),
    confidence_score DECIMAL(5, 4),
    risk_factors JSONB,
    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id)
);

-- Create System Logs Table
CREATE TABLE IF NOT EXISTS system_logs (
    log_id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    log_level VARCHAR(20),
    message TEXT,
    transaction_id VARCHAR(50),
    user_id VARCHAR(50),
    metadata JSONB
);

-- Create indices for performance
CREATE INDEX idx_transactions_user_id ON transactions(user_id);
CREATE INDEX idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX idx_transactions_risk_level ON transactions(risk_level);
CREATE INDEX idx_devices_user_id ON devices(user_id);
CREATE INDEX idx_fraud_predictions_timestamp ON fraud_predictions(prediction_timestamp);
CREATE INDEX idx_system_logs_timestamp ON system_logs(timestamp);
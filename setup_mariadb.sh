#!/bin/bash
# Setup MariaDB and load data

set -e

echo "=== Step 1: Install Python mariadb connector ==="
conda run -n fraud_mlops pip install mariadb -q
echo "✓ Python mariadb package installed"

echo "" 
echo "=== Step 2: Create database and tables ==="
sudo mysql -u root <<'SQL'
CREATE DATABASE IF NOT EXISTS fraud_mlops;
USE fraud_mlops;

CREATE TABLE IF NOT EXISTS processed_transactions (
    transaction_id BIGINT NOT NULL AUTO_INCREMENT,
    step INT,
    type VARCHAR(50),
    branch VARCHAR(100),
    amount DECIMAL(18,2),
    nameOrig VARCHAR(100),
    oldbalanceOrg DECIMAL(18,2),
    newbalanceOrig DECIMAL(18,2),
    nameDest VARCHAR(100),
    oldbalanceDest DECIMAL(18,2),
    newbalanceDest DECIMAL(18,2),
    unusuallogin INT,
    isFlaggedFraud TINYINT,
    acct_type VARCHAR(50),
    date_of_transaction VARCHAR(20),
    time_of_day VARCHAR(20),
    isFraud TINYINT,
    datetime DATETIME,
    PRIMARY KEY (transaction_id)
);

CREATE TABLE IF NOT EXISTS feature_transactions (
    transaction_id BIGINT NOT NULL,
    step INT,
    amount DECIMAL(18,2),
    oldbalanceOrg DECIMAL(18,2),
    newbalanceOrig DECIMAL(18,2),
    oldbalanceDest DECIMAL(18,2),
    newbalanceDest DECIMAL(18,2),
    balance_diff_org DECIMAL(18,2),
    balance_diff_dest DECIMAL(18,2),
    fraud_label TINYINT,
    PRIMARY KEY (transaction_id)
);
SQL
echo "✓ Database and tables created"

echo ""
echo "=== Step 3: Load processed data ==="
export DB_HOST=localhost
export DB_PORT=3306
export DB_USER=root
export DB_PASSWORD=
export DB_NAME=fraud_mlops
conda run -n fraud_mlops python src/ingestion/load_to_mariadb.py
echo "✓ Data loaded"

echo ""
echo "=== Complete! ==="

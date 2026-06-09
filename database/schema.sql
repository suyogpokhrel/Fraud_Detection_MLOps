CREATE DATABASE IF NOT EXISTS fraud_mlops;
USE fraud_mlops;

DROP TABLE IF EXISTS processed_transactions;
CREATE TABLE processed_transactions (
    id            BIGINT NOT NULL AUTO_INCREMENT,
    step          INT,
    type          VARCHAR(20),
    amount        DOUBLE,
    nameOrig      VARCHAR(20),
    oldbalanceOrg DOUBLE,
    newbalanceOrig DOUBLE,
    nameDest      VARCHAR(20),
    oldbalanceDest DOUBLE,
    newbalanceDest DOUBLE,
    isFraud       TINYINT,
    PRIMARY KEY (id)
);

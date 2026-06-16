#!/bin/bash
# Fix SQLite WAL mode and timeouts
python3 -c "import sqlite3; conn = sqlite3.connect('/opt/airflow/airflow.db'); conn.execute('PRAGMA journal_mode=WAL'); conn.close()" 2>/dev/null || true
# Fix worker timeout
sed -i 's/web_server_worker_timeout = 120/web_server_worker_timeout = 300/' /opt/airflow/airflow.cfg 2>/dev/null || true
# Create admin user if not exists / reset password
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com --password admin123 2>/dev/null || \nairflow users reset-password --username admin --password admin123 2>/dev/null || true

# Start airflow
exec airflow standalone

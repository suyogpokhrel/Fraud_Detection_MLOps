from dataclasses import dataclass
from pathlib import Path
import os


@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "3306"))
    user: str = os.getenv("DB_USER", "root")
    password: str = os.getenv("DB_PASSWORD", "")
    database: str = os.getenv("DB_NAME", "fraud_mlops")


def get_connection():
    try:
        import mariadb
    except ImportError as exc:
        raise ImportError(
            "mariadb package is required. Install it inside the fraud_mlops environment."
        ) from exc

    config = DatabaseConfig()
    return mariadb.connect(
        host=config.host,
        port=config.port,
        user=config.user,
        password=config.password,
        database=config.database,
    )

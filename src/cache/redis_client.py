import os
import redis
import pandas as pd
import json
import pickle
import logging

logger = logging.getLogger(__name__)

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL = 60 * 60 * 24  # 24 hours


def get_client():
    """Get a Redis connection. Returns None if Redis is unavailable."""
    try:
        client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, socket_timeout=3)
        client.ping()
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable: {e}. Will fall back to CSV.")
        return None


def save_dataframe(key: str, df: pd.DataFrame) -> bool:
    """Save a DataFrame to Redis. Returns True if successful."""
    client = get_client()
    if client is None:
        return False
    try:
        data = pickle.dumps(df)
        client.setex(key, REDIS_TTL, data)
        logger.info(f"Saved DataFrame to Redis: {key} ({len(df)} rows)")
        return True
    except Exception as e:
        logger.warning(f"Failed to save to Redis: {e}")
        return False


def load_dataframe(key: str) -> pd.DataFrame:
    """Load a DataFrame from Redis. Returns None if not found."""
    client = get_client()
    if client is None:
        return None
    try:
        data = client.get(key)
        if data is None:
            return None
        df = pickle.loads(data)
        logger.info(f"Loaded DataFrame from Redis: {key} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.warning(f"Failed to load from Redis: {e}")
        return None


def save_json(key: str, obj: dict) -> bool:
    """Save a dict/list to Redis as JSON."""
    client = get_client()
    if client is None:
        return False
    try:
        client.setex(key, REDIS_TTL, json.dumps(obj))
        logger.info(f"Saved JSON to Redis: {key}")
        return True
    except Exception as e:
        logger.warning(f"Failed to save JSON to Redis: {e}")
        return False


def load_json(key: str):
    """Load a dict/list from Redis. Returns None if not found."""
    client = get_client()
    if client is None:
        return None
    try:
        data = client.get(key)
        return json.loads(data) if data else None
    except Exception as e:
        logger.warning(f"Failed to load JSON from Redis: {e}")
        return None


def clear_pipeline_cache():
    """Clear all fraud pipeline keys from Redis."""
    client = get_client()
    if client is None:
        return
    try:
        keys = client.keys("fraud:*")
        if keys:
            client.delete(*keys)
            logger.info(f"Cleared {len(keys)} pipeline cache keys from Redis")
    except Exception as e:
        logger.warning(f"Failed to clear cache: {e}")

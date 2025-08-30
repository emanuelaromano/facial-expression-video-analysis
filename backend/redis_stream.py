import redis.asyncio as redis
import os
import time
import json
import logging

logger = logging.getLogger("hireview")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Initialize Redis client
try:
    r = redis.from_url(
        REDIS_URL,
        decode_responses=True,  
    )
    logger.info(f"Redis client initialized with URL: {REDIS_URL}")
except Exception as e:
    logger.error(f"Failed to initialize Redis client: {e}")
    r = None

def _status_key(uuid: str) -> str: return f"job:{uuid}:status"
def _cancel_key(uuid: str) -> str: return f"job:{uuid}:cancel"
def _channel(uuid: str) -> str: return f"job:{uuid}:ch"

async def _ensure_redis_connection():
    """Ensure Redis connection is available"""
    if r is None:
        raise ConnectionError("Redis client not initialized")
    
    try:
        await r.ping()
    except Exception as e:
        logger.error(f"Redis connection failed: {e}")
        raise ConnectionError(f"Cannot connect to Redis: {e}")

async def set_status(uuid: str, state: str, progress: int):
    try:
        await _ensure_redis_connection()
        payload = {"state": state, "progress": str(int(progress)), "ts": str(time.time())}
        await r.hset(_status_key(uuid), mapping=payload)
        await r.publish(_channel(uuid), json.dumps(payload))
        logger.debug(f"Status updated for {uuid}: {state} - {progress}%")
    except Exception as e:
        logger.error(f"Failed to set status for {uuid}: {e}")
        # Fallback to in-memory storage or raise error
        raise

async def get_status(uuid: str) -> dict:
    try:
        await _ensure_redis_connection()
        data = await r.hgetall(_status_key(uuid))
        if not data:
            return {"state": "unknown", "progress": 0}
        data["progress"] = int(float(data.get("progress", 0)))
        return data
    except Exception as e:
        logger.error(f"Failed to get status for {uuid}: {e}")
        # Return fallback status
        return {"state": "error", "progress": 0, "error": "Redis unavailable"}

async def clear_status(uuid: str):
    try:
        await _ensure_redis_connection()
        await r.delete(_status_key(uuid))
        logger.debug(f"Status cleared for {uuid}")
    except Exception as e:
        logger.error(f"Failed to clear status for {uuid}: {e}")

async def mark_cancel(uuid: str):
    try:
        await _ensure_redis_connection()
        await r.set(_cancel_key(uuid), "1", ex=3600)
        logger.debug(f"Cancel marked for {uuid}")
    except Exception as e:
        logger.error(f"Failed to mark cancel for {uuid}: {e}")

async def is_cancelled(uuid: str) -> bool:
    try:
        await _ensure_redis_connection()
        val = await r.get(_cancel_key(uuid))
        return val == "1"
    except Exception as e:
        logger.error(f"Failed to check cancel status for {uuid}: {e}")
        return False

async def clear_cancel(uuid: str):
    try:
        await _ensure_redis_connection()
        await r.delete(_cancel_key(uuid))
        logger.debug(f"Cancel cleared for {uuid}")
    except Exception as e:
        logger.error(f"Failed to clear cancel for {uuid}: {e}")

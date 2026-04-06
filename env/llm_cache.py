import hashlib
import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_TTL_DAYS = 450
DEFAULT_CACHE_DIR = ".cache/llm-responses"


class CachedMessage:
    def __init__(self, content: str | list[Any]):
        self.content = content


class CachedChoice:
    def __init__(self, message: CachedMessage):
        self.message = message


class CachedChatCompletion:
    def __init__(self, message_content: str | list[Any]):
        self.choices = [CachedChoice(CachedMessage(message_content))]


def _get_cache_dir() -> Path:
    target = os.getenv("LLM_CACHE_DIR", DEFAULT_CACHE_DIR)
    return Path(target).resolve()


def _get_ttl_seconds() -> int:
    try:
        days = float(os.getenv("LLM_CACHE_TTL_DAYS", DEFAULT_TTL_DAYS))
        return int(days * 24 * 3600)
    except ValueError:
        return int(DEFAULT_TTL_DAYS * 24 * 3600)


def _compute_cache_key(client: Any, kwargs: dict[str, Any]) -> str:
    payload = {
        "base_url": str(client.base_url),
        "kwargs": kwargs,
    }
    canonical_json = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _safe_write_cache(cache_dir: Path, key: str, data: dict[str, Any]) -> None:
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        final_path = cache_dir / f"{key}.json"
        
        fd, temp_path = tempfile.mkstemp(dir=cache_dir, prefix=f".{key}.tmp", text=True)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            
        os.replace(temp_path, final_path)
    except Exception as e:
        logger.warning(f"Failed to write to LLM cache: {e}")


def _read_cache(cache_dir: Path, key: str, ttl_seconds: int) -> dict[str, Any] | None:
    try:
        final_path = cache_dir / f"{key}.json"
        if not final_path.exists():
            return None
            
        stat = final_path.stat()
        age = time.time() - stat.st_mtime
        if age > ttl_seconds:
            return None
            
        with open(final_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to read from LLM cache: {e}")
        return None


def cached_chat_completion(client: Any, **kwargs: Any) -> Any:
    if kwargs.get("stream"):
        raise NotImplementedError("LLM response caching does not support stream=True")

    cache_dir = _get_cache_dir()
    cache_key = _compute_cache_key(client, kwargs)
    ttl_seconds = _get_ttl_seconds()

    cached_data = _read_cache(cache_dir, cache_key, ttl_seconds)
    if cached_data is not None:
        try:
            message_content = cached_data["payload"]["choices"][0]["message"]["content"]
            return CachedChatCompletion(message_content)
        except Exception as e:
            logger.warning(f"Failed to reconstruct cached completion, falling back to model: {e}")

    # Fall back to live call
    response = client.chat.completions.create(**kwargs)
    
    # Store response
    try:
        content = response.choices[0].message.content
        cache_payload = {
            "key": cache_key,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "ttl_days": ttl_seconds / (24 * 3600),
            "payload": {
                "choices": [{"message": {"content": content}}]
            }
        }
        _safe_write_cache(cache_dir, cache_key, cache_payload)
    except Exception as e:
        logger.warning(f"Failed to process live response for caching: {e}")

    return response

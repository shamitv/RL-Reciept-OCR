import json
import os
import pytest
import time
from unittest.mock import MagicMock, patch

from env.llm_cache import (
    cached_chat_completion,
    _compute_cache_key,
    _get_cache_dir,
    _get_ttl_seconds,
)


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockChoice:
    def __init__(self, message):
        self.message = message


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(MockMessage(content))]


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.base_url = "https://api.openai.com/v1"
    
    # Store mocked responses here for easy assertion
    client.chat.completions.create.return_value = MockResponse('{"test": "true"}')
    return client


def test_compute_cache_key(mock_client):
    kwargs1 = {"model": "gpt-4o", "temperature": 0.0, "messages": [{"role": "user", "content": "hi"}]}
    kwargs2 = {"temperature": 0.0, "messages": [{"role": "user", "content": "hi"}], "model": "gpt-4o"}
    
    key1 = _compute_cache_key(mock_client, kwargs1)
    key2 = _compute_cache_key(mock_client, kwargs2)
    
    assert key1 == key2

    kwargs3 = {"model": "gpt-4o", "temperature": 0.1, "messages": [{"role": "user", "content": "hi"}]}
    key3 = _compute_cache_key(mock_client, kwargs3)
    assert key1 != key3


def test_cache_hit_and_miss(mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    
    kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "hello"}]}
    
    # Miss
    response1 = cached_chat_completion(mock_client, **kwargs)
    assert response1.choices[0].message.content == '{"test": "true"}'
    assert mock_client.chat.completions.create.call_count == 1
    
    # Hit
    response2 = cached_chat_completion(mock_client, **kwargs)
    assert response2.choices[0].message.content == '{"test": "true"}'
    assert mock_client.chat.completions.create.call_count == 1  # Should not increment


def test_cache_ttl_expiration(mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_CACHE_TTL_DAYS", "0.00000001")  # very short TTL
    
    kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "ttl test"}]}
    
    # Miss
    cached_chat_completion(mock_client, **kwargs)
    assert mock_client.chat.completions.create.call_count == 1
    
    time.sleep(0.01) # Wait for expiry
    
    # Miss again due to expiry
    cached_chat_completion(mock_client, **kwargs)
    assert mock_client.chat.completions.create.call_count == 2


def test_stream_not_implemented(mock_client):
    kwargs = {"model": "gpt-4o", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
    with pytest.raises(NotImplementedError):
        cached_chat_completion(mock_client, **kwargs)


def test_graceful_io_error_fallback(mock_client, tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    kwargs = {"model": "gpt-4o", "messages": [{"role": "user", "content": "io error mock"}]}
    
    # Write corrupt data to cache
    key = _compute_cache_key(mock_client, kwargs)
    cache_file = tmp_path / f"{key}.json"
    cache_file.write_text("invalid json")
    
    # Should fall back to live call without raising exception
    response = cached_chat_completion(mock_client, **kwargs)
    assert response.choices[0].message.content == '{"test": "true"}'
    assert mock_client.chat.completions.create.call_count == 1

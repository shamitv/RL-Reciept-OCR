# Plan: Disk Cache For LLM Responses

## Status

- Current state: planned
- Current known LLM call sites:
  - `env/evaluation.py` extraction request
  - `env/evaluation.py` judge request

## Summary

- Add a shared on-disk cache layer for all LLM responses.
- Make cache lookup exact-match only: if the request payload matches 100%, serve the cached response instead of calling the model.
- Default cache TTL should be `450` days.
- Apply the cache to every OpenAI-compatible LLM request in the repo, not just the current eval flow.
- Keep the cache implementation local and deterministic so it works without extra infrastructure.

## Implementation Changes

- Introduce a shared cache module, likely `env/llm_cache.py`, that:
  - computes a canonical cache key using a cryptographic hash (e.g., `hashlib.sha256().hexdigest()`) of the full request payload to avoid filesystem limits.
  - stores cached responses on disk
  - enforces TTL checks on read
  - writes new responses after successful model calls using atomic disk writes (e.g., write to a temporary file and atomically rename it) to prevent corruption during parallel evaluations.
  - implements graceful degradation by catching I/O exceptions on disk operations, logging a warning, and falling back to calling the live model instead of crashing the core application.
- Route all current LLM requests through one cached helper instead of calling `client.chat.completions.create(...)` directly.
- Define exact-match cache semantics:
  - same base URL
  - same model
  - same endpoint type
  - same request body after canonical JSON serialization using `json.dumps(..., sort_keys=True)` to prevent ordering-based cache misses.
  - same message content, including image inputs and response-format parameters
  - same sampling parameters such as `temperature`
- For multimodal inputs, include the exact image payload reference in the cache key:
  - if using inline data URLs, hash the full serialized request body
  - do not perform fuzzy matching on filenames or derived metadata
- Keep cache entries on disk in a dedicated runtime directory, recommended:
  - `.cache/llm-responses/`
- Store each entry with metadata such as:
  - cache key
  - created time
  - expiry time
  - request fingerprint
  - raw response payload
- Expired entries should be treated as misses and replaced on the next successful request.
- Failed model calls should not be cached unless the team explicitly wants negative-result caching later.

## Interfaces

- Add environment-configurable cache settings:
  - `LLM_CACHE_DIR`
  - `LLM_CACHE_TTL_DAYS`
- Defaults:
  - `LLM_CACHE_DIR=.cache/llm-responses`
  - `LLM_CACHE_TTL_DAYS=450`
- Add a small wrapper API that callers use instead of raw OpenAI client calls, for example:
  - `cached_chat_completion(...)`
  - or a thin cached client wrapper around the OpenAI-compatible client
- When returning cached data, the wrapper must reconstruct the raw JSON payload into an object that provides identical attribute access (e.g., `completion.choices[0].message.content`) to a real OpenAI `ChatCompletion` object, ensuring existing call sites don't break.
- Return metadata to callers when helpful, such as whether the response came from cache, but keep the primary return shape compatible with the current code.

## Scope Rules

- In scope:
  - all current repo LLM requests
  - exact-match disk caching
  - TTL expiration
  - cache directory creation and safe file naming
  - tests for hit, miss, and expiry behavior
- Out of scope:
  - requests that specify `stream=True` (these should either bypass the cache entirely or raise a `NotImplementedError`)
  - semantic or approximate matching
  - distributed/shared cache services
  - cache invalidation by model quality heuristics
  - UI for browsing cache entries

## Test Plan

- Add unit tests for canonical key generation to prove equivalent request dicts produce the same key.
- Add tests for exact-match behavior:
  - first request is a miss and writes cache
  - second byte-for-byte equivalent request is a hit
  - changed model, message, image payload, or parameter produces a miss
- Add TTL tests:
  - unexpired entry is returned
  - expired entry is ignored and replaced
- Add integration-style tests around `env/evaluation.py` to verify both extraction and judge requests use the cache wrapper.
- Ensure tests do not require live model calls; use fake OpenAI client objects.

## Assumptions

- Exact-match means no normalization beyond canonical serialization of the request object.
- The cache is a runtime artifact and should not be committed to git.
- The current highest-value integration point is `env/evaluation.py`, but the implementation should make future LLM call sites use the same wrapper by default.

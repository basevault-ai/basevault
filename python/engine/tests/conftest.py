"""
Fixtures for ingestor tests.

Run pytest from the repo root:
    pytest python/engine/tests/
"""
import json
import os
import tempfile
import zipfile
import pytest


# ── Issue #264: helper for fakes that return a string ───────────────────────
# Pre-#264 fakes returned `str`; post-#264 the wrapper expects a
# `CompletionResult`. `_wrap` lets a fake stay terse while paying for
# the new return contract: `return _wrap(json.dumps(...), **kwargs)`.
# Threads `_call_id` from the wrapper-supplied kwargs back so callers
# reading `result.call_id` see the rec id the wrapper just begun.
from engine.llm import CompletionResult as _CR


def _wrap(content: str, **kwargs) -> _CR:
    cid = kwargs.get("_call_id")
    return _CR(
        content=content, call_id=cid,
        cache_key=f"k-{cid}" if cid else None, cached=False,
        finish_reason="stop", model="m", mode="test",
        prompt_tokens=10, completion_tokens=20,
        reasoning_tokens=0, reasoning_tokens_source=None,
        content_tokens=20, ttft_ms=5, ttfr_ms=None,
        last_token_ms=5, max_tokens_reserved=4096,
    )


@pytest.fixture
def wrap_completion():
    """Expose `_wrap` as a fixture for tests that prefer the
    fixture style. Tests can also import `_wrap` directly via
    `from .conftest import _wrap`."""
    return _wrap


# Disable the LLM prompt-hash cache by default in the test suite. Many
# tests call llm.complete() repeatedly with structurally identical
# inputs (stubbed providers, fixed prompts) and assert on what was
# sent to the provider client. With the cache active, the second call
# in a session would short-circuit to the cached response, never
# touching the stub — captured kwargs stay empty and assertions
# explode. Tests that specifically exercise the cache (test_llm_cache,
# test_golden_prompt_hashes) override this in their own fixtures.
# Also pin the cache root to a per-session tmp dir as belt-and-braces:
# any cache writes that slip through still land in tmp, never in the
# user's real ~/.basevault/cache/.
@pytest.fixture(autouse=True, scope="session")
def _isolated_llm_cache_root():
    with tempfile.TemporaryDirectory(prefix="bv-test-cache-") as tmp:
        os.environ["BASEVAULT_LLM_CACHE_DIR"] = tmp
        os.environ["BASEVAULT_LLM_CACHE_BYPASS"] = "1"
        yield tmp


# Reset cache hit/miss counters between tests so per-test assertions
# on llm_cache.get_cache_stats() don't see prior tests' totals.
@pytest.fixture(autouse=True)
def _reset_llm_cache_stats():
    try:
        from engine import llm_cache
        llm_cache.reset_cache_stats()
    except ImportError:
        pass
    yield


# Hermetic app config. The pipeline reads ~/.basevault/config.json directly
# via llm._read_app_config(); the suite must NOT depend on the developer's
# real, mutable config. A hand-edited stage_models entry (an unregistered
# model id) silently poisoned the whole suite through the per-stage routing
# resolver, and the failure count drifted with whatever model the dev last
# picked in Settings. Pin config reads to a fresh-install default ({}) for
# the session so suite results are machine-independent. Tests that exercise
# specific config monkeypatch _read_app_config themselves — their
# function-scoped patch wins and restores to this hermetic default on teardown.
@pytest.fixture(autouse=True, scope="session")
def _hermetic_app_config():
    from engine import llm

    def _empty_config():
        return {}

    orig_read = llm._read_app_config
    llm._read_app_config = _empty_config
    # chatbot_sidecar binds the name at module import, not per-call, so the
    # llm-side patch doesn't reach it — patch its bound name too.
    try:
        from engine import chatbot_sidecar
    except Exception:
        chatbot_sidecar = None
    else:
        chatbot_sidecar._read_app_config = _empty_config

    # `_STAGE_MODEL_MAP` is the per-stage routing map, computed ONCE at llm
    # import from the real config and read per-call thereafter. THIS is what
    # carried the dev's hand-edited `kimi+glm` stage entries into the
    # resolver, KeyError-cascading across the suite. Mutate it IN PLACE (not
    # rebind) to the shipped default: `vision.py` binds the dict object by
    # reference at its own import, so a rebind would desync them; rebuilt from
    # the empty config it resolves to registered models only.
    # (MODE_SPEC — the mode anchor — is left as-is: it reads `tee_model`,
    # which is a registered UI value, never the hand-edited `kimi+glm`.)
    orig_stage_map = dict(llm._STAGE_MODEL_MAP)
    hermetic_map = llm._resolve_stage_model_map()  # reads the patched {} config
    llm._STAGE_MODEL_MAP.clear()
    llm._STAGE_MODEL_MAP.update(hermetic_map)
    yield
    llm._read_app_config = orig_read
    llm._STAGE_MODEL_MAP.clear()
    llm._STAGE_MODEL_MAP.update(orig_stage_map)
    if chatbot_sidecar is not None:
        chatbot_sidecar._read_app_config = orig_read


# Reset cross-test global-state leaks. `runner._log_file` and
# `llm._current_stage` are module globals; a test that ran a pipeline (which
# opens then closes the run log) or set a stage tag could leave them dangling
# for the next test — a closed `_log_file` handle made `_log_write`
# raise "I/O operation on closed file", and a leaked `_current_stage` broke
# the chatbot stage_scope tests' "starts at None" precondition. Reset both
# around every test so order can't change the outcome.
@pytest.fixture(autouse=True)
def _reset_runner_globals():
    from engine import llm
    try:
        from engine import runner
    except Exception:
        runner = None
    llm._current_stage = None
    if runner is not None:
        runner._log_file = None
    yield
    llm._current_stage = None
    if runner is not None:
        runner._log_file = None


# ── Raw content samples ────────────────────────────────────────────────────────

WHATSAPP_CONTENT = """\
[1/15/24, 10:23:00 AM] Alice: Hey, are you free tomorrow?
[1/15/24, 10:25:00 AM] Bob: Yes, what time works for you?
[1/15/24, 10:26:00 AM] Alice: Let's say 3pm
<Media omitted>
[1/15/24, 10:27:00 AM] Bob: Sounds good!
"""

NOTION_MD_CONTENT = """\
# My Notion Page

This is exported from Notion.

## Section 1
Some content here.
"""

TXT_CONTENT = """\
Some plain text notes.
Nothing special about the format.
Could be anything.
"""

DAYONE_JSON_CONTENT = json.dumps({
    "entries": [
        {
            "uuid": "ABC123",
            "text": "First journal entry\\. Feeling good today\\.",
            "creationDate": "2024-03-01T08:00:00Z",
            "tags": ["personal", "morning"],
        },
        {
            "uuid": "DEF456",
            "text": "Second entry without tags\\.",
            "creationDate": "2024-03-02T09:30:00Z",
            "tags": [],
        },
    ]
})


# ── File fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def whatsapp_txt(tmp_path):
    f = tmp_path / "WhatsApp Chat - Alice.txt"
    f.write_text(WHATSAPP_CONTENT, encoding="utf-8")
    return f


@pytest.fixture
def whatsapp_zip(tmp_path):
    zip_path = tmp_path / "WhatsApp Chat - Alice.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("WhatsApp Chat - Alice.txt", WHATSAPP_CONTENT)
    return zip_path


@pytest.fixture
def txt_file(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text(TXT_CONTENT, encoding="utf-8")
    return f


@pytest.fixture
def notion_md_file(tmp_path):
    # Notion exports have a 32-char hex hash in the filename
    f = tmp_path / "My Page 1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d.md"
    f.write_text(NOTION_MD_CONTENT, encoding="utf-8")
    return f


@pytest.fixture
def notion_zip(tmp_path):
    zip_path = tmp_path / "Export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(
            "My Page 1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d.md",
            NOTION_MD_CONTENT,
        )
        zf.writestr(
            "Another Page aabbccddeeff00112233445566778899.md",
            "# Another Page\nSome content.",
        )
    return zip_path


@pytest.fixture
def dayone_json_file(tmp_path):
    f = tmp_path / "Journal.json"
    f.write_text(DAYONE_JSON_CONTENT, encoding="utf-8")
    return f


@pytest.fixture
def html_file(tmp_path):
    f = tmp_path / "export.html"
    f.write_text("<html><body>Some content</body></html>", encoding="utf-8")
    return f

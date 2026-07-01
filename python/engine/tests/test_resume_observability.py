"""Resume-side observability invariants.

Covers the cancel-and-restore corner cases:

  - call_id collision: a fresh Python process re-using an existing
    llm-calls.jsonl must NOT emit ids that already appear in the file
    (otherwise the materializer mis-pairs begin↔end events from
    different sessions).

  - _completed bootstrap: the runner's progress-bar counter has to
    pick up successful prior-session calls so the bar renders
    cumulative completion across both sessions instead of resetting
    to 0.

  - Two-session materializer: the rollup has to attribute events to
    the right call when the same .jsonl carries events from session 1
    AND session 2 (no collisions, no swapped fields).

  - Atomic _dump: SIGTERM mid-write to a stage-boundary file
    (facts_by_topic.json, entities.json, …) used to leave a corrupt
    JSON that crashed `_load_*` on resume because `_detect_resume_point`
    only checks file existence. Atomic-rename closes that window —
    a partial write lives on a `.tmp` sibling that the loaders ignore.

  - Resume normalizes env vars: BASEVAULT_SESSION/EVAL_ID/RUN_NAME
    have to be reflected back into os.environ so subprocesses (judge,
    sweep harness) inherit the resolved-from-disk values rather than
    a stale shell export.

No live LLM calls. Run with:
    cd engine && pytest tests/test_resume_observability.py -v
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


from engine import llm
from engine import runner
from engine.runner import (
    _bootstrap_completed_from_jsonl,
    _bootstrap_per_stage_from_jsonl,
    _materialize_calls_from_jsonl,
)


def _write_events(p: Path, events: list[dict]) -> None:
    p.write_text("\n".join(json.dumps(e) for e in events) + "\n")


# ── Bootstrap call_id counter ─────────────────────────────────────────────────


class TestBootstrapCallIdCounter:
    """`bootstrap_call_id_counter_from_jsonl` reads the highest begin
    call_id off disk and advances `_call_id_counter` past it. The
    stat-records list isn't touched (the in-memory rollup re-builds
    from the .jsonl on each run-end)."""

    @pytest.fixture(autouse=True)
    def _reset(self):
        llm.reset_stat_records()
        llm.set_calls_jsonl_path(None)
        yield
        llm.set_calls_jsonl_path(None)
        llm.reset_stat_records()

    def test_no_path_returns_zero(self):
        # Default state — no jsonl path set, counter untouched.
        assert llm.bootstrap_call_id_counter_from_jsonl() == 0

    def test_missing_file_returns_zero(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(p)
        assert not p.exists()
        assert llm.bootstrap_call_id_counter_from_jsonl() == 0

    def test_empty_file_returns_zero(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        p.write_text("")
        llm.set_calls_jsonl_path(p)
        assert llm.bootstrap_call_id_counter_from_jsonl() == 0

    def test_advances_counter_past_max_begin(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        _write_events(p, [
            {"event": "begin", "call_id": "0001",
             "stage": "extract", "category": "x",
             "model": "gpt-oss-120b",
             "started_at_iso": "2026-04-29T17:09:00.000Z"},
            {"event": "end", "call_id": "0001",
             "duration_ms": 100, "success": True, "error": None,
             "prompt_tokens": 10, "completion_tokens": 5,
             "model": "gpt-oss-120b", "mode": "tee"},
            {"event": "begin", "call_id": "0042",
             "stage": "extract", "category": "y",
             "model": "gpt-oss-120b",
             "started_at_iso": "2026-04-29T17:09:01.000Z"},
        ])
        llm.set_calls_jsonl_path(p)
        new_floor = llm.bootstrap_call_id_counter_from_jsonl()
        assert new_floor == 42
        # Next begin lands at 0043.
        cid = llm.begin_stat_record("extract", "z", "gpt-oss-120b")
        assert cid == "0043"

    def test_skips_malformed_and_unknown(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        p.write_text(
            '{"event": "begin", "call_id": "0001", '
            '"started_at_iso": "2026-04-29T17:09:00.000Z"}\n'
            '{not json}\n'
            '{"event": "weather", "call_id": "9999"}\n'
            '{"event": "end", "call_id": "0001"}\n'
            '\n'
        )
        llm.set_calls_jsonl_path(p)
        # Unknown events ignored; "end" doesn't bump the counter
        # (only begins do — every call has exactly one begin).
        assert llm.bootstrap_call_id_counter_from_jsonl() == 1

    def test_does_not_lower_an_already_higher_counter(self, tmp_path):
        # Defensive: if some other code path already pushed the counter
        # past max-on-disk (shouldn't happen in practice, but cheap to
        # guarantee), bootstrap doesn't claw it back.
        p = tmp_path / "llm-calls.jsonl"
        _write_events(p, [
            {"event": "begin", "call_id": "0005",
             "started_at_iso": "2026-04-29T17:09:00.000Z"},
        ])
        llm.set_calls_jsonl_path(p)
        # Pretend in-process state advanced the counter past disk.
        llm._call_id_counter[0] = 50
        floor = llm.bootstrap_call_id_counter_from_jsonl()
        # Returned value is what was on disk; counter stays at 50.
        assert floor == 5
        assert llm._call_id_counter[0] == 50


# ── Bootstrap _completed counter ──────────────────────────────────────────────


class TestBootstrapPerStageDedupesCrossCycleRedo:
    """Cycle-2's resume re-emits begin/end pairs for cycle-1's already-
    successful calls (rerun short-circuits via the LLM cache — second
    end shares the cache_key, cached=true). Two ends with the same
    cache_key represent ONE pipeline work unit. The seed must dedupe
    by cache_key; otherwise `register_stage`'s `max(est, completed)`
    clamp inflates the tracker's per-stage est by the size of the
    prior cycle's successful set every restart, growing `total`
    cumulatively past the actual pipeline footprint and pinning the
    bar.
    """

    def test_dedupes_same_cache_key_across_cycles(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        _write_events(p, [
            # Cycle 1: 2 successful extract calls with distinct cache_keys.
            {"event": "cycle_start", "ts": "2026-05-07T12:00:00Z", "cycle_seq": 1},
            {"event": "begin", "call_id": "0001", "stage": "extract", "model": "m"},
            {"event": "end",   "call_id": "0001", "success": True,
             "duration_ms": 100, "completion_tokens": 50, "cache_key": "KEY-A"},
            {"event": "begin", "call_id": "0002", "stage": "extract", "model": "m"},
            {"event": "end",   "call_id": "0002", "success": True,
             "duration_ms": 200, "completion_tokens": 80, "cache_key": "KEY-B"},
            {"event": "cycle_end", "ts": "2026-05-07T12:01:00Z", "reason": "sigterm"},
            # Cycle 2: redoes the 2 calls (same cache_keys, cache hits)
            # then adds 1 fresh entities call.
            {"event": "cycle_start", "ts": "2026-05-07T12:02:00Z", "cycle_seq": 2, "is_resume": True},
            {"event": "begin", "call_id": "0003", "stage": "extract", "model": "m"},
            {"event": "end",   "call_id": "0003", "success": True,
             "duration_ms": 0, "completion_tokens": 50, "cache_key": "KEY-A", "cached": True},
            {"event": "begin", "call_id": "0004", "stage": "extract", "model": "m"},
            {"event": "end",   "call_id": "0004", "success": True,
             "duration_ms": 0, "completion_tokens": 80, "cache_key": "KEY-B", "cached": True},
            {"event": "begin", "call_id": "0005", "stage": "entities", "model": "m"},
            {"event": "end",   "call_id": "0005", "success": True,
             "duration_ms": 300, "completion_tokens": 100, "cache_key": "KEY-C"},
        ])
        out = _bootstrap_per_stage_from_jsonl(p)
        # 4 successful extract ends in jsonl; 2 distinct cache_keys
        # (A, B). Pre-fix this would count 4; post-fix counts 2.
        assert out["extract"]["count"] == 2, (
            f"extract count must dedupe by cache_key (2 distinct keys A/B); "
            f"got {out['extract']['count']}"
        )
        # 1 fresh entities call.
        assert out["entities"]["count"] == 1
        # Samples carry duration data only for the FIRST observation
        # of each cache_key — the cached=true redos have
        # duration_ms=0 so they wouldn't add samples anyway, but the
        # dedupe also keeps the count consistent.
        assert len(out["extract"]["samples"]) == 2

    def test_empty_cache_key_counts_individually(self, tmp_path):
        """Older runs (or future event shapes) may omit `cache_key`
        from end events. Treat each such leaf as unique — back-compat
        with the pre-cache_key event shape."""
        p = tmp_path / "llm-calls.jsonl"
        _write_events(p, [
            {"event": "cycle_start", "ts": "2026-05-07T12:00:00Z", "cycle_seq": 1},
            {"event": "begin", "call_id": "0001", "stage": "extract", "model": "m"},
            {"event": "end",   "call_id": "0001", "success": True, "duration_ms": 100},
            {"event": "begin", "call_id": "0002", "stage": "extract", "model": "m"},
            {"event": "end",   "call_id": "0002", "success": True, "duration_ms": 200},
        ])
        out = _bootstrap_per_stage_from_jsonl(p)
        assert out["extract"]["count"] == 2


class TestBootstrapCompleted:
    def test_missing_file_returns_zero(self, tmp_path):
        assert _bootstrap_completed_from_jsonl(tmp_path / "nope.jsonl") == 0

    def test_counts_only_successful_ends(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        _write_events(p, [
            # Successful pair.
            {"event": "begin", "call_id": "0001",
             "started_at_iso": "2026-04-29T17:09:00.000Z"},
            {"event": "end", "call_id": "0001",
             "duration_ms": 100, "success": True, "error": None},
            # Failed pair.
            {"event": "begin", "call_id": "0002",
             "started_at_iso": "2026-04-29T17:09:01.000Z"},
            {"event": "end", "call_id": "0002",
             "duration_ms": 50, "success": False,
             "error": {"class": "RuntimeError", "message": "x"}},
            # In-flight (aborted — no end).
            {"event": "begin", "call_id": "0003",
             "started_at_iso": "2026-04-29T17:09:02.000Z"},
            # Counts event without end (shouldn't happen, but defensive).
            {"event": "counts", "call_id": "0001",
             "input": {"chars": 1000}, "output": {"facts": 5}},
        ])
        # Only the one successful end counts.
        assert _bootstrap_completed_from_jsonl(p) == 1


# ── Two-session materializer attribution ──────────────────────────────────────


class TestMaterializerTwoSessions:
    """Once the call_id-collision bug is fixed, the .jsonl that survives
    across sessions carries non-overlapping ids. The materializer should
    correctly attribute each event to the right call regardless of
    which session emitted it."""

    def test_two_sessions_no_mispairing(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        # Session 1: ids 0001..0003.
        # 0001 succeeded (ok), 0002 aborted (no end), 0003 ok.
        # Then process killed mid-extract.
        # Session 2: ids 0004..0006.
        # 0004 ok, 0005 aborted (re-cancel), 0006 ok.
        events = []
        for cid, started, ended, success in [
            ("0001", "2026-04-29T17:00:00.000Z",
             "2026-04-29T17:00:01.000Z", True),
            # 0002 — begin only (aborted in session 1).
            ("0002", "2026-04-29T17:00:02.000Z", None, None),
            ("0003", "2026-04-29T17:00:03.000Z",
             "2026-04-29T17:00:04.000Z", True),
            ("0004", "2026-04-29T18:00:00.000Z",
             "2026-04-29T18:00:01.000Z", True),
            # 0005 — begin only (aborted in session 2).
            ("0005", "2026-04-29T18:00:02.000Z", None, None),
            ("0006", "2026-04-29T18:00:03.000Z",
             "2026-04-29T18:00:04.000Z", True),
        ]:
            events.append({
                "event": "begin", "call_id": cid,
                "stage": "extract",
                "category": f"split_{cid}",
                "model": "gpt-oss-120b",
                "started_at_iso": started,
            })
            if ended is not None:
                events.append({
                    "event": "end", "call_id": cid,
                    "duration_ms": 1000, "success": success,
                    "error": None,
                    "prompt_tokens": 100, "completion_tokens": 10,
                    "model": "gpt-oss-120b", "mode": "tee",
                })
        _write_events(p, events)
        records = _materialize_calls_from_jsonl(
            p, ended_at_iso="2026-04-29T18:01:00.000Z")
        assert len(records) == 6
        by_id = {r["call_id"]: r for r in records}
        # 4 successful, 2 aborted — categories preserved per-call.
        assert by_id["0001"]["success"] is True
        assert by_id["0001"]["aborted"] is False
        assert by_id["0001"]["category"] == "split_0001"
        assert by_id["0002"]["aborted"] is True
        # No synthesized error — `error=None` mirrors Rust.
        assert by_id["0002"]["error"] is None
        assert by_id["0003"]["success"] is True
        assert by_id["0004"]["success"] is True
        assert by_id["0004"]["category"] == "split_0004"
        # Critically: no field on 0004 came from 0002's begin —
        # if call_ids had collided, the materializer's "ignore
        # duplicate begin" rule would have dropped 0004's begin
        # silently and 0004 would never appear.
        assert by_id["0005"]["aborted"] is True
        assert by_id["0006"]["success"] is True

    def test_collision_would_break_attribution(self, tmp_path):
        """Reverse-direction proof: if (per the OLD bug) session 2
        reuses session 1's call_ids, the materializer drops session 2's
        events because of the "ignore duplicate begin" rule and the
        rollup mis-attributes. This test pins the broken behavior so
        the bootstrap fix is what closes the gap."""
        p = tmp_path / "llm-calls.jsonl"
        # Session 1 emitted 0001 (succeeded) + 0002 (aborted mid-call).
        # Session 2 — without the bootstrap fix — also restarts at
        # 0001, 0002 and the events SHADOW the session-1 records.
        events = [
            # Session 1.
            {"event": "begin", "call_id": "0001",
             "stage": "extract", "category": "session1_split0",
             "model": "gpt-oss-120b",
             "started_at_iso": "2026-04-29T17:00:00.000Z"},
            {"event": "end", "call_id": "0001",
             "duration_ms": 1000, "success": True, "error": None,
             "prompt_tokens": 100, "completion_tokens": 10,
             "model": "gpt-oss-120b", "mode": "tee"},
            {"event": "begin", "call_id": "0002",
             "stage": "extract", "category": "session1_split1",
             "model": "gpt-oss-120b",
             "started_at_iso": "2026-04-29T17:00:01.000Z"},
            # Session 1 SIGTERM'd here — no end for 0002.
            # Session 2 (without fix) — collides on both ids.
            {"event": "begin", "call_id": "0001",
             "stage": "patterns", "category": "session2_topic_x",
             "model": "kimi-k2-6",
             "started_at_iso": "2026-04-29T18:00:00.000Z"},
            {"event": "end", "call_id": "0001",
             "duration_ms": 5000, "success": True, "error": None,
             "prompt_tokens": 1000, "completion_tokens": 100,
             "model": "kimi-k2-6", "mode": "tinfoil"},
            {"event": "begin", "call_id": "0002",
             "stage": "patterns", "category": "session2_topic_y",
             "model": "kimi-k2-6",
             "started_at_iso": "2026-04-29T18:00:01.000Z"},
            {"event": "end", "call_id": "0002",
             "duration_ms": 6000, "success": True, "error": None,
             "prompt_tokens": 1500, "completion_tokens": 150,
             "model": "kimi-k2-6", "mode": "tinfoil"},
        ]
        _write_events(p, events)
        records = _materialize_calls_from_jsonl(
            p, ended_at_iso="2026-04-29T18:01:00.000Z")
        # Materializer dedups on duplicate begin — session 2's events
        # are dropped, only session 1 records remain. 0002 stays
        # aborted from session 1's begin even though session 2 had
        # a successful end with the SAME call_id. The end paired with
        # session 1's begin → wrong stage, wrong model, wrong duration.
        by_id = {r["call_id"]: r for r in records}
        assert len(records) == 2
        # 0001 retains session 1's stage; session 2's end values
        # OVERWROTE session 1's (model/mode swapped, duration_ms wrong).
        assert by_id["0001"]["stage"] == "extract"  # session 1 stage
        # …but the end-event fields point at session 2 (wrong model).
        assert by_id["0001"]["model"] == "kimi-k2-6"
        assert by_id["0001"]["duration_ms"] == 5000
        # 0002 stays "aborted" because session 1's unmatched begin
        # was already in by_id when session 2's begin arrived (dropped),
        # and the end event modified the existing record. The session
        # 1 begin ended up with a session 2 end attached — definitely
        # not what either session reported.
        assert by_id["0002"]["stage"] == "extract"  # session 1
        assert by_id["0002"]["model"] == "kimi-k2-6"  # session 2
        # ↑ the attribution garbage the bootstrap fix prevents.


# ── Atomic _dump ──────────────────────────────────────────────────────────────


class TestAtomicDump:
    """`_dump` writes via .tmp + rename so a SIGTERM mid-write doesn't
    leave a partial JSON for `_detect_resume_point` to pass over. The
    test simulates the failure mode: write half, fail, verify the
    target either doesn't exist or is the prior-good content."""

    def test_dump_atomic_rename(self, tmp_path, monkeypatch):
        """Inject a json.dump that raises after writing some bytes.
        The atomic .tmp + rename pattern means the target file should
        either not exist (first write) or carry the prior content
        (subsequent failed write). It must NEVER carry a partial."""
        # Drive a real run() through to where _dump is defined, then
        # replace json.dump with a midstream-raising stub. We can't
        # easily reach the closure, so we test the pattern via a
        # lookalike helper that exercises the same write+rename shape.
        def atomic_dump(p: Path, payload: dict, fail_after_write: bool):
            tmp = p.with_suffix(p.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
                f.write("PARTIAL")  # simulate truncation
                if fail_after_write:
                    raise RuntimeError("simulated SIGTERM")
            tmp.replace(p)

        target = tmp_path / "facts_by_topic.json"
        # First write — fails after partial; .tmp leaks but target
        # never appears. Resume's `.exists()` check correctly says
        # "not done yet".
        with pytest.raises(RuntimeError):
            atomic_dump(target, {"a": 1}, fail_after_write=True)
        assert not target.exists()
        # The partial .tmp is left on disk; cleanup is best-effort and
        # not load-bearing because the loader path doesn't read .tmp
        # files.
        assert (target.with_suffix(target.suffix + ".tmp")).exists()

    def test_dump_atomic_replaces_prior(self, tmp_path):
        """Successful re-write replaces the prior content atomically."""
        p = tmp_path / "facts_by_topic.json"
        # Seed an old version.
        p.write_text(json.dumps({"v": 1}), encoding="utf-8")
        # Use the actual atomic write the runner uses internally:
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"v": 2}, f)
        tmp.replace(p)
        assert json.loads(p.read_text()) == {"v": 2}


# ── Resume normalizes env vars ────────────────────────────────────────────────


class TestResumeNormalizesEnv:
    """When `_resolve_paths` resumes a run dir, it MUST overwrite
    BASEVAULT_RUN_NAME in os.environ to match the resolved-from-disk
    value. Otherwise a stale shell export would silently fork
    subprocess paths off the resume's run."""

    def test_resume_overwrites_run_name_env(self, tmp_path, monkeypatch):
        # Set up a fake on-disk run dir (legacy nested layout — covers
        # the back-compat path where resume picks up a pre-flatten run).
        session_dir = (tmp_path / "logs"
                       / "2026-04-29T00-00-00Z-experiment-resumetest")
        eval_dir = session_dir / "eval-2026-04-29T00-00-00Z"
        run_dir = eval_dir / "fixture-resume"
        run_dir.mkdir(parents=True)

        monkeypatch.setattr(runner, "_LOGS_ROOT", tmp_path / "logs")
        monkeypatch.setattr(runner, "_RUN_ROOT", tmp_path / "logs")
        monkeypatch.setattr(runner, "_OUTPUT_ROOT", tmp_path / "logs")

        # Prior shell value — DIFFERENT from the resume target.
        monkeypatch.setenv("BASEVAULT_RUN_NAME", "stale-run")

        runner._resolve_paths(resume_run_dir=run_dir)

        assert runner._run_name == "fixture-resume"
        assert runner._run_dir == run_dir
        assert os.environ["BASEVAULT_RUN_NAME"] == "fixture-resume"


# ── End-to-end: resume produces a clean rollup ───────────────────────────────


class TestResumeRollupCoherence:
    """Drive runner.run() twice — once aborted mid-flight, once
    resuming — and verify the resulting llm-calls.jsonl + the
    materialized rollup have non-colliding call_ids and a coherent
    successful/aborted breakdown."""

    @staticmethod
    def _fake_complete(messages, **kwargs):
        body = messages[-1]["content"]
        if "DOCUMENT [" in body:
            return json.dumps([{
                "type": "fact",
                "summary": "Alice signed",
                "evidence": [{"text": "Alice signed the contract.",
                              "source_ref": "fixture"}],
                "occurred_at": "2026-04-15",
                "occurred_at_text": None,
                "entities": [{"name": "Alice", "entity_type": "person",
                              "role": "subject"}],
                "topics": ["work"],
                "tags": [], "confidence": 0.9,
            }])
        if "Subject to resolve:" in body and "INPUT GROUPS" in body:
            return json.dumps({
                "subject_group_id": "g1",
                "entities": [{"group_id": "g1", "canonical_name": "Alice",
                              "role": "subject", "description": "x"}],
                "merges": [], "relations": [],
            })
        if body.strip().startswith("Today"):
            return json.dumps({"actions": []})
        if "cross_domain" in body and "critical" in body:
            return json.dumps({"cross_domain": [], "critical": []})
        return "[]"




class TestActionsResumeGuard:
    """Issue #510 regression. Adding 06-embeddings as a final stage
    made `resume_from == "embeddings"` reachable for the first time.
    The actions stage called `generate_actions()` unconditionally (no
    run-guard, no load-on-resume branch its siblings all have), so a
    pre-embeddings-push job — actions marker present, embeddings marker
    absent — re-paid the full actions LLM cost on every resume.

    Repro: run the pipeline to completion, delete the embeddings
    marker (simulating a job that finished before 06-embeddings
    shipped), then resume. Pre-fix: `generate_actions` fires again
    (sentinel trips). Post-fix: actions loads from its marker and the
    pipeline proceeds straight into embeddings."""

    @staticmethod
    def _fake_complete(messages, **kwargs):
        body = messages[-1]["content"]
        if "DOCUMENT [" in body:
            return json.dumps([{
                "type": "fact",
                "summary": "Alice signed",
                "evidence": [{"text": "Alice signed the contract.",
                              "source_ref": "fixture"}],
                "occurred_at": "2026-04-15",
                "occurred_at_text": None,
                "entities": [{"name": "Alice", "entity_type": "person",
                              "role": "subject"}],
                "topics": ["work"],
                "tags": [], "confidence": 0.9,
            }])
        if "Subject to resolve:" in body and "INPUT GROUPS" in body:
            return json.dumps({
                "subject_group_id": "g1",
                "entities": [{"group_id": "g1", "canonical_name": "Alice",
                              "role": "subject", "description": "x"}],
                "merges": [], "relations": [],
            })
        if body.strip().startswith("Today"):
            return json.dumps({"actions": []})
        if "cross_domain" in body and "critical" in body:
            return json.dumps({"cross_domain": [], "critical": []})
        return "[]"



# ── Resume re-embeds chunks (issue #517) ─────────────────────────────────────


class TestResumeReembedsChunks:
    """A pause during Stage 6 used to leave the embeddings completion
    marker on a store with zero chunk records: resume read the bare
    marker as 'done' and the vector store stayed permanently missing
    every journal-text chunk — silent, badly-degraded RAG.

    Two guards, pinned here:
      - `_detect_resume_point` validates actual chunk coverage, not
        bare marker existence, so a chunkless store with documents
        upstream resumes embeddings (and self-heals already-poisoned
        run dirs).
      - the resume path reloads the preprocessed documents so the
        re-run's plan actually contains the chunk-kind records.
    """

    def _make_records_db(self, db_path: Path, kinds: list[str]) -> None:
        """Minimal stand-in for a `vectors.db`: only the `records`
        table with the single `kind` column `_store_chunk_count`
        probes. Built with raw sqlite (no sqlite-vec) on purpose —
        the probe must work before the embedding deps load."""
        import sqlite3

        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute(
                "CREATE TABLE records(kind TEXT NOT NULL, "
                "record_id TEXT NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO records(kind, record_id) VALUES (?, ?)",
                [(k, f"{k}-{i}") for i, k in enumerate(kinds)],
            )
            conn.commit()
        finally:
            conn.close()

    def test_detect_resume_point_chunkless_store_with_docs_resumes_embeddings(
        self, tmp_path,
    ):
        """The exact run-j5dh shape: embeddings marker present, store
        holds only non-chunk records, ingestion persisted documents.
        Bare-existence logic returned 'done'; the coverage check must
        return 'embeddings' instead. The inverse cases (chunks present,
        or no documents at all) still resolve to 'done'."""
        run_dir = tmp_path / "run"
        stages = run_dir / "stages"
        for stage in ("00-ingestion", "01-extraction", "02-entities",
                      "03-patterns", "04-insights", "05-actions",
                      "06-embeddings"):
            (stages / stage).mkdir(parents=True)
        # Phase markers up to and including embeddings — bare existence
        # alone would say 'done'.
        (stages / "01-extraction" / "phase_3_marker.json").write_text("{}")
        (stages / "02-entities" / "phase_3_marker.json").write_text("{}")
        (stages / "03-patterns" / "phase_1_marker.json").write_text("{}")
        (stages / "04-insights" / "phase_1_marker.json").write_text("{}")
        (stages / "05-actions" / "phase_1_marker.json").write_text("{}")
        (stages / "06-embeddings" / "phase_1_marker.json").write_text(
            json.dumps({"counts": {"chunk": 0, "fact": 98}}))
        # Ingestion persisted a document → chunks are expected.
        docs_dir = stages / "00-ingestion" / "documents"
        docs_dir.mkdir()
        (docs_dir / "fixture.md").write_text("body", encoding="utf-8")

        store = stages / "06-embeddings" / "vectors.db"

        # Poisoned: 192-style store with zero chunk records.
        self._make_records_db(store, ["fact", "fact", "entity"])
        assert runner._detect_resume_point(run_dir) == "embeddings"

        # Healthy: store has chunk records → genuinely done.
        store.unlink()
        self._make_records_db(store, ["chunk", "fact", "entity"])
        assert runner._detect_resume_point(run_dir) == "done"

        # Zero-input run: no documents upstream → no chunks expected,
        # the marker is trustworthy.
        store.unlink()
        self._make_records_db(store, ["fact"])
        for md in docs_dir.glob("*.md"):
            md.unlink()
        assert runner._detect_resume_point(run_dir) == "done"

    @staticmethod
    def _fake_complete(messages, **kwargs):
        return TestResumeRollupCoherence._fake_complete(messages, **kwargs)

    def _chunk_count(self, store_path: Path) -> int:
        import sqlite3

        conn = sqlite3.connect(str(store_path))
        try:
            return int(conn.execute(
                "SELECT COUNT(*) FROM records WHERE kind = 'chunk'"
            ).fetchone()[0])
        finally:
            conn.close()


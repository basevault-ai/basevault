"""Run config audit invariants.

Covers the run-start config snapshot, the per-call request_extras
event-log shape, and the rollup's per-stage reasoning aggregation.

No live LLM calls — the snapshot helpers exercise pure data-flow
(config.json → stage map → snapshot dict; jsonl events → materialized
records → per-stage stats).
"""
from __future__ import annotations

import json

import pytest


from engine import llm
from engine import runner


@pytest.fixture(autouse=True)
def _reset_global_run_state():
    """Tests below mutate `runner._run_state` directly; reset so a
    test that leaves state populated doesn't pollute the next."""
    runner._run_state = None
    yield
    runner._run_state = None


# ── _build_run_config_snapshot ────────────────────────────────────────────────


class TestBuildRunConfigSnapshot:
    """The snapshot captures every routing knob that materially shapes
    the run, plus pipeline version + input metadata."""

    def test_minimal_snapshot_with_empty_inputs(self, tmp_path, monkeypatch):
        # Empty inputs list — exercises the path where _file_metadata
        # has nothing to hash. `mode` is recorded as the enum value;
        # TEE is the only cloud mode (the old TEST mode was removed).
        spec = llm.get_mode_spec(llm.Mode.TEE)
        snap = runner._build_run_config_snapshot(
            mode=llm.Mode.TEE,
            spec=spec,
            paths=[],
            sentiment="neutral",
        )
        assert snap["mode"] == "tee"
        assert snap["provider"] == spec.provider.value
        assert snap["primary_model"] == spec.model_id
        assert snap["temperature"] == 0.0
        assert snap["sentiment"] == "neutral"
        assert snap["inputs"] == []
        # Always present even when no override env is set; None is
        # acceptable, missing key is not.
        assert "app_version" in snap
        # Per-stage maps cover every stage in progress.PIPELINE_STAGES.
        # The metadata stage was removed from the pipeline post-#111;
        # its absence here is intentional, not a missing key.
        assert set(snap["stage_models"]) >= {
            "extract", "entities",
            "entities_dedupe", "patterns", "insights", "actions",
        }
        assert set(snap["stage_reasoning"]) >= set(snap["stage_models"])

    def test_input_metadata_includes_size_and_sha256(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f1.write_text("hello world\n")
        f2 = tmp_path / "b.txt"
        f2.write_text("xyz")
        spec = llm.get_mode_spec(llm.Mode.TEE)
        snap = runner._build_run_config_snapshot(
            mode=llm.Mode.TEE, spec=spec,
            paths=[str(f1), str(f2)],
            sentiment="neutral",
        )
        assert len(snap["inputs"]) == 2
        a, b = snap["inputs"]
        assert a["path"] == str(f1)
        assert a["size_bytes"] == len(b"hello world\n")
        assert len(a["sha256"]) == 64  # hex sha256
        # Different content → different hashes.
        assert a["sha256"] != b["sha256"]

    def test_missing_input_path_records_error_not_raises(self, tmp_path):
        """A missing file shouldn't crash the snapshot — record the
        error so post-hoc analysis can see what wasn't readable."""
        spec = llm.get_mode_spec(llm.Mode.TEE)
        snap = runner._build_run_config_snapshot(
            mode=llm.Mode.TEE, spec=spec,
            paths=[str(tmp_path / "does-not-exist.txt")],
            sentiment="neutral",
        )
        assert len(snap["inputs"]) == 1
        entry = snap["inputs"][0]
        assert "error" in entry
        assert "size_bytes" not in entry

    def test_app_version_picked_up_from_env(self, monkeypatch, tmp_path):
        monkeypatch.setenv("BASEVAULT_APP_VERSION", "0.9.99")
        spec = llm.get_mode_spec(llm.Mode.TEE)
        snap = runner._build_run_config_snapshot(
            mode=llm.Mode.TEE, spec=spec,
            paths=[], sentiment="neutral",
        )
        assert snap["app_version"] == "0.9.99"

    def test_categories_default_when_config_absent(self, monkeypatch):
        from engine.content_extractor import _DEFAULT_TOPICS
        monkeypatch.setattr(llm, "_read_app_config", lambda: {})
        spec = llm.get_mode_spec(llm.Mode.TEE)
        snap = runner._build_run_config_snapshot(
            mode=llm.Mode.TEE, spec=spec, paths=[], sentiment="neutral",
        )
        assert snap["categories"] == list(_DEFAULT_TOPICS)

    def test_categories_pinned_from_config(self, monkeypatch):
        monkeypatch.setattr(
            llm, "_read_app_config",
            lambda: {"categories": ["fundraising", "research", "other"]},
        )
        spec = llm.get_mode_spec(llm.Mode.TEE)
        snap = runner._build_run_config_snapshot(
            mode=llm.Mode.TEE, spec=spec, paths=[], sentiment="neutral",
        )
        assert snap["categories"] == ["fundraising", "research", "other"]


# ── _resolve_stage_reasoning_map_for_run ──────────────────────────────────────


class TestResolveStageReasoningMap:
    """Reasoning is gated by `_REASONING_WHITELIST` ∧ user toggle.
    The snapshot mirrors that gate so the recorded flag matches what
    each call actually receives."""

    def test_all_off_for_local_mode(self):
        # LOCAL mode: not whitelisted, so every stage resolves False
        # regardless of any user toggle.
        m = runner._resolve_stage_reasoning_map_for_run(llm.Mode.LOCAL)
        assert all(v is False for v in m.values())

    def test_user_toggle_true_but_not_whitelisted_resolves_false(
        self, monkeypatch
    ):
        # Override the in-memory stage map so `extract` claims
        # reasoning=True against a non-whitelisted (provider, model,
        # stage) tuple — the gate must still resolve False.
        fake_map = {
            "extract":          {"model": "made-up-model", "reasoning": True},
            "entities":         {"model": "made-up-model", "reasoning": True},
            "entities_dedupe":  {"model": "made-up-model", "reasoning": True},
            "patterns":         {"model": "made-up-model", "reasoning": True},
            "insights":         {"model": "made-up-model", "reasoning": True},
            "actions":          {"model": "made-up-model", "reasoning": True},
        }
        monkeypatch.setattr(llm, "_STAGE_MODEL_MAP", fake_map)
        # Use TEE mode so per-stage routing actually applies.
        m = runner._resolve_stage_reasoning_map_for_run(llm.Mode.TEE)
        # Every stage resolves to False because the (provider,
        # made-up-model, stage) tuple isn't whitelisted.
        assert all(v is False for v in m.values()), m


class TestResolveStageModelMapForRun:
    """The run snapshot (config.json / run.log / Run details) shows the
    configured model verbatim — a multi-model sentinel stays whole —
    while tracker/ETA callers get the collapsed real-backend id their
    (stage, model) history keys on."""

    def test_raw_keeps_sentinel_whole_default_collapses(self, monkeypatch):
        fake_map = {
            "vision":           {"model": "kimi-k2-6",    "reasoning": False},
            "extract":          {"model": "kimi+glm",     "reasoning": False},
            "entities":         {"model": "gpt-oss-120b", "reasoning": False},
            "entities_dedupe":  {"model": "gpt-oss-120b", "reasoning": False},
            "patterns":         {"model": "kimi+glm",     "reasoning": False},
            "insights":         {"model": "kimi-k2-6",    "reasoning": False},
            "actions":          {"model": "kimi-k2-6",    "reasoning": False},
        }
        monkeypatch.setattr(llm, "_STAGE_MODEL_MAP", fake_map)
        spec = llm.get_mode_spec(llm.Mode.TEE)

        raw = runner._resolve_stage_model_map_for_run(
            llm.Mode.TEE, spec, raw=True)
        collapsed = runner._resolve_stage_model_map_for_run(llm.Mode.TEE, spec)

        # Display/record: sentinel preserved so the user sees the pair.
        assert raw["extract"] == "kimi+glm"
        assert raw["patterns"] == "kimi+glm"
        # Tracker/ETA: collapsed to the first constituent.
        assert collapsed["extract"] == "kimi-k2-6"
        assert collapsed["patterns"] == "kimi-k2-6"
        # Non-sentinel stages identical either way.
        assert raw["entities"] == collapsed["entities"] == "gpt-oss-120b"

    def test_local_snapshot_pins_every_stage_to_local_primary(
        self, monkeypatch
    ):
        # In local mode the run snapshot (config.json → Run details)
        # must show the local model that actually ran for EVERY stage —
        # not the configured cloud per-stage routing left over from TEE,
        # which persists in config even when the run is local. Local
        # dispatch is pinned
        # to the local primary by `_resolve_stage_override` (its stage
        # map is TEE-only); the raw snapshot must mirror that gate
        # instead of echoing the stale cloud ids.
        cloud_map = {
            "vision":           {"model": "gemma4:26b",   "reasoning": False},
            "extract":          {"model": "gpt-oss-120b", "reasoning": False},
            "entities":         {"model": "gpt-oss-120b", "reasoning": False},
            "entities_dedupe":  {"model": "gemma4-31b",   "reasoning": False},
            "patterns":         {"model": "kimi-k2-6",    "reasoning": False},
            "insights":         {"model": "kimi-k2-6",    "reasoning": False},
            "actions":          {"model": "kimi-k2-6",    "reasoning": False},
        }
        monkeypatch.setattr(llm, "_STAGE_MODEL_MAP", cloud_map)
        local_spec = llm.get_mode_spec(llm.Mode.LOCAL)

        raw = runner._resolve_stage_model_map_for_run(
            llm.Mode.LOCAL, local_spec, raw=True)

        # Every stage — vision included — reads the local primary, never
        # a configured cloud id. Pre-fix this returned the cloud_map
        # values verbatim for chat stages and "gemma4:26b" for vision.
        assert raw, "snapshot map should not be empty"
        for stage, model in raw.items():
            assert model == local_spec.model_id, (
                f"stage {stage!r} recorded {model!r}, "
                f"expected local primary {local_spec.model_id!r}"
            )
        # No cloud id from the configured map leaks into the local
        # snapshot.
        cloud_ids = {e["model"] for e in cloud_map.values()}
        assert not (set(raw.values()) & cloud_ids), raw

    def test_local_tracker_path_keeps_vision_dispatch_model(self):
        # The raw=False (tracker / ETA) path keys on the id the wrapper
        # records vision calls under. `_VISION_MODEL` DELIBERATELY has no
        # LOCAL entry (no single local vision id is correct across the mlx
        # / ollama backends — see vision.py), so a local vision call is
        # dispatched and recorded under the local primary
        # (`get_mode_spec(LOCAL).model_id`). The tracker map must mirror
        # that — vision resolves to the local primary, same as the chat
        # stages — so the (stage, model) historical-duration lookup keys
        # on the id actually written.
        from engine.vision import _VISION_MODEL
        assert llm.Mode.LOCAL not in _VISION_MODEL  # design invariant
        local_spec = llm.get_mode_spec(llm.Mode.LOCAL)
        collapsed = runner._resolve_stage_model_map_for_run(
            llm.Mode.LOCAL, local_spec)
        assert collapsed["vision"] == local_spec.model_id
        # Chat stages also resolve to the local primary (dispatch model).
        assert collapsed["extract"] == local_spec.model_id


# ── begin_stat_record carries request_extras ──────────────────────────────────


class TestBeginEventRequestExtras:
    @pytest.fixture(autouse=True)
    def _reset(self, tmp_path):
        llm.reset_stat_records()
        # Route begin events to a temp jsonl so we can read them back.
        self.jsonl = tmp_path / "llm-calls.jsonl"
        llm.set_calls_jsonl_path(self.jsonl)
        yield
        llm.set_calls_jsonl_path(None)
        llm.reset_stat_records()

    def test_request_extras_appears_in_begin_event_when_provided(self):
        llm.begin_stat_record(
            "extract", "doc-1", "kimi-k2-6",
            request_extras={"reasoning": True, "temperature": 0.0},
        )
        events = [json.loads(l) for l in
                  self.jsonl.read_text().splitlines() if l.strip()]
        assert len(events) == 1
        assert events[0]["event"] == "begin"
        assert events[0]["request_extras"] == {
            "reasoning": True, "temperature": 0.0,
        }

    def test_request_extras_omitted_when_none(self):
        # Pre-this-PR call shape — no request_extras passed; the begin
        # event must NOT grow a key (consumers reading old logs alongside
        # new logs need the field-presence to mean "the wrapper recorded
        # it" rather than "always present").
        llm.begin_stat_record("extract", "doc-1", "kimi-k2-6")
        events = [json.loads(l) for l in
                  self.jsonl.read_text().splitlines() if l.strip()]
        assert len(events) == 1
        assert "request_extras" not in events[0]


# ── _materialize_calls_from_jsonl propagates request_extras ───────────────────


class TestMaterializerPropagatesRequestExtras:
    def test_request_extras_lands_on_materialized_record(self, tmp_path):
        p = tmp_path / "llm-calls.jsonl"
        events = [
            {"event": "begin", "call_id": "0001",
             "stage": "patterns", "category": "topic-a",
             "model": "kimi-k2-6",
             "started_at_iso": "2026-04-29T17:09:00.000Z",
             "request_extras": {"reasoning": True, "temperature": 0}},
            {"event": "end", "call_id": "0001",
             "duration_ms": 1234, "success": True, "error": None,
             "prompt_tokens": 100, "completion_tokens": 50,
             "model": "kimi-k2-6", "mode": "tee"},
        ]
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        records = runner._materialize_calls_from_jsonl(
            p, "2026-04-29T17:10:00.000Z")
        assert len(records) == 1
        assert records[0]["request_extras"] == {
            "reasoning": True, "temperature": 0,
        }

    def test_pre_this_pr_records_have_none_request_extras(self, tmp_path):
        # An older jsonl (no request_extras on begin) materializes with
        # request_extras=None, NOT a missing key. Consumers can rely on
        # the field always being present; None signals "unrecorded".
        p = tmp_path / "llm-calls.jsonl"
        events = [
            {"event": "begin", "call_id": "0001",
             "stage": "extract", "category": "doc-1",
             "model": "kimi-k2-6",
             "started_at_iso": "2026-04-29T17:09:00.000Z"},
            {"event": "end", "call_id": "0001",
             "duration_ms": 100, "success": True, "error": None,
             "prompt_tokens": 10, "completion_tokens": 5,
             "model": "kimi-k2-6", "mode": "tee"},
        ]
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        records = runner._materialize_calls_from_jsonl(
            p, "2026-04-29T17:10:00.000Z")
        assert records[0]["request_extras"] is None


# ── _set_run_subject_resolution / _set_run_bundle_mode ────────────────────────


class TestRunConfigPostStageStamps:
    """The entities stage stamps subject_resolution + bundle_mode onto
    run.json's run_config so post-hoc analysis can answer "which subject
    branch did this run take" without scraping intermediate/."""

    def test_subject_resolution_lands_on_run_config(self):
        runner._run_state = {"run_config": {"mode": "test"}}
        runner._set_run_subject_resolution({
            "canonical_id": "alice",
            "display": "Alice",
            "source": "alias_match",
        })
        assert runner._run_state["run_config"]["subject_resolution"] == {
            "canonical_id": "alice",
            "display": "Alice",
            "source": "alias_match",
        }

    def test_bundle_mode_lands_on_run_config(self):
        runner._run_state = {"run_config": {"mode": "test"}}
        runner._set_run_bundle_mode(True)
        assert runner._run_state["run_config"]["bundle_mode"] is True
        runner._set_run_bundle_mode(False)
        assert runner._run_state["run_config"]["bundle_mode"] is False

    def test_subject_resolution_no_op_when_run_state_unset(self):
        # Tests / ad-hoc scripts may call this outside a run; must
        # not raise.
        runner._run_state = None
        runner._set_run_subject_resolution({"canonical_id": "x"})
        # Still None — no exception.
        assert runner._run_state is None


# ── llm-stats per_stage carries reasoning_enabled ─────────────────────────────


class TestLlmStatsPerStageReasoning:
    """The rollup aggregates per-call request_extras.reasoning into a
    per-stage `reasoning_enabled` flag so a debug-bundle reader can
    answer "which stages had reasoning ON" from the materialized
    payload (issue #189)."""

    def _write_jsonl(self, p, events):
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")

    def test_per_stage_reasoning_enabled_true_when_any_call_had_it(
        self, tmp_path
    ):
        out = tmp_path / "out"
        out.mkdir()
        # Two calls in `patterns`, both reasoning ON.
        events = [
            {"event": "cycle_start", "ts": "2026-04-29T17:09:00.000Z",
             "run_id": "test-run", "is_resume": False, "cycle_seq": 1},
            {"event": "begin", "call_id": "0001",
             "stage": "patterns", "category": "topic-a",
             "model": "glm-5-2",
             "started_at_iso": "2026-04-29T17:09:00.000Z",
             "request_extras": {"reasoning": True, "temperature": 0}},
            {"event": "end", "call_id": "0001",
             "duration_ms": 1000, "success": True, "error": None,
             "prompt_tokens": 100, "completion_tokens": 50,
             "model": "glm-5-2", "mode": "tee"},
            {"event": "begin", "call_id": "0002",
             "stage": "patterns", "category": "topic-b",
             "model": "glm-5-2",
             "started_at_iso": "2026-04-29T17:09:01.000Z",
             "request_extras": {"reasoning": True, "temperature": 0}},
            {"event": "end", "call_id": "0002",
             "duration_ms": 1000, "success": True, "error": None,
             "prompt_tokens": 100, "completion_tokens": 50,
             "model": "glm-5-2", "mode": "tee"},
        ]
        self._write_jsonl(out / "llm-calls.jsonl", events)
        rollup = runner.materialize_run_stats(
            out / "llm-calls.jsonl", out / "config.json",
        )
        per_stage = rollup["per_stage"]
        assert per_stage["patterns"]["reasoning_enabled"] is True
        assert per_stage["patterns"]["reasoning_mixed"] is False

    def test_per_stage_reasoning_enabled_false_when_no_call_had_it(
        self, tmp_path
    ):
        out = tmp_path / "out"
        out.mkdir()
        events = [
            {"event": "cycle_start", "ts": "2026-04-29T17:09:00.000Z",
             "run_id": "test-run", "is_resume": False, "cycle_seq": 1},
            {"event": "begin", "call_id": "0001",
             "stage": "extract", "category": "doc-1",
             "model": "kimi-k2-6",
             "started_at_iso": "2026-04-29T17:09:00.000Z",
             "request_extras": {"reasoning": False, "temperature": 0}},
            {"event": "end", "call_id": "0001",
             "duration_ms": 100, "success": True, "error": None,
             "prompt_tokens": 10, "completion_tokens": 5,
             "model": "kimi-k2-6", "mode": "tee"},
        ]
        self._write_jsonl(out / "llm-calls.jsonl", events)
        rollup = runner.materialize_run_stats(
            out / "llm-calls.jsonl", out / "config.json",
        )
        assert rollup["per_stage"]["extract"]["reasoning_enabled"] is False

    def test_text_summary_annotates_reasoning_models(self, tmp_path):
        out = tmp_path / "out"
        out.mkdir()
        events = [
            {"event": "cycle_start", "ts": "2026-04-29T17:09:00.000Z",
             "run_id": "test-run", "is_resume": False, "cycle_seq": 1},
            {"event": "begin", "call_id": "0001",
             "stage": "patterns", "category": "topic-a",
             "model": "glm-5-2",
             "started_at_iso": "2026-04-29T17:09:00.000Z",
             "request_extras": {"reasoning": True, "temperature": 0}},
            {"event": "end", "call_id": "0001",
             "duration_ms": 1000, "success": True, "error": None,
             "prompt_tokens": 100, "completion_tokens": 50,
             "model": "glm-5-2", "mode": "tee"},
            {"event": "begin", "call_id": "0002",
             "stage": "extract", "category": "doc-1",
             "model": "kimi-k2-6",
             "started_at_iso": "2026-04-29T17:09:01.000Z",
             "request_extras": {"reasoning": False, "temperature": 0}},
            {"event": "end", "call_id": "0002",
             "duration_ms": 100, "success": True, "error": None,
             "prompt_tokens": 10, "completion_tokens": 5,
             "model": "kimi-k2-6", "mode": "tee"},
        ]
        (out / "llm-calls.jsonl").write_text(
            "\n".join(json.dumps(e) for e in events) + "\n"
        )
        runner._reset_llm_stats_dump_state()
        runner._run_name = "test-run"
        try:
            runner._do_write_llm_stats(
                out, llm.Mode.TEE, "kimi-k2-6",
                "2026-04-29T17:09:00.000Z",
            )
        finally:
            runner._run_name = None
        text = (out / "llm-stats.txt").read_text()
        # The glm line carries the reasoning marker; the kimi line
        # does not.
        glm_lines = [l for l in text.splitlines()
                          if "glm-5-2" in l and "calls" in l]
        kimi_lines = [l for l in text.splitlines()
                      if "kimi-k2-6" in l and "calls" in l]
        assert any("reasoning ON" in l for l in glm_lines), glm_lines
        assert all("reasoning ON" not in l for l in kimi_lines), kimi_lines

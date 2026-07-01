"""The shareable content-free diagnostics trust contract.

These tests pin the four legs of the by-construction guarantee:

1. **Single writer.** ``_shareable_root()`` is the only site in the
   pipeline tree that joins ``shareable``; ``emit()`` is the only
   public writer and takes only a typed ``Marker``.
2. **Typed input + runtime assert.** A clean marker passes; any free
   string, dict, unknown-enum, or foreign type *crashes* (never a
   silent write).
3. **Never the raw record id.** A candidate row carries a marker-local
   positional ordinal, and there is no ``record_id`` field anywhere in
   the marker tree.
4. **Perma-id keying.** The filename leads with the validated 4-letter
   perma-id; ``resolve_perma_id`` reads the persisted ``short_id``
   (run ``config.json`` / chat ``transcript.json``) or the canonical
   dir-name suffix, and returns ``None`` for a legacy non-conforming
   dir that predates the scheme.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path

import pytest

from engine import shareable as s
from engine import shareable_markers as sm


def _stage_stat(**over) -> s.StageStat:
    """A clean StageStat fixture — every required field filled with a
    safe default so a test can dataclasses.replace the one or two
    fields it actually cares about."""
    kw = dict(
        stage=s.StageToken.extraction,
        stage_index=0,
        present=True,
        completed=True,
        success=True,
        started_at=None,
        ended_at=None,
        wall_ms=None,
        call_count=0,
        success_count=0,
        failure_count=0,
        cache_hit_count=0,
        retry_count=0,
        prompt_tokens_sum=0,
        completion_tokens_sum=0,
        item_count=None,
        duration_ms_p50=None,
        duration_ms_p95=None,
        duration_ms_mean=None,
        ttft_ms_p50=None,
        ttft_ms_p95=None,
        outcome_dist=(),
        retry_class_dist=(),
        calls=(),
        successful_calls_sampled=False,
    )
    kw.update(over)
    return s.StageStat(**kw)


def _llm_call(**over) -> s.LlmCall:
    kw = dict(
        stage=s.StageToken.chatbot,
        category=s.CategoryToken.chatbot_answer,
        model=s.ModelToken.kimi_k2,
        provider=s.Provider.tee,
        reasoning=False,
        outcome=s.Outcome.success,
        prompt_tokens=10, completion_tokens=20,
        reasoning_tokens=0, content_tokens=20, total_tokens=30,
        duration_ms=12.5, ttft_ms=4.0, max_tokens_reserved=4096,
        attempt=1, is_retry=False, parse_error=False,
        started_at=s._now_iso_z(), ended_at=s._now_iso_z(),
        call_id="0001", cached=False, finish_reason=s.FinishReason.stop,
        retry_of_call_id=None,
        retry_class=s.RetryClass.none, retry_transforms=(),
    )
    kw.update(over)
    return s.LlmCall(**kw)


def _clean_lookup_shape(**over) -> s.LookupShape:
    """A minimal valid LookupShape — query-bearing fact lookup with
    clean dispatch outcome. Tests use ``dataclasses.replace`` to
    perturb individual fields."""
    kw = dict(
        entry_types=(s.RecordKind.fact,),
        k_requested=15,
        has_neighbor_count=0,
        has_neighbor_kinds=(),
        exact_match_count=0,
        query_present=True,
        query_char_len=42,
        embed_dim=768,
        embed_norm=1.0,
        embed_all_zero=False,
        embed_non_finite=False,
        degenerate=False,
        tied=False,
        k_returned=10,
        junk_dropped=0,
        kind_counts=(s.KindCount(s.RecordKind.fact, 10),),
        score_samples=s.ScoreSamples(closest=0.2, middle=0.4, farthest=0.6),
    )
    kw.update(over)
    return s.LookupShape(**kw)


def _clean_hop_marker(**over) -> s.HopMarker:
    """A minimal valid HopMarker — a single grounded_decision hop with
    a one-element lookups array."""
    kw = dict(
        call_id="0002",
        hop_outcome=s.HopOutcome.tool_call,
        streamed_to_user=False,
        lookups_remaining_in_budget=3,
        previous_attempts_count=0,
        store_open_latency_ms=2.5,
        dispatch_latency_ms=47.0,
        union_size_after=10,
        lookups=(_clean_lookup_shape(),),
    )
    kw.update(over)
    return s.HopMarker(**kw)


def _clean_chat_marker() -> s.ChatMarker:
    return s.ChatMarker(
        schema_version=s.SCHEMA_VERSION,
        ts=s._now_iso_z(),
        turn_index=3,
        session_id="0123456789abcdef",
        lookup_fired=True,
        llm_calls=s.LlmCallsBlock(
            call_count=1,
            total_prompt_tokens=10,
            total_completion_tokens=20,
            wall_ms_total=12.5,
            calls=(_llm_call(),),
        ),
        hops=(_clean_hop_marker(),),
    )


# ── leg 2: typed input + runtime assert ───────────────────────────────


def test_clean_marker_passes_guard():
    s._assert_content_free(_clean_chat_marker())


@pytest.mark.parametrize("bad_ts", [
    "obviously free text",
    "2026-05-17 09:00:00",          # space, no Z — not ISO-Z
    "Mom's medical history notes",  # the leak we exist to stop
])
def test_free_string_leaf_crashes(bad_ts):
    bad = dataclasses.replace(_clean_chat_marker(), ts=bad_ts)
    with pytest.raises(s.ContentFreeViolation):
        s._assert_content_free(bad)


def test_dict_leaf_crashes():
    # A dict is the classic free-text smuggling channel — unrepresentable
    # in the typed tree, and the guard rejects it if forced in.
    @dataclasses.dataclass(frozen=True)
    class _Sneak(s.Marker):
        payload: dict

    bad = _Sneak(schema_version=1, payload={"query": "secret vault text"})
    with pytest.raises(s.ContentFreeViolation):
        s._assert_content_free(bad)


def test_unknown_enum_value_crashes():
    class _Rogue(str):
        pass

    # Smuggle a non-enum, non-ISO-Z str into a LookupShape's
    # query_char_len (an int slot) — the guard's recursive walk hits
    # the str leaf inside hops[0].lookups[0] and rejects it.
    bad_lookup = dataclasses.replace(
        _clean_lookup_shape(), query_char_len=_Rogue("not-a-number"),
    )
    bad_hop = dataclasses.replace(
        _clean_hop_marker(), lookups=(bad_lookup,),
    )
    bad = dataclasses.replace(_clean_chat_marker(), hops=(bad_hop,))
    with pytest.raises(s.ContentFreeViolation):
        s._assert_content_free(bad)


def test_iso_z_and_enum_strings_are_the_only_allowed_strings():
    assert s._ISO_Z_RE.match("2026-05-17T09:00:00Z")
    assert s._ISO_Z_RE.match("2026-05-17T09:00:00.123Z")
    assert not s._ISO_Z_RE.match("2026-05-17T09:00:00")
    # every closed-enum value is in the vocabulary
    for tok in (s.ModelToken.kimi_k2, s.StageToken.embeddings,
                s.CategoryToken.chatbot_answer, s.RecordKind.fact):
        assert tok.value in s._ENUM_VOCAB


# ── leg 3: never the raw record id ────────────────────────────────────


def test_no_record_id_field_anywhere_in_marker_tree():
    seen = set()

    def walk(cls):
        if cls in seen or not dataclasses.is_dataclass(cls):
            return
        seen.add(cls)
        for f in dataclasses.fields(cls):
            assert "record_id" not in f.name, (
                f"{cls.__name__}.{f.name} must never carry record_id"
            )
            inner = f.type
            for c in (s.CandidateRow, s.LlmCall, s.StageStat,
                      s.QueryStats, s.LlmCallsBlock, s.RunMarker,
                      s.ChatMarker, s.KindCount, s.SizeHistBucket):
                if c.__name__ in str(inner):
                    walk(c)

    walk(s.ChatMarker)
    walk(s.RunMarker)


def test_candidate_row_uses_positional_ordinal_not_record_id():
    names = {f.name for f in dataclasses.fields(s.CandidateRow)}
    assert "position" in names
    assert "record_id" not in names


# ── leg 4: perma-id keying ────────────────────────────────────────────


def test_perma_id_regex_matches_the_576_alphabet():
    assert s._PERMA_ID_RE.match("j5dh")
    assert s._PERMA_ID_RE.match("ab3k")
    # excluded ambiguous letters / wrong length rejected
    assert not s._PERMA_ID_RE.match("ab1k")   # '1' not in alphabet
    assert not s._PERMA_ID_RE.match("abcd1")  # length 5
    assert not s._PERMA_ID_RE.match("ABCD")   # uppercase


def test_resolve_perma_id_from_run_config(tmp_path: Path):
    d = tmp_path / "2026-05-17T09-00-00Z-zzzz"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"short_id": "ab3k"}))
    assert s.resolve_perma_id(d) == "ab3k"


def test_resolve_perma_id_from_chat_transcript(tmp_path: Path):
    d = tmp_path / "2026-05-17T09-00-00Z-my chat"
    d.mkdir()
    (d / "transcript.json").write_text(json.dumps({"short_id": "qmkp"}))
    assert s.resolve_perma_id(d) == "qmkp"


def test_resolve_perma_id_dir_suffix_fallback(tmp_path: Path):
    # A run dir whose config.json predates the short_id field still
    # surfaces the 4-letter from the canonical dir-name suffix.
    d = tmp_path / "2026-05-17T09-00-00Z-ab3k"
    d.mkdir()
    assert s.resolve_perma_id(d) == "ab3k"


def test_resolve_perma_id_merged_model_chat_dir_resolves(tmp_path: Path):
    # Post perma-id model: the conversation dir is <iso-z>-<short_id>
    # and transcript.json carries short_id, so the chat side now
    # RESOLVES (no longer a no-op). Both sidecar and dir-suffix work.
    d = tmp_path / "2026-05-17T09-00-00Z-qmkp"
    d.mkdir()
    (d / "transcript.json").write_text(
        json.dumps({"open": True, "turns": [], "short_id": "qmkp"})
    )
    assert s.resolve_perma_id(d) == "qmkp"
    # dir-suffix alone (transcript not yet written) still resolves
    d2 = tmp_path / "2026-05-17T09-00-01Z-ab3k"
    d2.mkdir()
    assert s.resolve_perma_id(d2) == "ab3k"


def test_resolve_perma_id_none_for_legacy_nonconforming_dir(tmp_path: Path):
    # A legacy dir that predates the scheme (no sidecar short_id, no
    # 4-letter suffix) resolves to None; the caller best-effort skips.
    d = tmp_path / "2026-05-17T09-00-00Z-grocery list"
    d.mkdir()
    assert s.resolve_perma_id(d) is None
    assert s.resolve_perma_id(None) is None


# ── leg 1 + write path: single guarded writer ─────────────────────────


def test_emit_rejects_bad_key_and_non_marker(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(ValueError):
        s.emit(s.Stream.CHAT, "BAD!", _clean_chat_marker())
    with pytest.raises(ValueError):
        s.emit(s.Stream.CHAT, "j5dh", object())  # type: ignore[arg-type]


def test_emit_one_file_per_perma_id_yaml_multidoc(tmp_path, monkeypatch):
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    s.emit(s.Stream.CHAT, "j5dh", _clean_chat_marker())
    s.emit(s.Stream.CHAT, "j5dh", _clean_chat_marker())
    base = tmp_path / ".basevault" / "shareable" / "chat-diagnostics"
    # Exactly ONE file for the perma-id (consolidated, not per-turn).
    files = list(base.glob("*-j5dh-anonymized.yaml"))
    assert len(files) == 1, files
    name = files[0].name
    # Date-first: <iso-z>-<perma-id>-anonymized.yaml
    assert re.match(r"^2\d{3}-.*Z-j5dh-anonymized\.yaml$", name), name
    text = files[0].read_text()
    # YAML multi-doc, legible as-is (no trick): header doc + one doc
    # per turn, '---'-separated. Valid YAML to a parser.
    assert text.startswith("---\n")
    docs = list(yaml.safe_load_all(text))
    # header + 2 turns
    assert len(docs) == 3
    header = docs[0]
    assert header["schema_version"] == s.SCHEMA_VERSION
    assert header["perma_id"] == "j5dh"
    assert header["stream"] == "chat-diagnostics"
    # The chat marker now carries per-hop ReAct trace, not a flat
    # candidates block. Verify the hop joined into the yaml.
    assert docs[1]["hops"][0]["hop_outcome"] == "tool_call"
    assert docs[2]["turn_index"] == _clean_chat_marker().turn_index
    # A third turn is a pure append — header/created_at unchanged.
    created = header["created_at"]
    s.emit(s.Stream.CHAT, "j5dh", _clean_chat_marker())
    text3 = files[0].read_text()
    docs3 = list(yaml.safe_load_all(text3))
    assert len(docs3) == 4
    assert docs3[0]["created_at"] == created
    # No free text anywhere — never the raw record_id.
    assert "record_id" not in text3


def _run_marker() -> s.RunMarker:
    return s.RunMarker(
        schema_version=s.SCHEMA_VERSION,
        created_at=s._now_iso_z(),
        embedding=s.EmbeddingStats(
            records_embedded=686, embed_dim=768,
            embed_outcome=s.Outcome.success,
            embed_call_count=12, batch_size=64,
        ),
        record_counts_by_kind=(
            s.KindCount(s.RecordKind.chunk, 466),
            s.KindCount(s.RecordKind.fact, 129),
        ),
        chunk_size_hist=(s.SizeHistBucket(s.SizeBucket.lt_200, 57),),
        file_size_hist=(s.SizeHistBucket(s.SizeBucket.gte_2k, 0),),
        stages=(
            _stage_stat(
                stage=s.StageToken.extraction, stage_index=0,
                item_count=42, call_count=1, success_count=1,
                prompt_tokens_sum=900, completion_tokens_sum=120,
                duration_ms_p50=3100.0, duration_ms_p95=3100.0,
                duration_ms_mean=3100.0,
                ttft_ms_p50=210.0, ttft_ms_p95=210.0,
                outcome_dist=(s.OutcomeCount(s.Outcome.success, 1),),
                calls=(_llm_call(
                    stage=s.StageToken.extraction,
                    category=s.CategoryToken.other,
                    prompt_tokens=900, completion_tokens=120,
                    content_tokens=120, total_tokens=1020,
                    duration_ms=3100.0, ttft_ms=210.0,
                ),),
                successful_calls_sampled=True,
            ),
            _stage_stat(
                stage=s.StageToken.embeddings, stage_index=6,
                present=True, completed=False, success=False,
                item_count=None,
            ),
        ),
        # Run-scope llm_calls keeps the four-int rollup only — the
        # per-call detail moved under each StageStat.calls.
        llm_calls=s.LlmCallsBlock(
            call_count=1, total_prompt_tokens=900,
            total_completion_tokens=120, wall_ms_total=3100.0,
            calls=(),
        ),
    )


def test_run_stream_written_once_not_duplicated_per_turn(
    tmp_path, monkeypatch
):
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    # Three "turns" all try to emit the run/corpus file.
    s.emit(s.Stream.RUN, "qxj6", _run_marker())
    s.emit(s.Stream.RUN, "qxj6", _run_marker())
    s.emit(s.Stream.RUN, "qxj6", _run_marker())
    base = tmp_path / ".basevault" / "shareable" / "run-diagnostics"
    files = list(base.glob("*-qxj6-anonymized.yaml"))
    assert len(files) == 1, files
    docs = list(yaml.safe_load_all(files[0].read_text()))
    # header + EXACTLY ONE run doc — no per-turn duplication.
    assert len(docs) == 2
    run = docs[1]
    assert run["embedding"]["records_embedded"] == 686
    assert {k["kind"] for k in run["record_counts_by_kind"]} == {
        "chunk", "fact"
    }
    assert len(run["stages"]) == 2


def test_runner_hook_emits_run_driven_and_is_idempotent(
    tmp_path, monkeypatch
):
    """A completed run drops its content-free run/corpus file with NO
    chat involved (run-driven), via the same single guarded emitter,
    and re-running the hook never duplicates it."""
    import yaml

    from engine import runner

    monkeypatch.setenv("HOME", str(tmp_path))
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-zzzz"
    (run_dir / "stages" / "00-ingestion").mkdir(parents=True)
    (run_dir / "stages" / "06-embeddings").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "zzzz"}))
    (run_dir / "stages" / "00-ingestion" / "phase_1_marker.json").write_text(
        "{}"
    )

    runner._emit_shareable_run_marker(run_dir)
    runner._emit_shareable_run_marker(run_dir)  # idempotent

    base = tmp_path / ".basevault" / "shareable" / "run-diagnostics"
    files = list(base.glob("*-zzzz-anonymized.yaml"))
    assert len(files) == 1, files
    docs = list(yaml.safe_load_all(files[0].read_text()))
    assert len(docs) == 2  # header + one run doc, not duplicated
    run = docs[1]
    # Run schema, no CHAT-turn fields leaked in. (No llm_calls here
    # only because the synthetic run dir has no llm-calls.jsonl — the
    # run's own pipeline calls are wired separately and tested below.)
    assert "stages" in run
    assert "turn_index" not in run and "candidates" not in run
    assert "query" not in run and "rerank" not in run


def test_emit_run_diagnostic_cli_emits_for_cancelled_partial_run(
    tmp_path, monkeypatch
):
    """`runner.py --emit-run-diagnostic <run_dir>` — the one-shot the Rust
    cancel-settle path spawns — drops the run-diagnostics shareable for an
    already-terminal run, capturing PARTIAL state. The contract: cancel a
    run → the diagnostic IS produced (a canceled run is high-signal). A
    canceled mid-run has no embeddings (embed_outcome
    'unknown') yet still itemizes its own content-free pipeline LLM calls —
    the breakdown you want when you stopped because something looked wrong.
    """
    import sys
    import yaml

    from engine import runner

    monkeypatch.setenv("HOME", str(tmp_path))
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-cxkd"
    (run_dir / "stages" / "00-ingestion").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "cxkd"}))
    # A canceled mid-run: ingestion ran + one completed extract call, then
    # a user cancel. No embeddings stage → partial.
    events = [
        {"event": "cycle_start", "ts": "2026-05-17T09:00:00Z", "cycle_seq": 1},
        {"event": "begin", "call_id": "0001", "stage": "extract",
         "model": "kimi-k2-6", "mode": "tee"},
        {"event": "end", "call_id": "0001", "stage": "extract",
         "success": True, "prompt_tokens": 120, "completion_tokens": 40,
         "model": "kimi-k2-6", "mode": "tee"},
        {"event": "cycle_cancelled", "ts": "2026-05-17T09:00:30Z",
         "reason": "user"},
    ]
    (run_dir / "llm-calls.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n"
    )

    monkeypatch.setattr(
        sys, "argv", ["runner.py", "--emit-run-diagnostic", str(run_dir)]
    )
    runner.main()

    base = tmp_path / ".basevault" / "shareable" / "run-diagnostics"
    files = list(base.glob("*-cxkd-anonymized.yaml"))
    assert len(files) == 1, files
    docs = list(yaml.safe_load_all(files[0].read_text()))
    run = docs[1]
    # Partial run: embeddings never ran — record_counts_by_kind is
    # empty, so ``embed_outcome`` is None and ``_to_jsonable`` omits
    # the key entirely (neutral / no-signal, not a failure).
    assert "embed_outcome" not in run.get("embedding", {})
    # The run's own content-free pipeline calls are captured.
    assert run["llm_calls"]["call_count"] == 1
    # Content-free: no chat-turn fields, no raw text/ids leaked in.
    assert "query" not in run and "candidates" not in run


def test_run_shareable_is_latest_wins_overwrite(tmp_path, monkeypatch):
    """The RUN stream is latest-wins: a later emit for the same run
    perma-id REPLACES the prior file rather than no-opping (write-once) or
    appending. This is what lets a partial pause snapshot be superseded by
    the eventual resume→complete one — still exactly one file, carrying
    the most recent state."""
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    first = dataclasses.replace(
        _run_marker(),
        embedding=dataclasses.replace(
            _run_marker().embedding, records_embedded=10
        ),
    )
    s.emit(s.Stream.RUN, "ab3k", first)            # partial
    s.emit(s.Stream.RUN, "ab3k", _run_marker())    # fuller (686), overwrites

    base = tmp_path / ".basevault" / "shareable" / "run-diagnostics"
    files = list(base.glob("*-ab3k-anonymized.yaml"))
    assert len(files) == 1, files                  # one file, not two
    docs = list(yaml.safe_load_all(files[0].read_text()))
    assert len(docs) == 2                           # header + one run doc
    # Latest wins: the second emit's value, not the first.
    assert docs[1]["embedding"]["records_embedded"] == 686


def test_run_and_chat_schemas_carry_no_duplicated_info():
    chat_fields = {f.name for f in dataclasses.fields(s.ChatMarker)}
    run_fields = {f.name for f in dataclasses.fields(s.RunMarker)}
    # The only shared field NAMES are schema_version and llm_calls —
    # and llm_calls is NOT duplicated info: chat's is the conversation
    # turn's calls (from the convo's llm-calls.jsonl), run's is the
    # run's OWN pipeline-stage calls (from the run dir's
    # llm-calls.jsonl). Different source, different scope, never the
    # same data.
    assert (chat_fields & run_fields) - {"schema_version", "llm_calls"} \
        == set()
    # Chat-only (turn) fields never appear in run; run-only (corpus)
    # fields never appear in chat — the real no-duplication invariant.
    # The flat query/shape/distances/rerank/candidates block was
    # dropped in #781's multi-hop rev (replaced by per-hop trace).
    for chat_only in (
        "turn_index", "lookup_fired", "session_id", "hops",
        "retrieve_skipped_reason", "bound_run", "store_stats",
        "history_turn_count", "resources_emitted_count",
    ):
        assert chat_only in chat_fields and chat_only not in run_fields
    for run_only in (
        "stages", "record_counts_by_kind", "embedding",
        "chunk_size_hist", "file_size_hist", "created_at",
    ):
        assert run_only in run_fields and run_only not in chat_fields


def test_yaml_is_succinct_no_null_or_empty_noise(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    # A no-lookup turn: hops empty, bound_run None, store_stats None,
    # latencies None — they must be OMITTED, not emitted as `: null`.
    m = s.ChatMarker(
        schema_version=s.SCHEMA_VERSION, ts=s._now_iso_z(), turn_index=1,
        session_id="0123456789abcdef",
        lookup_fired=False, llm_calls=s.LlmCallsBlock(0, 0, 0, 0.0, ()),
        hops=(),
    )
    s.emit(s.Stream.CHAT, "j5dh", m)
    f = next((tmp_path / ".basevault" / "shareable"
              / "chat-diagnostics").glob("*-j5dh-anonymized.yaml"))
    text = f.read_text()
    assert "null" not in text
    # The dropped #818 fields stay dropped — the multi-hop rev moved
    # query/shape/distances/rerank/candidates into the per-hop tuple.
    assert "query" not in text and "shape" not in text
    assert "candidates" not in text and "distances" not in text
    # And the new tuple is empty on a no-lookup turn → omitted.
    assert "hops:" not in text


def test_shareable_root_is_sibling_of_cache_not_under_logs(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = s._shareable_root()
    assert root.name == "shareable"
    assert root.parent.name == ".basevault"
    assert root.parent == (s._state_root())
    # never nested under logs/ (the locked-strategy invariant)
    assert "logs" not in root.parts


def test_only_shareable_module_joins_the_shareable_root():
    """Single-writer enforcement: no other pipeline source may name a
    path under ``shareable/``. Pins the trust boundary against drift."""
    pipeline = Path(__file__).resolve().parent.parent
    # Match a string/path join that introduces the shareable root, e.g.
    # `/ "shareable"`, `.join("shareable")`, `"shareable/..."`.
    pat = re.compile(r"""["']shareable(?:/|["'])""")
    offenders = []
    for py in pipeline.rglob("*.py"):
        if py.name in ("shareable.py", "shareable_markers.py"):
            continue
        if "/tests/" in py.as_posix():
            continue
        txt = py.read_text(encoding="utf-8", errors="ignore")
        if pat.search(txt):
            offenders.append(py.relative_to(pipeline).as_posix())
    assert not offenders, (
        f"only shareable.py may join the shareable root; found: {offenders}"
    )


def test_emit_is_the_only_public_writer_and_takes_a_typed_marker():
    import inspect

    sig = inspect.signature(s.emit)
    params = list(sig.parameters)
    assert params == ["stream", "perma_id", "marker"], params
    assert sig.parameters["marker"].annotation in (s.Marker, "Marker")
    # no other public function in shareable.py opens a file for writing
    src = (Path(s.__file__).read_text())
    assert src.count("os.open(") == 1, "exactly one write syscall site"


# ── builders stay content-free end to end ─────────────────────────────


def test_build_llm_calls_block_is_content_free(tmp_path):
    calls = tmp_path / "llm-calls.jsonl"
    calls.write_text(
        json.dumps({"event": "begin", "call_id": "0001",
                    "stage": "chatbot", "category": "chatbot_answer",
                    "model": "moonshotai/Kimi-K2.6-TEE",
                    "started_at_iso": "2026-05-17T09:00:00.001Z"}) + "\n" +
        json.dumps({"event": "end", "call_id": "0001", "success": True,
                    "error": None, "duration_ms": 1200,
                    "prompt_tokens": 100, "completion_tokens": 50,
                    "model": "moonshotai/Kimi-K2.6-TEE", "mode": "tee",
                    "ts": "2026-05-17T09:00:01Z"}) + "\n"
    )
    block = sm.build_llm_calls_block(calls, since_call_id=0)
    s._assert_content_free(block)
    assert block.call_count == 1
    c = block.calls[0]
    assert c.model is s.ModelToken.kimi_k2
    assert c.outcome is s.Outcome.success
    assert c.prompt_tokens == 100


def test_retry_and_cache_extracted_content_free(tmp_path):
    # The category embeds a filename/date/entry prefix (vault content)
    # then the structural retry chain. Only closed tokens from AFTER
    # the first '/' may surface; nothing before it, ever.
    poison = "026-05-09 - Journal.stripped-2025.json::entry_353"
    calls = tmp_path / "llm-calls.jsonl"
    calls.write_text(
        json.dumps({
            "event": "begin", "call_id": "0007",
            "stage": "extract",
            "category": f"{poison}/half-1/half-2 - retry/sizing",
            "retry_of_call_id": "0003", "retry_delay_ms": 250,
            "started_at_iso": "2026-05-17T09:00:00.001Z",
        }) + "\n" +
        json.dumps({
            "event": "end", "call_id": "0007", "success": True,
            "error": None, "duration_ms": 1800, "prompt_tokens": 700,
            "completion_tokens": 90, "cached": True,
            "cache_key": "deadbeefcafebabe", "finish_reason": "stop",
            "model": "gpt-oss-120b", "mode": "tee",
            "ts": "2026-05-17T09:00:02Z",
        }) + "\n"
    )
    block = sm.build_llm_calls_block(calls, since_call_id=0)
    s._assert_content_free(block)  # would crash on any free string
    c = block.calls[0]
    assert c.stage is s.StageToken.extraction          # alias mapped
    assert c.category is s.CategoryToken.other          # filename → other
    # Runner's standard zero-padded 4-digit form, preserved verbatim.
    assert c.call_id == "0007" and c.retry_of_call_id == "0003"
    assert c.cached is True
    assert c.finish_reason is s.FinishReason.stop
    assert c.retry_class is s.RetryClass.sizing
    assert s.RetryTransform.half in c.retry_transforms
    # Nothing from the content prefix or the cache key leaked.
    import yaml
    blob = yaml.safe_dump(s._to_jsonable(block))
    for poison_bit in ("Journal", "stripped", "entry_353", "deadbeef",
                       "026-05-09"):
        assert poison_bit not in blob


def test_build_llm_calls_block_slices_by_baseline(tmp_path):
    calls = tmp_path / "llm-calls.jsonl"
    calls.write_text(
        json.dumps({"event": "begin", "call_id": "0001",
                    "stage": "chatbot"}) + "\n" +
        json.dumps({"event": "begin", "call_id": "0002",
                    "stage": "rerank"}) + "\n"
    )
    block = sm.build_llm_calls_block(calls, since_call_id=1)
    assert [c.stage for c in block.calls] == [s.StageToken.rerank]


def test_build_chat_marker_never_emits_record_id(tmp_path):
    # The per-record candidates block of #818 (which DID carry per-row
    # rank+kind+distance+rerank_score) was dropped in #781's multi-hop
    # rewrite; the marker now never carries any per-record row at all —
    # only per-lookup ``kind_counts`` and ``score_samples``. So the
    # ``record_id``-never-leaks invariant simplifies: no path in the
    # marker tree even has a record_id-shaped field. Pin that.
    marker = sm.build_chat_marker(
        turn_index=1, session_id="0123456789abcdef",
        lookup_fired=True, hops_diag=[
            {
                "hop_outcome": "tool_call",
                "streamed_to_user": False,
                "lookups_remaining_in_budget": 3,
                "previous_attempts_count": 0,
                "store_open_latency_ms": 1.5,
                "dispatch_latency_ms": 30.0,
                "union_size_after": 1,
                "per_lookup": [{
                    "entry_types": ["fact"], "k_requested": 15,
                    "has_neighbor_count": 0, "has_neighbor_kinds": [],
                    "exact_match_count": 0, "query_present": True,
                    "query_char_len": 5, "embed_dim": 768,
                    "embed_norm": 1.0, "embed_all_zero": False,
                    "embed_non_finite": False,
                    "degenerate": False, "tied": False,
                    "k_returned": 1, "junk_dropped": 0,
                    # The dispatcher emits ``kind_counts`` as a closed-
                    # enum-keyed int dict — even if a value here were
                    # vault text (it never is, but defense in depth),
                    # the builder maps through ``RecordKind`` and drops
                    # unknowns, so no free string can sneak through.
                    "kind_counts": {"fact": 1},
                    "score_samples": (0.42, 0.42, 0.42),
                }],
            },
        ],
        llm_calls=s.LlmCallsBlock(0, 0, 0, 0.0, ()),
    )
    s._assert_content_free(marker)
    blob = json.dumps(s._to_jsonable(marker))
    # No record_id-shaped field exists anywhere in the marker.
    assert "record_id" not in blob
    # No vault text could have leaked even if the test fixture
    # contained any.
    assert "Divorce" not in blob


# ── #782 run-diagnostics compression: per-stage rollup + sampling ────


def _stage_calls_jsonl(
    path: Path, stage: str, model: str, count: int, *,
    outcomes: list[bool] | None = None,
    base_minute: int = 0,
    cid_offset: int = 0,
) -> None:
    """Write `count` paired begin/end events for `stage` into `path`.

    Each call gets a stable ISO-Z stamp at minute `base_minute + i`
    seconds — deterministic order across runs. `outcomes` (when set)
    overrides per-call `success`; default = all True. `cid_offset`
    shifts the call-id space so concurrent stages within one fixture
    don't collide on cid (the run-scope jsonl shares one global cid
    namespace; the builder de-duplicates by cid).
    """
    if outcomes is None:
        outcomes = [True] * count
    lines: list[str] = []
    for i in range(count):
        cid = f"{cid_offset + i + 1:04d}"
        start_iso = f"2026-05-17T09:{base_minute:02d}:{i:02d}.000Z"
        end_iso = f"2026-05-17T09:{base_minute:02d}:{i:02d}.500Z"
        lines.append(json.dumps({
            "event": "begin", "call_id": cid, "stage": stage,
            "model": model, "mode": "tee",
            "started_at_iso": start_iso,
        }))
        lines.append(json.dumps({
            "event": "end", "call_id": cid, "stage": stage,
            "success": outcomes[i],
            "error": None if outcomes[i] else "rate_limited",
            "duration_ms": 500.0 + i, "prompt_tokens": 100,
            "completion_tokens": 50, "ttft_ms": 150.0,
            "model": model, "mode": "tee",
            "ts": end_iso, "ended_at_iso": end_iso,
        }))
    existing = path.read_text() if path.is_file() else ""
    path.write_text(existing + "\n".join(lines) + "\n")


def _populate_run_dir(
    run_dir: Path, *, perma_id: str, stages_with_calls: dict[str, int],
    high_volume_count: int | None = None,
) -> None:
    """Build a fake run dir with the canonical layout used by
    `build_run_marker` / `_emit_shareable_run_marker`: stages/<NN-name>/
    + phase_1_marker.json sentinels + a run-level llm-calls.jsonl.

    `stages_with_calls` maps the stage's run-dir leaf (e.g.
    "01-extraction") to the count of LLM calls to fake for it.
    """
    (run_dir / "stages" / "00-ingestion").mkdir(parents=True)
    (run_dir / "config.json").write_text(
        json.dumps({"short_id": perma_id})
    )
    (run_dir / "stages" / "00-ingestion" / "phase_1_marker.json").write_text(
        "{}"
    )
    calls_path = run_dir / "llm-calls.jsonl"
    cid_offset = 0
    for idx, (leaf, n) in enumerate(stages_with_calls.items(), start=1):
        sd = run_dir / "stages" / leaf
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "phase_1_marker.json").write_text("{}")
        # The runner emits `set_stage(...)` with the alias name (e.g.
        # "extract", not "extraction"); the marker mapper resolves both.
        alias = leaf.split("-", 1)[1] if "-" in leaf else leaf
        if alias == "extraction":
            alias = "extract"
        _stage_calls_jsonl(
            calls_path, alias, "kimi-k2-6", n,
            base_minute=idx, cid_offset=cid_offset,
        )
        cid_offset += n


def test_stage_stat_carries_identity_outcome_timing_work_perf_dists(tmp_path):
    """A fully-populated stage entry materializes every new enrichment
    axis (identity, outcome, timing, work, perf, distributions) and
    passes the content-free guard end-to-end."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-aaaa"
    _populate_run_dir(
        run_dir, perma_id="aaaa",
        stages_with_calls={"04-patterns": 4},  # low-volume → full enum
    )
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    s._assert_content_free(marker)
    stages = {st.stage: st for st in marker.stages}
    pat = stages[s.StageToken.patterns]
    # Identity
    assert pat.stage is s.StageToken.patterns
    assert pat.present is True
    assert pat.completed is True
    # Outcome
    assert pat.success is True
    # Timing — derived from the per-call ISO-Z stamps in llm-calls.jsonl
    assert pat.started_at is not None and pat.started_at.endswith("Z")
    assert pat.ended_at is not None and pat.ended_at.endswith("Z")
    assert pat.wall_ms is not None and pat.wall_ms > 0
    # Work rollup
    assert pat.call_count == 4
    assert pat.success_count == 4
    assert pat.failure_count == 0
    assert pat.cache_hit_count == 0
    assert pat.prompt_tokens_sum == 4 * 100
    assert pat.completion_tokens_sum == 4 * 50
    # Performance distribution
    assert pat.duration_ms_p50 is not None
    assert pat.duration_ms_p95 is not None
    assert pat.duration_ms_mean is not None
    assert pat.ttft_ms_p50 == 150.0
    # Categorical: one row for outcome=ok; retry-class empty (no retries)
    assert pat.outcome_dist == (s.OutcomeCount(s.Outcome.success, 4),)
    assert pat.retry_class_dist == ()


def test_high_volume_stage_samples_successes_keeps_all_failures(tmp_path):
    """extract/entities/embeddings get cap-3 deterministic
    first/median/last successes + ALL failures; successful_calls_sampled = True."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-bbbb"
    (run_dir / "stages" / "01-extraction").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "bbbb"}))
    (run_dir / "stages" / "01-extraction" / "phase_1_marker.json").write_text("{}")
    # 10 successes + 2 failures — high-volume stage.
    outcomes = [True] * 10 + [False, False]
    _stage_calls_jsonl(
        run_dir / "llm-calls.jsonl", "extract", "kimi-k2-6",
        len(outcomes), outcomes=outcomes, base_minute=1,
    )

    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    extract = next(
        st for st in marker.stages
        if st.stage is s.StageToken.extraction
    )
    assert extract.call_count == 12
    assert extract.success_count == 10
    assert extract.failure_count == 2
    assert extract.successful_calls_sampled is True
    # 3 successes (capped) + 2 failures = 5 calls emitted.
    assert len(extract.calls) == 5
    # Failures preserved as-is.
    failures = [c for c in extract.calls if c.outcome is not s.Outcome.success]
    assert len(failures) == 2
    # Successes are the deterministic first/median/last by started_at.
    # The 10 successes started at seconds :00 .. :09; first=:00,
    # median (10//2 = 5) = :05, last = :09.
    success_starts = sorted(
        c.started_at for c in extract.calls if c.outcome is s.Outcome.success
    )
    expected = [
        "2026-05-17T09:01:00.000Z",
        "2026-05-17T09:01:05.000Z",
        "2026-05-17T09:01:09.000Z",
    ]
    assert success_starts == expected


def test_low_volume_stage_keeps_every_call_no_sampling(tmp_path):
    """patterns/insights/actions/dedupe/vision stay fully enumerated;
    successful_calls_sampled = False even with many calls."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-cccc"
    (run_dir / "stages" / "03-patterns").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "cccc"}))
    (run_dir / "stages" / "03-patterns" / "phase_1_marker.json").write_text("{}")
    _stage_calls_jsonl(
        run_dir / "llm-calls.jsonl", "patterns", "kimi-k2-6",
        13, base_minute=1,  # 13 topics, the production default
    )
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    pat = next(
        st for st in marker.stages if st.stage is s.StageToken.patterns
    )
    assert pat.successful_calls_sampled is False
    assert pat.call_count == 13
    assert len(pat.calls) == 13


def test_run_marker_llm_calls_keeps_rollup_drops_calls_tuple(tmp_path):
    """Run-scope LlmCallsBlock keeps the four-int aggregate but the
    flat per-call tuple is gone (it lives under each StageStat now)."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-dddd"
    _populate_run_dir(
        run_dir, perma_id="dddd",
        stages_with_calls={"01-extraction": 5, "03-patterns": 3},
    )
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    assert marker.llm_calls is not None
    # The headline rollup survives: 5 + 3 calls, paired token sums.
    assert marker.llm_calls.call_count == 8
    assert marker.llm_calls.total_prompt_tokens == 8 * 100
    assert marker.llm_calls.total_completion_tokens == 8 * 50
    # The flat per-call tuple is empty — per-call detail moved under
    # the per-stage StageStat.calls.
    assert marker.llm_calls.calls == ()
    # And it does appear under stages.
    extract = next(
        st for st in marker.stages if st.stage is s.StageToken.extraction
    )
    pat = next(
        st for st in marker.stages if st.stage is s.StageToken.patterns
    )
    assert extract.call_count == 5 and len(extract.calls) <= 5
    assert pat.call_count == 3 and len(pat.calls) == 3


def test_run_marker_yaml_is_substantially_smaller_than_flat(
    tmp_path, monkeypatch
):
    """The compression-win acceptance from the ticket: a ~50-call run
    no longer dumps every call inline. Specifically: the high-volume
    stages compress to <=3+failures while the low-volume stages stay
    full, and the YAML's call-block line count drops sharply."""
    import yaml

    monkeypatch.setenv("HOME", str(tmp_path))
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-eeee"
    _populate_run_dir(
        run_dir, perma_id="eeee",
        stages_with_calls={
            "01-extraction": 25,   # high-volume → sampled
            "02-entities":   15,   # high-volume → sampled
            "03-patterns":   13,   # low-volume → full
            "04-insights":    1,
            "05-actions":     1,
            "06-embeddings":  8,   # high-volume → sampled
        },
    )
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    s.emit(s.Stream.RUN, "eeee", marker)
    out = next(
        (tmp_path / ".basevault" / "shareable" / "run-diagnostics")
        .glob("*-eeee-anonymized.yaml")
    )
    docs = list(yaml.safe_load_all(out.read_text()))
    run = docs[1]
    by_stage = {st["stage"]: st for st in run["stages"]}
    # High-volume stages compress to <=3 calls (no failures here).
    assert by_stage["extraction"]["successful_calls_sampled"] is True
    assert by_stage["entities"]["successful_calls_sampled"] is True
    assert by_stage["embeddings"]["successful_calls_sampled"] is True
    assert len(by_stage["extraction"]["calls"]) == 3
    assert len(by_stage["entities"]["calls"]) == 3
    assert len(by_stage["embeddings"]["calls"]) == 3
    # Low-volume stages stay full.
    assert by_stage["patterns"]["successful_calls_sampled"] is False
    assert len(by_stage["patterns"]["calls"]) == 13
    # And there's no flat top-level enumeration left at the run scope.
    # (`llm_calls.calls` is empty so `_to_jsonable` drops the field.)
    assert "calls" not in run.get("llm_calls", {})
    # The total per-stage emitted calls ≈ 23, vs 63 calls fired — the
    # compression win the ticket asks for. `calls` is omitted from the
    # YAML for stages that fired none (e.g. ingestion), so `.get(...)`.
    total_emitted = sum(len(st.get("calls", [])) for st in run["stages"])
    total_fired = sum(st.get("call_count", 0) for st in run["stages"])
    assert total_fired == 63
    assert total_emitted == 3 + 3 + 13 + 1 + 1 + 3  # = 24
    assert total_emitted < total_fired // 2  # at least 2× smaller


def test_stage_started_ended_at_match_min_max_of_call_isoz(tmp_path):
    """Per-stage timing is derived from the per-call ISO-Z stamps in
    the run's llm-calls.jsonl (NOT plumbed from ProgressTracker — the
    cancel-settle CLI runs without a live tracker, and this design
    keeps both paths uniform). started_at = min(call.started_at);
    ended_at = max(call.ended_at) — the latest end, since a parallel
    fan-out batch may finish out-of-order vs. start."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-ffff"
    (run_dir / "stages" / "01-extraction").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "ffff"}))
    (run_dir / "stages" / "01-extraction" / "phase_1_marker.json").write_text("{}")
    _stage_calls_jsonl(
        run_dir / "llm-calls.jsonl", "extract", "kimi-k2-6", 5,
        base_minute=2,
    )
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    extract = next(
        st for st in marker.stages if st.stage is s.StageToken.extraction
    )
    # Calls start at :02:00, :02:01, :02:02, :02:03, :02:04 and have
    # durations 500/501/502/503/504 ms (helper's ``500 + i`` formula).
    # ended_at is derived as started_at + duration_ms; the LATEST end
    # is the last call's start (:02:04.000) + 504ms = :02:04.504.
    assert extract.started_at == "2026-05-17T09:02:00.000Z"
    assert extract.ended_at == "2026-05-17T09:02:04.504Z"
    # ~4504ms wall-clock (last start - first start + last duration).
    assert extract.wall_ms is not None
    assert abs(extract.wall_ms - 4504.0) < 1.0


def test_pick_first_median_last_picks_are_deterministic():
    """The sampler picks indices [0], [n//2], [n-1] — same picks every
    invocation. Reproducibility > randomness for diagnostics."""
    items = [f"x{i}" for i in range(11)]
    picks = sm._pick_first_median_last(items, 3)
    assert picks == [items[0], items[5], items[10]]
    # Cap >= n: returns the whole list verbatim.
    assert sm._pick_first_median_last(items, 11) == items
    assert sm._pick_first_median_last(items[:2], 3) == items[:2]


def test_stage_success_flag_false_on_any_failure(tmp_path):
    """success is the derived AND-of-everything-clean rollup: a stage
    with even one failure is not a successful stage."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-gggg"
    (run_dir / "stages" / "01-extraction").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "gggg"}))
    (run_dir / "stages" / "01-extraction" / "phase_1_marker.json").write_text("{}")
    _stage_calls_jsonl(
        run_dir / "llm-calls.jsonl", "extract", "kimi-k2-6", 5,
        outcomes=[True, True, False, True, True], base_minute=1,
    )
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    extract = next(
        st for st in marker.stages if st.stage is s.StageToken.extraction
    )
    assert extract.failure_count == 1
    assert extract.success is False
    # Outcome dist surfaces the runner's canonical label. A generic
    # error string with no specific failure signal classifies as
    # ``failed (other)`` per ``_failure_class_for_label``.
    by_outcome = {row.outcome: row.count for row in extract.outcome_dist}
    assert by_outcome[s.Outcome.success] == 4
    assert by_outcome[s.Outcome.failed_other] == 1


def test_outcomes_mirror_runner_classifier_verbatim(tmp_path):
    """The marker's ``Outcome`` enum is the runner's ``OUTCOME_*``
    vocabulary verbatim, and classification is delegated to
    ``runner._classify_outcome`` — so the diagnostic shows the SAME
    label the run-details UI shows for the SAME call, no parallel
    classifier to drift. This pins both the 1:1 mapping AND the
    delegation.

    Covers the historically-buggy cases: dvqm/0107 (success-empty
    via ``output={\"facts\": 0}`` from the counts event), kqzc/0086
    (success-empty via the ``_SuccessEmpty`` exception-class name
    in the error dict), aborted (begin without end), skipped (user-
    skip marker file)."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-mmmm"
    (run_dir / "stages" / "01-extraction").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "mmmm"}))
    (run_dir / "stages" / "01-extraction" / "phase_1_marker.json").write_text("{}")
    # User-skipped: marker file under skipped_calls/.
    (run_dir / "skipped_calls").mkdir()
    (run_dir / "skipped_calls" / "0024").write_text("")
    lines: list[str] = []
    # 0001: plain success
    lines.append(json.dumps({
        "event": "begin", "call_id": "0001", "stage": "extract",
        "model": "kimi-k2-6", "mode": "tee",
        "started_at_iso": "2026-05-17T09:01:01.000Z",
    }))
    lines.append(json.dumps({
        "event": "end", "call_id": "0001", "stage": "extract",
        "success": True, "error": None,
        "duration_ms": 500, "prompt_tokens": 100,
        "completion_tokens": 50, "content_tokens": 50,
        "finish_reason": "stop", "model": "kimi-k2-6", "mode": "tee",
        "ts": "2026-05-17T09:01:01.500Z",
    }))
    lines.append(json.dumps({
        "event": "counts", "call_id": "0001",
        "input": {"facts": 100}, "output": {"facts": 5},
    }))
    # 0002: success-empty via the output counts (parseable [] result).
    lines.append(json.dumps({
        "event": "begin", "call_id": "0002", "stage": "extract",
        "model": "kimi-k2-6", "mode": "tee",
        "started_at_iso": "2026-05-17T09:01:02.000Z",
    }))
    lines.append(json.dumps({
        "event": "end", "call_id": "0002", "stage": "extract",
        "success": True, "error": None,
        "duration_ms": 500, "prompt_tokens": 100,
        "completion_tokens": 5, "content_tokens": 5,
        "finish_reason": "stop", "model": "kimi-k2-6", "mode": "tee",
        "ts": "2026-05-17T09:01:02.500Z",
    }))
    lines.append(json.dumps({
        "event": "counts", "call_id": "0002",
        "input": {"facts": 100}, "output": {"facts": 0},
    }))
    # 0024: SKIPPED — user-skip marker exists on disk for this id.
    lines.append(json.dumps({
        "event": "begin", "call_id": "0024", "stage": "extract",
        "model": "kimi-k2-6", "mode": "tee",
        "started_at_iso": "2026-05-17T09:01:24.000Z",
    }))
    # 0047: ABORTED — begin without end (in-flight at wind-down).
    lines.append(json.dumps({
        "event": "begin", "call_id": "0047", "stage": "extract",
        "model": "kimi-k2-6", "mode": "tee",
        "started_at_iso": "2026-05-17T09:01:47.000Z",
    }))
    (run_dir / "llm-calls.jsonl").write_text("\n".join(lines) + "\n")
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    s._assert_content_free(marker)
    extract = next(
        st for st in marker.stages if st.stage is s.StageToken.extraction
    )
    by_outcome = {row.outcome: row.count for row in extract.outcome_dist}
    # Each call gets the same label the runner's UI would show.
    assert by_outcome[s.Outcome.success] == 1
    assert by_outcome[s.Outcome.success_empty] == 1
    assert by_outcome[s.Outcome.skipped] == 1
    assert by_outcome[s.Outcome.aborted] == 1
    # Failure-count includes aborted (in-flight at wind-down) per the
    # runner's accounting: a begin-without-end is a real call that
    # didn't complete. Skipped is the explicit human signal.
    assert extract.call_count == 4
    assert extract.success_count == 2  # success + success_empty


def test_reasoning_off_success_preserved_wholesale_in_high_volume(tmp_path):
    """Director ask: reasoning-off-recovery successes are high-signal
    in the retry chain (the UI's run-details surface dumps the
    payload). High-volume sampler preserves them alongside failures
    — never sampled out — so the \"how did this stage recover?\"
    question keeps its answer."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-llll"
    (run_dir / "stages" / "01-extraction").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "llll"}))
    (run_dir / "stages" / "01-extraction" / "phase_1_marker.json").write_text("{}")
    # 10 plain successes + 1 reasoning-off success in the middle. The
    # plain successes would normally get sampled to 3 (first/median/last);
    # the reasoning-off success must NOT be sampled out.
    lines: list[str] = []
    for i in range(10):
        cid = f"{i+1:04d}"
        lines.append(json.dumps({
            "event": "begin", "call_id": cid, "stage": "extract",
            "model": "kimi-k2-6", "mode": "tee",
            "started_at_iso": f"2026-05-17T09:01:{i:02d}.000Z",
        }))
        lines.append(json.dumps({
            "event": "end", "call_id": cid, "stage": "extract",
            "success": True, "error": None,
            "duration_ms": 500, "prompt_tokens": 100,
            "completion_tokens": 50, "content_tokens": 50,
            "finish_reason": "stop", "model": "kimi-k2-6", "mode": "tee",
            "ts": f"2026-05-17T09:01:{i:02d}.500Z",
        }))
    # The reasoning-off-recovery success. Category embeds the retry
    # chain so the marker parses ``RetryTransform.reasoning_off``.
    lines.append(json.dumps({
        "event": "begin", "call_id": "0011", "stage": "extract",
        "category": "topic_prefix/reasoning-off - retry/other",
        "retry_of_call_id": "0007",
        "model": "kimi-k2-6", "mode": "tee",
        "started_at_iso": "2026-05-17T09:01:11.000Z",
    }))
    lines.append(json.dumps({
        "event": "end", "call_id": "0011", "stage": "extract",
        "success": True, "error": None,
        "duration_ms": 600, "prompt_tokens": 100,
        "completion_tokens": 80, "content_tokens": 80,
        "finish_reason": "stop", "model": "kimi-k2-6", "mode": "tee",
        "ts": "2026-05-17T09:01:11.500Z",
    }))
    (run_dir / "llm-calls.jsonl").write_text("\n".join(lines) + "\n")
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    extract = next(
        st for st in marker.stages if st.stage is s.StageToken.extraction
    )
    assert extract.successful_calls_sampled is True
    # 3 sampled successes + the reasoning-off-recovery call preserved.
    assert len(extract.calls) == 4
    reasoning_off_calls = [
        c for c in extract.calls
        if s.RetryTransform.reasoning_off in c.retry_transforms
    ]
    assert len(reasoning_off_calls) == 1


def test_orphan_calls_land_in_fallback_stage_entry_not_dropped(tmp_path):
    """Codex P1 regression: some stage tokens fire calls without a
    matching ``stages/<NN-name>/`` dir leaf (vision dispatches inside
    ingestion; entities_dedupe inside entities; anything that maps to
    ``other``). Without a carve-out, those calls would silently vanish
    from the diagnostic now that ``RunMarker.llm_calls.calls`` is empty
    by construction. The carve-out: each orphan stage token gets its
    own ``StageStat`` with ``present=False``, so the run-level
    aggregate and the per-stage detail stay reconciled."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-iiii"
    # Only the extraction dir is created — but the jsonl carries calls
    # for extract (matches a dir), vision (no dir), and
    # entities_dedupe (no dir).
    (run_dir / "stages" / "01-extraction").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "iiii"}))
    (run_dir / "stages" / "01-extraction" / "phase_1_marker.json").write_text("{}")
    calls_path = run_dir / "llm-calls.jsonl"
    _stage_calls_jsonl(
        calls_path, "extract", "kimi-k2-6", 3,
        base_minute=1, cid_offset=0,
    )
    _stage_calls_jsonl(
        calls_path, "vision", "kimi-k2-6", 2,
        base_minute=2, cid_offset=3,
    )
    _stage_calls_jsonl(
        calls_path, "entities_dedupe", "kimi-k2-6", 2,
        base_minute=3, cid_offset=5,
    )

    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    s._assert_content_free(marker)
    by_stage = {st.stage: st for st in marker.stages}

    # Real-dir entry: present=True.
    assert s.StageToken.extraction in by_stage
    assert by_stage[s.StageToken.extraction].present is True
    assert by_stage[s.StageToken.extraction].call_count == 3

    # Orphan stages get entries — ``present=True`` because they
    # actually ran (fired calls). ``completed=False`` because there's
    # no phase_1_marker.json for them (they don't have a dir leaf).
    assert s.StageToken.vision in by_stage
    assert by_stage[s.StageToken.vision].present is True
    assert by_stage[s.StageToken.vision].completed is False
    assert by_stage[s.StageToken.vision].call_count == 2

    assert s.StageToken.entities_dedupe in by_stage
    assert by_stage[s.StageToken.entities_dedupe].present is True
    assert by_stage[s.StageToken.entities_dedupe].completed is False
    assert by_stage[s.StageToken.entities_dedupe].call_count == 2

    # Reconciliation: every call in the run-level rollup has a home
    # under some StageStat — none silently dropped.
    assert marker.llm_calls is not None
    assert marker.llm_calls.call_count == 7
    total_under_stages = sum(st.call_count for st in marker.stages)
    assert total_under_stages == marker.llm_calls.call_count


def test_unknown_stage_token_lands_in_other_bucket_not_dropped(tmp_path):
    """A stage label the closed enum doesn't know about (a drifted /
    misfired tag) maps to ``StageToken.other`` and still gets a
    fallback entry — defensive against runner/marker schema drift."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-jjjj"
    (run_dir / "stages" / "00-ingestion").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "jjjj"}))
    (run_dir / "stages" / "00-ingestion" / "phase_1_marker.json").write_text("{}")
    _stage_calls_jsonl(
        run_dir / "llm-calls.jsonl", "newfangled-stage-2027",
        "kimi-k2-6", 4, base_minute=1,
    )
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    by_stage = {st.stage: st for st in marker.stages}
    assert s.StageToken.other in by_stage
    # ``present=True`` because the stage fired calls (even though no
    # dir leaf exists); ``completed=False`` (no marker).
    assert by_stage[s.StageToken.other].present is True
    assert by_stage[s.StageToken.other].completed is False
    assert by_stage[s.StageToken.other].call_count == 4
    # Run-level rollup ↔ per-stage detail still reconciles.
    assert marker.llm_calls.call_count == 4


def test_call_id_preserves_zero_padded_4digit_form(tmp_path):
    """Director ask: show the call ids in the runner's standard
    zero-padded 4-digit form (``\"0041\"``), not as a bare int (``41``).
    This matches the join key in ``llm-calls.jsonl`` /
    ``llm-payloads.jsonl`` so a reader can grep the same id across all
    three streams. The content-free guard accepts the ``^\\d{4}$``
    form alongside ISO-Z + closed-enum strings."""
    calls_path = tmp_path / "llm-calls.jsonl"
    calls_path.write_text(
        json.dumps({
            "event": "begin", "call_id": "0041", "stage": "extract",
            "model": "kimi-k2-6", "mode": "tee",
            "started_at_iso": "2026-05-17T09:00:00.000Z",
            "retry_of_call_id": "0007",
        }) + "\n" +
        json.dumps({
            "event": "end", "call_id": "0041", "stage": "extract",
            "success": True, "duration_ms": 500, "prompt_tokens": 100,
            "completion_tokens": 50, "content_tokens": 50,
            "model": "kimi-k2-6", "mode": "tee",
            "ts": "2026-05-17T09:00:01Z",
        }) + "\n"
    )
    block = sm.build_llm_calls_block(calls_path, since_call_id=0)
    # Re-validation through the content-free guard (4-digit form is
    # registered as an allowed leaf next to ISO-Z + enum strings).
    s._assert_content_free(block)
    c = block.calls[0]
    assert c.call_id == "0041"
    assert c.retry_of_call_id == "0007"
    # The serialized YAML carries the padded form verbatim, NOT a
    # truncated int.
    blob = json.dumps(s._to_jsonable(block))
    assert '"call_id": "0041"' in blob
    assert '"retry_of_call_id": "0007"' in blob


def test_stages_emit_in_canonical_pipeline_order(tmp_path):
    """Director ask: dedupe should appear between entities and patterns,
    not at the end. The combined real-dir + orphan list is sorted by
    canonical pipeline order so a reader sees stages top-to-bottom in
    their natural execution sequence — ``entities_dedupe`` (orphan; no
    dir leaf since it fires inside the entities stage code) lands
    between ``entities`` and ``patterns``, not appended last by
    enum-value sort."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-mmmm"
    # Real dirs for entities + patterns + insights; entities_dedupe is
    # an orphan (no dir leaf), and should land between entities and
    # patterns in the emitted order.
    for leaf in ("02-entities", "03-patterns", "04-insights"):
        (run_dir / "stages" / leaf).mkdir(parents=True)
        (run_dir / "stages" / leaf / "phase_1_marker.json").write_text("{}")
    (run_dir / "config.json").write_text(json.dumps({"short_id": "mmmm"}))
    cid_off = 0
    for alias, n in (("entities", 3),
                     ("entities_dedupe", 1),
                     ("patterns", 2),
                     ("insights", 1)):
        _stage_calls_jsonl(
            run_dir / "llm-calls.jsonl", alias, "kimi-k2-6", n,
            base_minute=cid_off + 1, cid_offset=cid_off,
        )
        cid_off += n
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    s._assert_content_free(marker)
    stages = [st.stage for st in marker.stages]
    # entities_dedupe sits BETWEEN entities and patterns, not at end.
    i_ent = stages.index(s.StageToken.entities)
    i_ded = stages.index(s.StageToken.entities_dedupe)
    i_pat = stages.index(s.StageToken.patterns)
    i_ins = stages.index(s.StageToken.insights)
    assert i_ent < i_ded < i_pat < i_ins, stages
    # stage_index reassigned post-sort so the printed ordinal matches
    # the emit position (a reader's mental model).
    for printed_idx, st in enumerate(marker.stages):
        assert st.stage_index == printed_idx


def test_scaffolding_only_stage_is_dropped_from_marker(tmp_path):
    """A stage dir that the runner created but never produced work
    for (no LLM calls AND no ``phase_1_marker.json`` completion
    sentinel) is NOT a real stage entry — the marker drops it so a
    reader doesn't see ghost rows. ``actions`` with zero inputs was
    the regression that surfaced this: the dir existed (runner
    bookkeeping) but the stage didn't run."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-hhhh"
    # Scaffolding dirs only — no marker, no calls.
    (run_dir / "stages" / "05-actions").mkdir(parents=True)
    (run_dir / "stages" / "01-extraction").mkdir(parents=True)
    # extraction has a completion sentinel so it IS present.
    (run_dir / "stages" / "01-extraction" / "phase_1_marker.json").write_text("{}")
    (run_dir / "config.json").write_text(json.dumps({"short_id": "hhhh"}))
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    s._assert_content_free(marker)
    stage_tokens = {st.stage for st in marker.stages}
    assert s.StageToken.extraction in stage_tokens
    assert s.StageToken.actions not in stage_tokens


# ── Codex review regressions ───────────────────────────────────────


def test_call_id_4digit_bypass_restricted_to_call_id_fields(tmp_path):
    """Codex P1 (9ab6062): the 4-digit string allowance in the
    content-free guard must apply ONLY to the ``call_id`` /
    ``retry_of_call_id`` fields. A 4-digit free string on any other
    str field (e.g. ``ChatMarker.ts``) must still fail — defense-in-
    depth on top of the typed schema, so a future caller can't smuggle
    a 4-digit numeric through a non-call-id str leaf."""
    bad = dataclasses.replace(_clean_chat_marker(), ts="1234")
    with pytest.raises(s.ContentFreeViolation):
        s._assert_content_free(bad)
    # And the same value on a real call_id field still passes (the
    # narrow allowance is intact).
    good_call = _llm_call(call_id="1234")
    s._assert_content_free(good_call)


def test_derive_ended_at_iso_normalizes_ms_rollover():
    """Codex P2 (9ab6062): ``derive_ended_at_iso`` must NOT emit
    ``...:04.1000Z`` when the rounded millisecond hits 1000. The
    rollover has to carry into the seconds component so the rendered
    timestamp stays a valid ISO-Z (millisecond field 000-999)."""
    from engine.common.dates import derive_ended_at_iso
    # start at :00.000, duration 999.6ms → end at :00.9996s.
    # round(0.9996 * 1000) == 1000 — would emit ``:00.1000Z`` under
    # the old split-then-round formulation. Correct behavior: roll
    # over into the next second → ``:01.000Z``.
    out = derive_ended_at_iso("2026-05-17T09:00:00.000Z", 999.6)
    assert out == "2026-05-17T09:00:01.000Z", out
    # Sanity: no rollover when the round stays under 1000.
    out2 = derive_ended_at_iso("2026-05-17T09:00:00.000Z", 504.0)
    assert out2 == "2026-05-17T09:00:00.504Z", out2


def test_embed_outcome_neutral_when_no_records_embedded(tmp_path):
    """Codex P2 (9ab6062): a run that produced no embedded records
    (canceled / unfinished / pre-embeddings-stage abort) must NOT be
    classified as ``failed_other`` — that misclassifies no-signal as a
    real failure. ``embed_outcome`` is now ``None`` (omitted from the
    YAML) for the no-signal case."""
    run_dir = tmp_path / ".basevault" / "logs" / "2026-05-17T09-00-00Z-pppp"
    (run_dir / "stages" / "00-ingestion").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"short_id": "pppp"}))
    # No store, no records → by_kind empty.
    marker = sm.build_run_marker(store=None, run_dir=run_dir)
    s._assert_content_free(marker)
    assert marker.embedding.embed_outcome is None


# ── #781: multi-hop chat-diagnostic marker (supersedes the #818 flat ──
#    retrieval block; the post-#799 ReAct loop fires up to MAX_HOPS+1
#    LLM calls per turn, each potentially dispatching a search array of
#    1..MAX_LOOKUPS lookups. Marker now carries the per-hop trace + the
#    per-lookup detail).


class _FakeStore:
    """Tiny shim with the count()/count_by_kind() surface the builder
    reads. Used in lieu of opening a real sqlite-vec store in unit
    tests."""
    def __init__(self, total: int, by_kind: dict):
        self._total = total
        self._by_kind = by_kind

    def count(self) -> int:
        return self._total

    def count_by_kind(self) -> dict:
        return self._by_kind


def _empty_llm_block() -> s.LlmCallsBlock:
    return s.LlmCallsBlock(0, 0, 0, 0.0, ())


def _hop_diag(
    *,
    hop_outcome: str = "tool_call",
    streamed: bool = False,
    lookups_remaining: int | None = 3,
    prev_attempts: int = 0,
    store_open_ms: float | None = 2.0,
    dispatch_ms: float | None = 40.0,
    union_after: int | None = 5,
    per_lookup: list[dict] | None = None,
) -> dict:
    """Synthesize a per-hop diag dict the marker builder consumes."""
    d: dict = {
        "hop_outcome": hop_outcome,
        "streamed_to_user": streamed,
        "lookups_remaining_in_budget": lookups_remaining,
        "previous_attempts_count": prev_attempts,
        "store_open_latency_ms": store_open_ms,
        "dispatch_latency_ms": dispatch_ms,
        "union_size_after": union_after,
    }
    if per_lookup is not None:
        d["per_lookup"] = per_lookup
    return d


def _lookup_diag(
    *,
    entry_types: list[str] = None,
    k: int = 15,
    neighbor_count: int = 0,
    neighbor_kinds: list[str] = None,
    exact_match_count: int = 0,
    query: bool = True,
    degenerate: bool = False,
    tied: bool = False,
    k_returned: int = 10,
    junk_dropped: int = 0,
    kind_counts: dict | None = None,
    score_samples: tuple | None = (0.2, 0.4, 0.6),
) -> dict:
    """Synthesize a per-lookup diag dict matching what
    ``_dispatch_search`` emits in its ``per_lookup`` list."""
    d: dict = {
        "entry_types": entry_types or ["fact"],
        "k_requested": k,
        "has_neighbor_count": neighbor_count,
        "has_neighbor_kinds": neighbor_kinds or [],
        "exact_match_count": exact_match_count,
        "query_present": query,
        "degenerate": degenerate,
        "tied": tied,
        "k_returned": k_returned,
        "junk_dropped": junk_dropped,
        "kind_counts": kind_counts if kind_counts is not None else {"fact": k_returned},
        "score_samples": score_samples,
    }
    if query:
        d.update({
            "query_char_len": 80, "embed_dim": 768, "embed_norm": 1.0,
            "embed_all_zero": False, "embed_non_finite": False,
        })
    return d


def _build(**over) -> s.ChatMarker:
    """Test helper: build a chat marker with sane defaults and any
    test-specific overrides."""
    base = dict(
        turn_index=1, session_id="0123456789abcdef",
        lookup_fired=False, hops_diag=[],
        llm_calls=_empty_llm_block(),
    )
    base.update(over)
    return sm.build_chat_marker(**base)


# ── A: turn-level fields (preserved from #818 work) ──────────────────


def test_skipped_reason_none_for_conversational_turn():
    m = _build(lookup_fired=False)
    assert m.retrieve_skipped_reason is s.RetrieveSkippedReason.none


def test_skipped_reason_no_bound_run_when_session_unbound():
    # Lookup fired but sidecar had no _SESSION_STORE_PATH.
    m = _build(lookup_fired=True, store_bound=False)
    assert m.retrieve_skipped_reason is s.RetrieveSkippedReason.no_bound_run


def test_legacy_run_with_no_perma_id_is_not_falsely_no_bound_run():
    # Bindable legacy run with no resolvable 4-letter short_id: store
    # IS bound (store_bound=True) but perma-id unresolvable
    # (bound_run=None). Retrieval ran cleanly via a real hop. Codex P2
    # (#818): skip-reason must NOT say no_bound_run here.
    m = _build(
        lookup_fired=True, store_bound=True, bound_run=None,
        hops_diag=[_hop_diag(
            per_lookup=[_lookup_diag()],
        )],
    )
    assert m.retrieve_skipped_reason is s.RetrieveSkippedReason.none
    assert m.bound_run is None  # informational, null fine


def test_bound_run_perma_id_passes_content_free_guard():
    m = _build(lookup_fired=True, store_bound=True, bound_run="ab3k")
    s._assert_content_free(m)
    assert m.bound_run == "ab3k"


def test_bound_run_free_string_crashes_runtime_guard():
    # The "Mom's notes" smuggling leak the perma-id guard exists to stop.
    bad = dataclasses.replace(_build(), bound_run="Mom's notes")
    with pytest.raises(s.ContentFreeViolation):
        s._assert_content_free(bad)


def test_perma_id_shape_only_allowed_on_bound_run_field():
    # ``ab3k`` is a valid perma-id but smuggled into ``.ts`` is still
    # a free string — narrowing the bypass keeps the guard tight.
    bad = dataclasses.replace(_build(), ts="ab3k")
    with pytest.raises(s.ContentFreeViolation):
        s._assert_content_free(bad)


def test_session_id_16_hex_passes_guard():
    m = _build(session_id="0123456789abcdef")
    s._assert_content_free(m)
    assert m.session_id == "0123456789abcdef"


def test_session_id_free_string_crashes_guard():
    # Any non-16-hex on .session_id must crash — same path-narrowed
    # pattern as bound_run and call_id.
    bad = dataclasses.replace(_build(), session_id="Mom's notes")
    with pytest.raises(s.ContentFreeViolation):
        s._assert_content_free(bad)


def test_session_id_shape_only_allowed_on_session_id_field():
    # A valid 16-hex value smuggled into .ts is still rejected — the
    # bypass narrows by field path, mirroring perma-id and call-id.
    bad = dataclasses.replace(_build(), ts="0123456789abcdef")
    with pytest.raises(s.ContentFreeViolation):
        s._assert_content_free(bad)


def test_build_store_stats_maps_kinds_and_strips_unknown():
    stats = sm.build_store_stats(
        _FakeStore(686, {"chunk": 466, "fact": 129, "bogus_kind": 99}),
    )
    s._assert_content_free(stats)
    assert stats.total_records == 686
    kinds = {kc.kind for kc in stats.records_by_kind}
    assert s.RecordKind.chunk in kinds and s.RecordKind.fact in kinds
    assert all(isinstance(kc.kind, s.RecordKind)
               for kc in stats.records_by_kind)


def test_build_store_stats_raises_on_count_failure_no_silent_zero():
    # Codex P2 carried forward: build_store_stats must NOT mask
    # count() exceptions into total_records=0.
    class _Broken:
        def count(self):
            raise RuntimeError("vectors.db is corrupt")

        def count_by_kind(self):
            return {}

    with pytest.raises(RuntimeError):
        sm.build_store_stats(_Broken())


# ── B: hops_diag → HopMarker → ChatMarker.hops ───────────────────────


def test_hops_join_call_id_from_llm_calls_block_by_order():
    # Each HopMarker.call_id is filled from llm_calls.calls[i] by
    # order. Synthesize 3 LlmCalls and 3 hop dicts; verify the join.
    calls = (
        _llm_call(call_id="0010"),
        _llm_call(call_id="0011"),
        _llm_call(call_id="0012"),
    )
    block = s.LlmCallsBlock(3, 0, 0, 0.0, calls)
    m = _build(
        lookup_fired=True, store_bound=True,
        llm_calls=block,
        hops_diag=[
            _hop_diag(per_lookup=[_lookup_diag()]),
            _hop_diag(per_lookup=[_lookup_diag()]),
            _hop_diag(hop_outcome="prose_answer", streamed=True,
                      store_open_ms=None, dispatch_ms=None,
                      union_after=None),
        ],
    )
    assert [h.call_id for h in m.hops] == ["0010", "0011", "0012"]


def test_hops_call_id_falls_back_when_llm_block_short():
    # Hop count > llm_calls.calls length → the missing call_ids fall
    # back to the "0000" sentinel rather than crashing.
    m = _build(
        lookup_fired=True, store_bound=True,
        llm_calls=_empty_llm_block(),
        hops_diag=[_hop_diag(per_lookup=[_lookup_diag()])],
    )
    assert m.hops[0].call_id == "0000"


def test_hop_outcomes_map_through_closed_enum():
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[
            _hop_diag(hop_outcome="tool_call",
                      per_lookup=[_lookup_diag()]),
            _hop_diag(hop_outcome="invalid_tool_call"),
            _hop_diag(hop_outcome="prose_answer", streamed=True,
                      store_open_ms=None, dispatch_ms=None,
                      union_after=None),
        ],
    )
    assert m.hops[0].hop_outcome is s.HopOutcome.tool_call
    assert m.hops[1].hop_outcome is s.HopOutcome.invalid_tool_call
    assert m.hops[2].hop_outcome is s.HopOutcome.prose_answer


def test_prose_hop_carries_no_lookups_block():
    # When hop_outcome != tool_call, the lookups tuple stays empty —
    # the dispatcher never fired for this hop.
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[
            _hop_diag(hop_outcome="prose_answer", streamed=True,
                      store_open_ms=None, dispatch_ms=None,
                      union_after=None,
                      per_lookup=[_lookup_diag()]),  # would be ignored
        ],
    )
    assert m.hops[0].lookups == ()


def test_invalid_tool_call_carries_no_lookups_block():
    # Same guarantee for invalid_tool_call: dispatch never ran, so
    # whatever the hop carries doesn't become per-lookup detail.
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[_hop_diag(hop_outcome="invalid_tool_call")],
    )
    assert m.hops[0].lookups == ()
    assert m.hops[0].hop_outcome is s.HopOutcome.invalid_tool_call


def test_hop_marker_carries_react_budget_and_attempts():
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[_hop_diag(
            lookups_remaining=3, prev_attempts=1,
            per_lookup=[_lookup_diag()],
        )],
    )
    assert m.hops[0].lookups_remaining_in_budget == 3
    assert m.hops[0].previous_attempts_count == 1


def test_hop_marker_lookups_remaining_null_on_decision_and_final():
    # Per chatbot_turn: decision and grounded_final pass None; only
    # grounded_decision exposes the budget.
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[_hop_diag(
            lookups_remaining=None,
            per_lookup=[_lookup_diag()],
        )],
    )
    assert m.hops[0].lookups_remaining_in_budget is None


def test_per_hop_latencies_round_trip():
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[_hop_diag(
            store_open_ms=3.14, dispatch_ms=47.2,
            per_lookup=[_lookup_diag()],
        )],
    )
    assert m.hops[0].store_open_latency_ms == 3.14
    assert m.hops[0].dispatch_latency_ms == 47.2


# ── C: per-lookup detail (the dispatch-side legibility win) ──────────


def test_lookup_shape_request_knobs_round_trip():
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[_hop_diag(per_lookup=[_lookup_diag(
            entry_types=["chunk", "fact"], k=20,
            neighbor_count=2, neighbor_kinds=["entity", "pattern"],
            exact_match_count=3, query=True,
        )])],
    )
    lk = m.hops[0].lookups[0]
    assert lk.entry_types == (s.RecordKind.chunk, s.RecordKind.fact)
    assert lk.k_requested == 20
    assert lk.has_neighbor_count == 2
    assert lk.has_neighbor_kinds == (s.RecordKind.entity, s.RecordKind.pattern)
    assert lk.exact_match_count == 3
    assert lk.query_present is True


def test_lookup_shape_filter_only_omits_embed_signal():
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[_hop_diag(per_lookup=[_lookup_diag(
            query=False, kind_counts={"fact": 5}, score_samples=None,
            k_returned=5,
        )])],
    )
    lk = m.hops[0].lookups[0]
    assert lk.query_present is False
    # The five embed signals all None when query_present=False — the
    # YAML omits the block entirely.
    assert lk.query_char_len is None
    assert lk.embed_dim is None
    assert lk.embed_norm is None
    assert lk.embed_all_zero is None
    assert lk.embed_non_finite is None
    # Filter-only lookups have no distances → score_samples is None.
    assert lk.score_samples is None


def test_lookup_shape_score_samples_round_trip():
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[_hop_diag(per_lookup=[_lookup_diag(
            score_samples=(0.21, 0.28, 0.35),
        )])],
    )
    ss = m.hops[0].lookups[0].score_samples
    assert isinstance(ss, s.ScoreSamples)
    assert ss.closest == 0.21
    assert ss.middle == 0.28
    assert ss.farthest == 0.35


def test_lookup_shape_kind_counts_filters_unknown_kinds():
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[_hop_diag(per_lookup=[_lookup_diag(
            kind_counts={"fact": 5, "chunk": 3, "bogus_kind": 99},
        )])],
    )
    kinds = {kc.kind for kc in m.hops[0].lookups[0].kind_counts}
    assert s.RecordKind.fact in kinds and s.RecordKind.chunk in kinds
    # bogus_kind is dropped silently — closed-vocab guard at the
    # builder boundary keeps the leaf type tight even if the
    # dispatcher's dict drifts.
    assert all(isinstance(kc.kind, s.RecordKind)
               for kc in m.hops[0].lookups[0].kind_counts)


def test_degenerate_lookup_per_lookup_flag_set():
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[_hop_diag(per_lookup=[_lookup_diag(
            degenerate=True, k_returned=0, kind_counts={},
            score_samples=None,
        )])],
    )
    lk = m.hops[0].lookups[0]
    assert lk.degenerate is True
    assert lk.k_returned == 0
    assert lk.score_samples is None


def test_tied_lookup_per_lookup_flag_set():
    m = _build(
        lookup_fired=True, store_bound=True,
        hops_diag=[_hop_diag(per_lookup=[_lookup_diag(
            tied=True,
        )])],
    )
    assert m.hops[0].lookups[0].tied is True


def test_full_marker_passes_content_free_guard_end_to_end():
    # 3-hop turn: invalid attempt + tool_call + prose finalize.
    m = _build(
        lookup_fired=True, store_bound=True, bound_run="ab3k",
        store_stats=sm.build_store_stats(_FakeStore(50, {"chunk": 50})),
        history_turn_count=4, resources_emitted_count=3,
        hops_diag=[
            _hop_diag(hop_outcome="invalid_tool_call",
                      store_open_ms=None, dispatch_ms=None,
                      union_after=None),
            _hop_diag(per_lookup=[
                _lookup_diag(entry_types=["fact"], k=20),
                _lookup_diag(entry_types=["chunk"], k=20,
                             k_returned=0, kind_counts={},
                             score_samples=None,
                             junk_dropped=5),
            ]),
            _hop_diag(hop_outcome="prose_answer", streamed=True,
                      store_open_ms=None, dispatch_ms=None,
                      union_after=None),
        ],
    )
    s._assert_content_free(m)


# ── D: skip-reason resolver (hops-aware) ─────────────────────────────


def test_skipped_reason_empty_store_when_dispatch_short_circuits():
    # _dispatch_search hits store.count()==0 and emits empty=True
    # alongside per_lookup=[]. Marker resolver should pick empty_store.
    m = _build(
        lookup_fired=True, store_bound=True, bound_run="ab3k",
        hops_diag=[{
            "hop_outcome": "tool_call",
            "streamed_to_user": False,
            "lookups_remaining_in_budget": 3,
            "previous_attempts_count": 0,
            "store_open_latency_ms": 1.5, "dispatch_latency_ms": 0.5,
            "union_size_after": 0,
            "per_lookup": [],
            "empty": True,
        }],
    )
    assert m.retrieve_skipped_reason is s.RetrieveSkippedReason.empty_store


def test_skipped_reason_degenerate_query_when_all_lookups_degenerate():
    # Every lookup in the tool call degenerated → union ended empty.
    # The dispatcher emits degenerate_dropped>0 + union_size=0.
    m = _build(
        lookup_fired=True, store_bound=True, bound_run="ab3k",
        store_stats=sm.build_store_stats(_FakeStore(50, {"chunk": 50})),
        hops_diag=[{
            "hop_outcome": "tool_call",
            "streamed_to_user": False,
            "lookups_remaining_in_budget": 3,
            "previous_attempts_count": 0,
            "store_open_latency_ms": 1.5, "dispatch_latency_ms": 0.5,
            "union_size_after": 0,
            "per_lookup": [_lookup_diag(
                degenerate=True, k_returned=0, kind_counts={},
                score_samples=None,
            )],
            "degenerate_dropped": 1, "tied_dropped": 0,
            "union_size": 0, "junk_dropped": 0,
        }],
    )
    assert m.retrieve_skipped_reason is s.RetrieveSkippedReason.degenerate_query


def test_skipped_reason_none_when_dispatch_partially_recovers():
    # Some degenerate lookups but the union ended non-empty (other
    # lookups contributed). Not a skip.
    m = _build(
        lookup_fired=True, store_bound=True, bound_run="ab3k",
        store_stats=sm.build_store_stats(_FakeStore(50, {"chunk": 50})),
        hops_diag=[{
            "hop_outcome": "tool_call",
            "streamed_to_user": False,
            "lookups_remaining_in_budget": 3,
            "previous_attempts_count": 0,
            "store_open_latency_ms": 1.5, "dispatch_latency_ms": 5.0,
            "union_size_after": 7,
            "per_lookup": [_lookup_diag(), _lookup_diag(degenerate=True)],
            "degenerate_dropped": 1, "tied_dropped": 0,
            "union_size": 7, "junk_dropped": 0,
        }],
    )
    assert m.retrieve_skipped_reason is s.RetrieveSkippedReason.none


def test_skipped_reason_none_when_dispatch_ran_clean_no_match():
    # Healthy dispatch, all lookups returned 0 (real corpus miss). Not
    # a skip — diag is present and shows the dispatch ran.
    m = _build(
        lookup_fired=True, store_bound=True, bound_run="ab3k",
        store_stats=sm.build_store_stats(_FakeStore(50, {"chunk": 50})),
        hops_diag=[{
            "hop_outcome": "tool_call",
            "streamed_to_user": False,
            "lookups_remaining_in_budget": 3,
            "previous_attempts_count": 0,
            "store_open_latency_ms": 1.5, "dispatch_latency_ms": 4.0,
            "union_size_after": 0,
            "per_lookup": [_lookup_diag(
                k_returned=0, kind_counts={}, score_samples=None,
            )],
            "degenerate_dropped": 0, "tied_dropped": 0,
            "union_size": 0, "junk_dropped": 0,
        }],
    )
    assert m.retrieve_skipped_reason is s.RetrieveSkippedReason.none


def test_skipped_reason_sink_never_called_when_no_hop_dispatched():
    # Bound non-empty store, lookup_fired=True (the model SHOULD have
    # dispatched), but no hop carries per_lookup → dispatch never ran.
    # Defensive: this is the "sink_never_called" case from #818.
    m = _build(
        lookup_fired=True, store_bound=True, bound_run="ab3k",
        store_stats=sm.build_store_stats(_FakeStore(50, {"chunk": 50})),
        hops_diag=[_hop_diag(
            hop_outcome="invalid_tool_call",
            store_open_ms=None, dispatch_ms=None, union_after=None,
        )],
    )
    assert m.retrieve_skipped_reason is s.RetrieveSkippedReason.sink_never_called


def test_skipped_reason_empty_store_when_no_dispatch_and_store_empty():
    # No hop dispatched (chat-side pre-check sidecar would skip
    # retrieve) AND store_stats reports 0 records.
    m = _build(
        lookup_fired=True, store_bound=True, bound_run="ab3k",
        store_stats=s.StoreStats(total_records=0, records_by_kind=()),
        hops_diag=[_hop_diag(
            hop_outcome="prose_answer", streamed=True,
            store_open_ms=None, dispatch_ms=None, union_after=None,
        )],
    )
    assert m.retrieve_skipped_reason is s.RetrieveSkippedReason.empty_store


# ── E: end-to-end YAML acceptance (the original #781 motivating case) ──


def test_m9pp_repro_distinguishable_from_real_miss_in_yaml(
    tmp_path, monkeypatch,
):
    """The #781 acceptance contract carried forward: a turn fired
    against an unbound store renders distinguishable in the yaml from
    a healthy retrieval turn — without inspecting transcript.json or
    sidecar source."""
    import yaml as pyyaml

    monkeypatch.setenv("HOME", str(tmp_path))

    empty_binding = _build(
        turn_index=1, lookup_fired=True, store_bound=False,
        bound_run=None, store_stats=None, hops_diag=[
            # Sidecar guarded the dispatch path entirely (no store
            # path resolved); the hop_outcome stays invalid /
            # prose-only because validate or dispatch never ran. In a
            # real session this hop wouldn't have fired at all (the
            # sidecar would have prose-finalized at the converse
            # call), but the marker schema still represents it
            # cleanly via retrieve_skipped_reason=no_bound_run.
        ],
    )
    healthy = _build(
        turn_index=2, lookup_fired=True, store_bound=True,
        bound_run="ab3k",
        store_stats=sm.build_store_stats(_FakeStore(50, {"chunk": 50})),
        hops_diag=[_hop_diag(per_lookup=[_lookup_diag()])],
    )
    s.emit(s.Stream.CHAT, "m9pp", empty_binding)
    s.emit(s.Stream.CHAT, "m9pp", healthy)

    f = next(
        (tmp_path / ".basevault" / "shareable" / "chat-diagnostics")
        .glob("*-m9pp-anonymized.yaml")
    )
    docs = list(pyyaml.safe_load_all(f.read_text()))
    assert len(docs) == 3  # header + 2 turns
    bad, good = docs[1], docs[2]
    assert bad["retrieve_skipped_reason"] == "no_bound_run"
    assert good["retrieve_skipped_reason"] == "none"
    assert "bound_run" not in bad  # null → omitted
    assert good["bound_run"] == "ab3k"
    # The good turn has the hops trace; the bad one has none.
    assert "hops" in good
    assert "hops" not in bad


def test_multihop_yaml_reads_per_call_per_lookup_detail(
    tmp_path, monkeypatch,
):
    """End-to-end: synthesize a real-shape 3-hop turn (similar to
    egas turn 7 in the motivating corpus) and verify the yaml
    exposes per-hop hop_outcome + per-lookup entry_types / k / kind_counts
    / score_samples. The acceptance criterion this pins is that the
    multi-hop trace is reconstructable from the shareable alone."""
    import yaml as pyyaml

    monkeypatch.setenv("HOME", str(tmp_path))

    m = _build(
        turn_index=1, lookup_fired=True, store_bound=True,
        bound_run="ab3k",
        store_stats=sm.build_store_stats(_FakeStore(50, {"chunk": 50})),
        hops_diag=[
            _hop_diag(hop_outcome="invalid_tool_call",
                      store_open_ms=None, dispatch_ms=None,
                      union_after=None,
                      prev_attempts=0),
            _hop_diag(per_lookup=[
                _lookup_diag(entry_types=["chunk"], k=20,
                             k_returned=0, kind_counts={},
                             score_samples=None),
            ], prev_attempts=1),
            _hop_diag(per_lookup=[
                _lookup_diag(entry_types=["fact", "chunk"], k=20,
                             k_returned=12,
                             kind_counts={"fact": 7, "chunk": 5},
                             score_samples=(0.26, 0.31, 0.34)),
            ], prev_attempts=2, union_after=12),
        ],
    )
    s.emit(s.Stream.CHAT, "abcd", m)

    f = next(
        (tmp_path / ".basevault" / "shareable" / "chat-diagnostics")
        .glob("*-abcd-anonymized.yaml")
    )
    docs = list(pyyaml.safe_load_all(f.read_text()))
    turn = docs[1]
    assert len(turn["hops"]) == 3
    assert turn["hops"][0]["hop_outcome"] == "invalid_tool_call"
    assert turn["hops"][1]["hop_outcome"] == "tool_call"
    assert turn["hops"][1]["lookups"][0]["entry_types"] == ["chunk"]
    assert turn["hops"][1]["lookups"][0]["k_requested"] == 20
    assert turn["hops"][1]["lookups"][0]["k_returned"] == 0
    # The healthy lookup carries the full request+outcome
    hop3 = turn["hops"][2]
    assert hop3["lookups"][0]["entry_types"] == ["fact", "chunk"]
    assert hop3["lookups"][0]["k_returned"] == 12
    kc = {row["kind"]: row["count"] for row in hop3["lookups"][0]["kind_counts"]}
    assert kc == {"fact": 7, "chunk": 5}
    ss = hop3["lookups"][0]["score_samples"]
    assert ss["closest"] == 0.26
    assert ss["middle"] == 0.31
    assert ss["farthest"] == 0.34
    # The previous_attempts_count surfaces per hop
    assert turn["hops"][2]["previous_attempts_count"] == 2


def test_chat_and_run_no_duplicated_info_after_781_v2():
    """The disjoint-schemas invariant survives the multi-hop rev: all
    new chat-only fields stay chat-only; run schema unchanged."""
    chat_fields = {f.name for f in dataclasses.fields(s.ChatMarker)}
    run_fields = {f.name for f in dataclasses.fields(s.RunMarker)}
    for chat_only in (
        "retrieve_skipped_reason", "bound_run", "store_stats",
        "history_turn_count", "resources_emitted_count",
        "session_id", "hops", "lookup_fired",
    ):
        assert chat_only in chat_fields and chat_only not in run_fields

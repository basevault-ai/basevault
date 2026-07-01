"""Unit tests for the sqlite-vec-backed local vector store."""
from __future__ import annotations

import sqlite3

import pytest

from engine.rag_vector_store import (
    EMBEDDING_DIM,
    StoredRecord,
    VectorStore,
    open_store,
)


def _vec(seed: float, dim: int) -> list[float]:
    return [seed + i * 0.001 for i in range(dim)]


def test_open_creates_file_and_schema(tmp_path):
    path = tmp_path / "v.db"
    with VectorStore(path, dim=8) as store:
        assert path.exists()
        assert store.count() == 0
        # Records + vectors tables are reachable via raw connection.
        rows = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','virtual')"
        ).fetchall()
        names = {r[0] for r in rows}
        assert "records" in names
        assert "vectors" in names


def test_add_and_count_roundtrip(tmp_path):
    path = tmp_path / "v.db"
    recs = [
        StoredRecord(kind="chunk", record_id="f1@0", text="alpha", file_id="f1"),
        StoredRecord(kind="fact", record_id="health:0", text="beta", topic="health"),
        StoredRecord(kind="entity", record_id="e1", text="gamma"),
    ]
    vecs = [_vec(0.1, 8), _vec(0.2, 8), _vec(0.3, 8)]
    with VectorStore(path, dim=8) as store:
        store.add(recs, vecs)
        assert store.count() == 3
        assert store.count(kind="chunk") == 1
        assert store.count(kind="fact") == 1
        assert store.count_by_kind() == {"chunk": 1, "fact": 1, "entity": 1}


def test_query_filtered_returns_nearest_first(tmp_path):
    path = tmp_path / "v.db"
    recs = [
        StoredRecord(kind="chunk", record_id="r0", text="t0"),
        StoredRecord(kind="chunk", record_id="r1", text="t1"),
        StoredRecord(kind="chunk", record_id="r2", text="t2"),
    ]
    vecs = [_vec(0.0, 8), _vec(0.5, 8), _vec(0.99, 8)]
    with VectorStore(path, dim=8) as store:
        store.add(recs, vecs)
        out = store.query_filtered(_vec(0.0, 8), k=3)
    assert [r.record_id for r, _ in out] == ["r0", "r1", "r2"]
    assert out[0][1] < out[1][1] < out[2][1]


def test_query_filtered_by_kind(tmp_path):
    path = tmp_path / "v.db"
    recs = [
        StoredRecord(kind="chunk", record_id="c0", text="t"),
        StoredRecord(kind="fact", record_id="f0", text="t", topic="x"),
        StoredRecord(kind="entity", record_id="e0", text="t"),
    ]
    vecs = [_vec(0.0, 4), _vec(0.0, 4), _vec(0.0, 4)]
    with VectorStore(path, dim=4) as store:
        store.add(recs, vecs)
        facts_only = store.query_filtered(_vec(0.0, 4), k=10, kinds=("fact",))
    assert {r.kind for r, _ in facts_only} == {"fact"}
    assert [r.record_id for r, _ in facts_only] == ["f0"]


def test_source_filter_matches_file_id_case_insensitive_substring(tmp_path):
    """The `source` (file_id) filter anchors a lookup on a named file:
    a case-insensitive substring of file_id, on both the query-bearing
    (`query_filtered`) and filter-only (`filter_select`) paths."""
    path = tmp_path / "v.db"
    recs = [
        StoredRecord(kind="document", record_id="aurelius-meditations.txt",
                     text="meditations file", file_id="aurelius-meditations.txt"),
        StoredRecord(kind="chunk", record_id="aurelius-meditations.txt@0",
                     text="on the shortness of life", file_id="aurelius-meditations.txt"),
        StoredRecord(kind="chunk", record_id="other.txt@0",
                     text="unrelated", file_id="other.txt"),
    ]
    vecs = [_vec(0.0, 4), _vec(0.1, 4), _vec(0.9, 4)]
    with VectorStore(path, dim=4) as store:
        store.add(recs, vecs)
        # Filter-only: every record whose file_id contains "meditations".
        sel = store.filter_select(limit=10, file_ids=("Meditations",))
        assert {r.file_id for r in sel} == {"aurelius-meditations.txt"}
        assert {r.record_id for r in sel} == {
            "aurelius-meditations.txt", "aurelius-meditations.txt@0",
        }
        # Query-bearing: same restriction, ranked by distance.
        hits = store.query_filtered(
            _vec(0.0, 4), k=10, file_ids=("meditations.txt",),
        )
        assert all(r.file_id == "aurelius-meditations.txt" for r, _ in hits)
        # A name that matches nothing returns nothing (not a fall-through
        # to the whole store).
        assert store.filter_select(limit=10, file_ids=("nope",)) == []


def test_source_filter_composes_with_kind(tmp_path):
    """`source` AND-composes with `entry_type` — file X's document record
    only, the shape "tell me about file X" issues."""
    path = tmp_path / "v.db"
    recs = [
        StoredRecord(kind="document", record_id="trip.txt", text="d",
                     file_id="trip.txt"),
        StoredRecord(kind="chunk", record_id="trip.txt@0", text="c",
                     file_id="trip.txt"),
    ]
    with VectorStore(path, dim=4) as store:
        store.add(recs, [_vec(0.0, 4), _vec(0.1, 4)])
        sel = store.filter_select(
            limit=10, kinds=("document",), file_ids=("trip",),
        )
    assert [(r.kind, r.record_id) for r in sel] == [("document", "trip.txt")]


def test_dim_mismatch_on_add_raises(tmp_path):
    path = tmp_path / "v.db"
    with VectorStore(path, dim=8) as store:
        with pytest.raises(ValueError):
            store.add(
                [StoredRecord(kind="chunk", record_id="r", text="t")],
                [[0.0] * 7],
            )


def test_records_and_vectors_alignment_required(tmp_path):
    path = tmp_path / "v.db"
    with VectorStore(path, dim=4) as store:
        with pytest.raises(ValueError):
            store.add(
                [StoredRecord(kind="chunk", record_id="r", text="t")],
                [],
            )


def test_open_store_context_manager_yields_open_store(tmp_path):
    with open_store(tmp_path / "v.db", dim=4) as store:
        assert isinstance(store, VectorStore)
        assert store.count() == 0


def test_standalone_open_reads_persisted_data(tmp_path):
    """An external consumer (Python REPL, CLI) can open the .db file
    and query its contents — sanity check the acceptance criterion."""
    path = tmp_path / "v.db"
    rec = StoredRecord(
        kind="chunk", record_id="r0", text="hello",
        file_id="f1", topic="x", char_offset=42,
    )
    with VectorStore(path, dim=4) as store:
        store.add([rec], [_vec(0.0, 4)])

    db = sqlite3.connect(str(path))
    rows = db.execute(
        "SELECT kind, record_id, text, file_id, topic, char_offset FROM records"
    ).fetchall()
    db.close()
    assert rows == [("chunk", "r0", "hello", "f1", "x", 42)]


def test_meta_table_records_embedding_dim(tmp_path):
    path = tmp_path / "v.db"
    with VectorStore(path, dim=EMBEDDING_DIM):
        pass
    db = sqlite3.connect(str(path))
    val = db.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()
    db.close()
    assert val == (str(EMBEDDING_DIM),)


# ── Cosine metric + L2->cosine migration ────────────────────────────────────


def _load_sqlite_vec(conn: sqlite3.Connection) -> None:
    import sqlite_vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _vectors_ddl(path) -> str:
    db = sqlite3.connect(str(path))
    try:
        row = db.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type='table' AND name='vectors'"
        ).fetchone()
        return (row and row[0]) or ""
    finally:
        db.close()


def _seed_l2_db(path, dim: int = 4) -> int:
    """Build a pre-cosine (L2) vec0 db at ``path`` with a couple of
    unit-normalized records + vectors. Returns the row count for the
    migration assertion."""
    import struct
    db = sqlite3.connect(str(path))
    _load_sqlite_vec(db)
    db.executescript("""
        CREATE TABLE records(
          rowid INTEGER PRIMARY KEY, kind TEXT NOT NULL,
          record_id TEXT NOT NULL, text TEXT NOT NULL,
          file_id TEXT NOT NULL DEFAULT '',
          source_ref TEXT NOT NULL DEFAULT '',
          section_path TEXT NOT NULL DEFAULT '',
          topic TEXT NOT NULL DEFAULT '',
          char_offset INTEGER NOT NULL DEFAULT 0,
          extra_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    db.execute(
        f"CREATE VIRTUAL TABLE vectors USING vec0(embedding float[{dim}])"
    )
    db.execute(
        "INSERT INTO meta(key,value) VALUES('embedding_dim',?)", (str(dim),)
    )
    # Two orthogonal unit vectors so the post-migration cosine query has
    # a well-defined expectation (self-match → 0.0, orthogonal → 1.0).
    e0 = [1.0] + [0.0] * (dim - 1)
    e1 = [0.0, 1.0] + [0.0] * (dim - 2)
    db.execute(
        "INSERT INTO records(rowid,kind,record_id,text) VALUES (1,'fact','a','alpha')"
    )
    db.execute(
        "INSERT INTO vectors(rowid,embedding) VALUES (1,?)",
        (struct.pack(f"{dim}f", *e0),),
    )
    db.execute(
        "INSERT INTO records(rowid,kind,record_id,text) VALUES (2,'fact','b','beta')"
    )
    db.execute(
        "INSERT INTO vectors(rowid,embedding) VALUES (2,?)",
        (struct.pack(f"{dim}f", *e1),),
    )
    db.commit()
    db.close()
    return 2


def test_fresh_db_creates_cosine_vec0_table(tmp_path):
    """A brand-new store creates the vec0 table with the cosine metric
    declared on the column, not the sqlite-vec L2 default."""
    path = tmp_path / "v.db"
    with VectorStore(path, dim=8):
        pass
    assert "distance_metric=cosine" in _vectors_ddl(path).lower()


def test_fresh_db_stamps_distance_metric_in_meta(tmp_path):
    path = tmp_path / "v.db"
    with VectorStore(path, dim=8):
        pass
    db = sqlite3.connect(str(path))
    val = db.execute(
        "SELECT value FROM meta WHERE key='distance_metric'"
    ).fetchone()
    db.close()
    assert val == ("cosine",)


def test_migrates_l2_db_to_cosine_preserving_data(tmp_path, capsys):
    """Open a pre-cosine L2 db: migration runs once (logs to stderr),
    rebuilds the vec0 table as cosine, records and vectors are
    preserved, query returns sensible cosine distances, and the
    transitional ``.migrating`` temp file is cleaned up."""
    path = tmp_path / "v.db"
    n = _seed_l2_db(path, dim=4)
    assert "distance_metric=cosine" not in _vectors_ddl(path).lower()

    with VectorStore(path, dim=4) as store:
        # Post-migration shape.
        assert "distance_metric=cosine" in _vectors_ddl(path).lower()
        assert store.count() == n
        # Cosine semantics: querying with the first stored unit vector
        # finds it at distance 0 and the orthogonal one at distance 1.
        results = store.query_filtered([1.0, 0.0, 0.0, 0.0], k=2)
        ids = [r.record_id for r, _ in results]
        dists = [d for _, d in results]
        assert ids == ["a", "b"]
        assert dists[0] == pytest.approx(0.0, abs=1e-5)
        assert dists[1] == pytest.approx(1.0, abs=1e-5)

    # The atomic temp file must be removed on success.
    assert not (tmp_path / "v.db.migrating").exists()
    # One migration log line, naming the file and row count.
    captured = capsys.readouterr().err
    assert "L2 -> cosine" in captured
    assert f"({n} vectors)" in captured


def test_migration_is_idempotent_for_already_cosine_db(tmp_path, capsys):
    """Opening a db that's already cosine must NOT run the rebuild — no
    migration log line, no temp file."""
    path = tmp_path / "v.db"
    with VectorStore(path, dim=4):
        pass
    capsys.readouterr()  # drain anything from the first open
    with VectorStore(path, dim=4):
        pass
    assert "L2 -> cosine" not in capsys.readouterr().err
    assert not (tmp_path / "v.db.migrating").exists()


def test_fail_loud_when_l2_db_bypasses_migration(tmp_path, monkeypatch):
    """The guard in ``open()`` outlives the transitional migration body:
    if an L2 db ever reaches the connection (here: by stubbing the
    migration to a no-op), ``open()`` must raise rather than silently
    apply the cosine junk floor to L2 distances."""
    from engine import rag_vector_store as rvs
    path = tmp_path / "v.db"
    _seed_l2_db(path, dim=4)
    monkeypatch.setattr(rvs, "_migrate_to_cosine_if_needed", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="stale L2 vector db"):
        VectorStore(path, dim=4).open()


# ── Edge table — schema, roundtrip, idempotency ──────────────────────────────


def test_open_creates_edges_table_and_indices(tmp_path):
    path = tmp_path / "v.db"
    with VectorStore(path, dim=4) as store:
        rows = store.conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type IN ('table','index') AND name LIKE 'edges%'"
        ).fetchall()
    names = {r[0] for r in rows}
    assert "edges" in names
    assert "edges_src" in names
    assert "edges_dst" in names


def test_add_edges_persists_and_count_roundtrips(tmp_path):
    path = tmp_path / "v.db"
    edges = [
        ("chunk", "f1@0", "fact", "health:0", "containment"),
        ("fact", "health:0", "chunk", "f1@0", "containment"),
        ("entity", "alice", "fact", "health:0", "mention"),
        ("fact", "health:0", "entity", "alice", "mention"),
    ]
    def _count(store, where="", params=()):
        sql = "SELECT COUNT(*) FROM edges" + (f" WHERE {where}" if where else "")
        return store.conn.execute(sql, params).fetchone()[0]

    with VectorStore(path, dim=4) as store:
        inserted = store.add_edges(edges)
        assert inserted == 4
        assert _count(store) == 4
        assert _count(store, "src_kind = ?", ("chunk",)) == 1
        assert _count(store, "src_kind = ?", ("fact",)) == 2
        assert _count(store, "src_kind = ? AND dst_kind = ?", ("fact", "entity")) == 1


def test_add_edges_is_idempotent_on_repeat(tmp_path):
    """Re-emitting the same edge set must not duplicate rows — PK on
    the 4-tuple of endpoints collapses identical anchor↔neighbor pairs
    to one row."""
    path = tmp_path / "v.db"
    edges = [("entity", "alice", "fact", "health:0", "mention")] * 3
    with VectorStore(path, dim=4) as store:
        store.add_edges(edges)
        store.add_edges(edges)
        assert store.conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == 1


def test_add_edges_stamps_meta_edges_emitted(tmp_path):
    path = tmp_path / "v.db"
    with VectorStore(path, dim=4) as store:
        store.add_edges([("entity", "a", "fact", "x:0", "mention")])
    db = sqlite3.connect(str(path))
    val = db.execute("SELECT value FROM meta WHERE key='edges_emitted'").fetchone()
    db.close()
    assert val == ("1",)


def test_add_edges_empty_still_stamps_meta_emitted(tmp_path):
    """An empty `add_edges([])` MUST still stamp ``meta.edges_emitted=1``:
    the call itself is the marker that stage-2 ran, distinguishing a
    legitimately-empty graph (single chunk with no extracted artifacts,
    every batch gave up, etc.) from a pre-stage-2 db that never saw
    `add_edges` at all. Without the stamp on the empty path the
    compatibility marker is undefined for the zero-edges case."""
    path = tmp_path / "v.db"
    with VectorStore(path, dim=4) as store:
        assert store.add_edges([]) == 0
    db = sqlite3.connect(str(path))
    val = db.execute("SELECT value FROM meta WHERE key='edges_emitted'").fetchone()
    db.close()
    assert val == ("1",)


def test_add_edges_never_called_leaves_meta_unstamped(tmp_path):
    """A db opened but never written to via `add_edges` (or fully
    pre-stage-2 vintage) has no `edges_emitted` meta key — that's the
    "not edge-aware" signal a future has_neighbor consumer keys on to
    decide between returning empty results and walking the index."""
    path = tmp_path / "v.db"
    with VectorStore(path, dim=4):
        pass
    db = sqlite3.connect(str(path))
    val = db.execute("SELECT value FROM meta WHERE key='edges_emitted'").fetchone()
    db.close()
    assert val is None


def test_has_neighbor_or_join_shape(tmp_path):
    """The dispatcher's future has_neighbor filter is a parameterized
    `WHERE (src_kind, src_id) IN (...)` join — pin the SQL shape here
    to lock the schema's neighbor-walk surface (#785 acceptance #3).
    """
    path = tmp_path / "v.db"
    edges = [
        ("entity", "alice", "fact", "x:0", "mention"),
        ("entity", "alice", "fact", "x:1", "mention"),
        ("entity", "bob",   "fact", "x:0", "mention"),
        ("fact",   "x:0",   "entity", "alice", "mention"),
    ]
    with VectorStore(path, dim=4) as store:
        store.add_edges(edges)
        # Bound parameters — no string interpolation from model input
        # ever required (slice-2 acceptance #3 / #620 safety).
        rows = store.conn.execute(
            """SELECT dst_kind, dst_id FROM edges
               WHERE (src_kind, src_id) IN (
                   (?, ?),
                   (?, ?)
               )
               AND dst_kind = ?
               ORDER BY dst_id""",
            ("entity", "alice", "entity", "bob", "fact"),
        ).fetchall()
    assert rows == [("fact", "x:0"), ("fact", "x:0"), ("fact", "x:1")]

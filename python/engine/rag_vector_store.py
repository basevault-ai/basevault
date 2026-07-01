"""
Sqlite-vec-backed local vector store for the Embeddings stage.

Single .db file per run under `<run_dir>/stages/06-embeddings/vectors.db`.
One `vec0` virtual table holds the float[768] embeddings; a sibling
`records` table holds the per-row metadata that downstream retrieval
(PR 4.1) and enrichment (PR 4.2) need. An `edges` table holds the
derivation/mention adjacency between records keyed on the same
`(kind, record_id)` pair the records table uses, so neighbor walks
join on indexed columns without re-deriving from upstream artifacts.

Pick rationale (gate-1, vector store choice = sqlite-vec):
  - Single-process, single-file → fits the Tauri Python-sidecar model;
    no extra runtime to bundle or port to manage.
  - Standard SQL surface — KNN via `embedding MATCH ?` on the vec0
    table, joined to a sibling records table for filter-then-KNN or
    KNN-then-filter. Familiar shape for the retrieval worker.
  - Apache-2.0, distribution-friendly.
"""
from __future__ import annotations

import os
import sqlite3
import struct
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


EMBEDDING_DIM = 768

# Distance metric for the vec0 table. Cosine on the unit-normalized
# embeddings the embedder emits — the right metric for text similarity,
# and what the retrieval junk floor below is calibrated against.
_DISTANCE_METRIC = "cosine"

# Cosine-distance above which a `closest_match` hit is treated as junk
# and dropped by the retrieval surface. Calibrated on real (unit-norm)
# embeddings: relevant hits land ≲0.4, junk ≳0.55, so 0.5 is the boundary
# (equivalently the measured L2 1.0 boundary, since unit-norm cosine and
# L2 are monotone). Recalibrate if the embedder changes.
COSINE_JUNK_DISTANCE = 0.5

# Canonical set of record kinds. Order matches the pipeline stages so
# inserts land grouped by stage when callers iterate in order; the
# downstream retrieval surface filters on `kind` cheaply.
#
# `document` is the file-level record: one per ingested source file,
# carrying its name / date / section structure / chunk count so the
# chatbot can answer file-inventory questions ("what files do I have?",
# "tell me about file X") via ordinary retrieval, and `has_neighbor`
# walks between a chunk and its document.
RECORD_KINDS: tuple[str, ...] = (
    "document", "chunk", "fact", "entity", "pattern", "insight", "action",
)


@dataclass(frozen=True)
class StoredRecord:
    kind: str           # one of RECORD_KINDS
    record_id: str      # canonical id within (kind, run)
    text: str           # embed input text — graph-enriched, what the
                        # embedder sees
    file_id: str = ""
    source_ref: str = ""
    section_path: str = ""
    topic: str = ""
    char_offset: int = 0
    extra: dict = field(default_factory=dict)
    # Bare content prose for the model-facing CONTEXT surface — same
    # source record as `text`, with the graph-enriched prefix stripped
    # so the model never sees canonical-id-shape leakage (kind brackets,
    # name-lists referencing other records) when reading retrieved
    # context. Empty on records minted before the split (read path
    # falls back to `text` until the next re-extraction).
    display_text: str = ""


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def _unpack(buf: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"{dim}f", buf))


def _load_extension(conn: sqlite3.Connection) -> None:
    import sqlite_vec
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


_SCHEMA_RECORDS = """
CREATE TABLE IF NOT EXISTS records(
    rowid INTEGER PRIMARY KEY,
    kind TEXT NOT NULL,
    record_id TEXT NOT NULL,
    text TEXT NOT NULL,
    file_id TEXT NOT NULL DEFAULT '',
    source_ref TEXT NOT NULL DEFAULT '',
    section_path TEXT NOT NULL DEFAULT '',
    topic TEXT NOT NULL DEFAULT '',
    char_offset INTEGER NOT NULL DEFAULT 0,
    extra_json TEXT NOT NULL DEFAULT '{}',
    display_text TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS records_kind ON records(kind);
CREATE INDEX IF NOT EXISTS records_file_id ON records(file_id);
"""

_SCHEMA_META = """
CREATE TABLE IF NOT EXISTS meta(
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

# Directed edge relation between two records, keyed on the same
# `(kind, record_id)` pair the `records` table uses. The chatbot
# dispatcher's `has_neighbor` filter joins on `(src_kind, src_id)`
# against bound id params, narrowed optionally by `dst_kind`.
# `edge_kind` is a short descriptive label (mention / derivation /
# sibling / relation / containment) used for diagnostics + future
# selective traversal; the primary key is the 4-tuple of endpoints so
# re-emit is idempotent and the same pair carries one edge. Two
# indices: the forward `(src_kind, src_id)` covers the dispatcher's
# hot path; the backward `(dst_kind, dst_id)` keeps reverse queries
# constant-time for the future ReAct surfaces that will read both
# directions.
_SCHEMA_EDGES = """
CREATE TABLE IF NOT EXISTS edges(
    src_kind TEXT NOT NULL,
    src_id TEXT NOT NULL,
    dst_kind TEXT NOT NULL,
    dst_id TEXT NOT NULL,
    edge_kind TEXT NOT NULL,
    PRIMARY KEY (src_kind, src_id, dst_kind, dst_id)
);
CREATE INDEX IF NOT EXISTS edges_src ON edges(src_kind, src_id);
CREATE INDEX IF NOT EXISTS edges_dst ON edges(dst_kind, dst_id);
"""

# Edge-kind labels surfaced from `rag_enricher.build_edges`. The
# dispatcher's neighbor filter doesn't key on this field; it's a
# diagnostic label so a SQL inspection (`SELECT edge_kind, COUNT(*)
# FROM edges GROUP BY edge_kind`) is legible.
EDGE_KIND_MENTION = "mention"
EDGE_KIND_DERIVATION = "derivation"
EDGE_KIND_SIBLING = "sibling"
EDGE_KIND_RELATION = "relation"
EDGE_KIND_CONTAINMENT = "containment"

def _ensure_records_columns(conn: sqlite3.Connection) -> None:
    """Idempotent column-add migration for ``records`` so a db minted
    before a column was introduced picks it up at open() without a
    backfill pass. Existing rows take the DEFAULT and the read path's
    fallback handles them (e.g. ``display_text`` empty → read code
    falls back to the embedded ``text``). New writes carry the full
    column set from this run forward."""
    have = set(_records_columns(conn))
    if "display_text" not in have:
        conn.execute(
            "ALTER TABLE records "
            "ADD COLUMN display_text TEXT NOT NULL DEFAULT ''"
        )


def _records_columns(conn: sqlite3.Connection) -> list[str]:
    """Column names of the ``records`` table, in declaration order.
    Driven from ``PRAGMA table_info`` so the L2→cosine migration
    automatically picks up any future column added to
    ``_SCHEMA_RECORDS`` — no parallel hardcoded list to keep in sync,
    and a column added to the schema without a backfill plan flows
    through the migration losslessly instead of being silently dropped
    (the prior hardcoded-list approach only failed loud for NOT NULL
    columns).
    """
    return [r[1] for r in conn.execute("PRAGMA table_info(records)")]


def _vectors_is_cosine(conn: sqlite3.Connection) -> bool | None:
    """Inspect ``sqlite_master`` to tell whether this db's ``vectors``
    table is the cosine-metric vec0. Returns ``True`` / ``False`` when
    a vectors table exists, ``None`` when it doesn't (fresh db: the
    caller will create one cosine)."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master "
        "WHERE type='table' AND name='vectors'"
    ).fetchone()
    if row is None or not row[0]:
        return None
    return "distance_metric=cosine" in row[0].lower()


def _copy_as_cosine(
    src: sqlite3.Connection, dst: sqlite3.Connection, dim: int,
) -> int:
    """Materialize a cosine clone of ``src`` into the empty ``dst``:
    copy ``records`` + ``meta`` verbatim, recreate ``vectors`` as a
    cosine vec0, and re-insert each embedding by its raw bytes (no
    unpack / repack — sqlite-vec returns the same packed-float blob the
    insert took). Returns the vector count for the migration log.
    """
    dst.executescript(_SCHEMA_RECORDS)
    dst.executescript(_SCHEMA_META)
    dst.execute(
        f"CREATE VIRTUAL TABLE vectors USING vec0("
        f"embedding float[{dim}] distance_metric={_DISTANCE_METRIC})"
    )
    # Derive the records-table column list from the source db itself
    # rather than a hardcoded constant — keeps the copy lossless when
    # ``_SCHEMA_RECORDS`` is extended later. If the source has a column
    # the destination's freshly-created schema lacks (i.e. the schema
    # change wasn't carried to ``_SCHEMA_RECORDS``), the INSERT below
    # fails loud with ``no such column: …`` — the missing-column
    # detection the previous hardcoded ``_RECORDS_COLS`` couldn't give
    # for nullable additions.
    cols = _records_columns(src)
    cols_csv = ", ".join(cols)
    placeholders = ", ".join("?" * len(cols))
    records_sql = f"SELECT {cols_csv} FROM records"
    insert_records = (
        f"INSERT INTO records ({cols_csv}) VALUES ({placeholders})"
    )
    for row in src.execute(records_sql):
        dst.execute(insert_records, row)
    for kv in src.execute("SELECT key, value FROM meta"):
        dst.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)", kv,
        )
    n = 0
    for rowid, emb in src.execute("SELECT rowid, embedding FROM vectors"):
        dst.execute(
            "INSERT INTO vectors(rowid, embedding) VALUES (?, ?)",
            (rowid, emb),
        )
        n += 1
    dst.execute(
        "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
        ("distance_metric", _DISTANCE_METRIC),
    )
    return n


def _migrate_to_cosine_if_needed(path: Path, dim: int) -> None:
    """Transitional: rebuild a pre-cosine (L2) vec0 db as cosine, copying
    the **stored** vectors (no re-embed). One-time, lazy — called once at
    ``VectorStore.open()``; nothing in the query path. Atomic via a
    sibling ``.migrating`` temp file + ``os.replace``: a crash mid-copy
    leaves the original db untouched and the next open retries.

    Remove this function (and its call site) once no L2 dbs remain in the
    wild; watch the migration log for the drain. The fail-loud guard in
    ``open()`` outlives this body and keeps catching a stale L2 db so the
    cosine junk floor never silently runs against L2 distances.
    """
    if not path.exists():
        return
    tmp = Path(str(path) + ".migrating")
    src = sqlite3.connect(str(path))
    _load_extension(src)
    migrated_count: int | None = None
    try:
        is_cos = _vectors_is_cosine(src)
        if is_cos is None or is_cos:
            return  # no vectors table yet, or already cosine
        if tmp.exists():
            tmp.unlink()
        dst = sqlite3.connect(str(tmp))
        _load_extension(dst)
        try:
            migrated_count = _copy_as_cosine(src, dst, dim)
            dst.commit()
        finally:
            dst.close()
    finally:
        src.close()
    if migrated_count is not None:
        os.replace(str(tmp), str(path))
        print(
            f"rag_vector_store: migrated {path.name} L2 -> cosine "
            f"({migrated_count} vectors)",
            file=sys.stderr,
        )


class VectorStore:
    """Thin wrapper over a sqlite-vec backed SQLite file. Open / close
    via the context manager; `add` is the bulk insert path; `query`
    fires a top-k KNN against the vec0 table and joins the matching
    records.
    """

    def __init__(self, path: Path, dim: int = EMBEDDING_DIM):
        self.path = Path(path)
        self.dim = dim
        self._conn: sqlite3.Connection | None = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("VectorStore is not open; use 'with' or call open()")
        return self._conn

    def open(self) -> "VectorStore":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Lazy, one-time migration of any pre-cosine db sitting on disk.
        # Runs before the working connection is established so the rest
        # of open() always sees a cosine db (or a fresh one we create
        # cosine below). Self-contained — nothing in the query path.
        _migrate_to_cosine_if_needed(self.path, self.dim)
        conn = sqlite3.connect(str(self.path))
        _load_extension(conn)
        conn.executescript(_SCHEMA_RECORDS)
        conn.executescript(_SCHEMA_META)
        conn.executescript(_SCHEMA_EDGES)
        _ensure_records_columns(conn)
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vectors USING vec0("
            f"embedding float[{self.dim}] distance_metric={_DISTANCE_METRIC})"
        )
        # Fail-loud guard. After migration + create, the vectors table
        # MUST be cosine; an L2 table here would mean a stale db slipped
        # past the rebuild — never silently apply the cosine junk floor
        # to L2 distances. Outlives the transitional migration above:
        # keep this check even after that body is removed.
        if _vectors_is_cosine(conn) is False:
            conn.close()
            raise RuntimeError(
                f"stale L2 vector db at {self.path}; re-run embeddings"
            )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("embedding_dim", str(self.dim)),
        )
        conn.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("distance_metric", _DISTANCE_METRIC),
        )
        conn.commit()
        self._conn = conn
        return self

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "VectorStore":
        return self.open()

    def __exit__(self, *_exc) -> None:
        self.close()

    def add(
        self,
        records: list[StoredRecord],
        vectors: list[list[float]],
    ) -> None:
        """Insert `records` + their `vectors` atomically. Lengths must
        match; vector dimensions must match the store's `dim`."""
        if len(records) != len(vectors):
            raise ValueError(
                f"records ({len(records)}) and vectors ({len(vectors)}) must align"
            )
        if not records:
            return
        import json as _json
        conn = self.conn
        cur = conn.cursor()
        for rec, vec in zip(records, vectors):
            if len(vec) != self.dim:
                raise ValueError(
                    f"vector dim {len(vec)} != store dim {self.dim} for {rec.kind}/{rec.record_id}"
                )
            cur.execute(
                """INSERT INTO records
                   (kind, record_id, text, file_id, source_ref,
                    section_path, topic, char_offset, extra_json,
                    display_text)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    rec.kind, rec.record_id, rec.text, rec.file_id,
                    rec.source_ref, rec.section_path, rec.topic,
                    rec.char_offset, _json.dumps(rec.extra, sort_keys=True),
                    rec.display_text,
                ),
            )
            rowid = cur.lastrowid
            cur.execute(
                "INSERT INTO vectors (rowid, embedding) VALUES (?, ?)",
                (rowid, _pack(vec)),
            )
        conn.commit()

    def count(self, kind: str | None = None) -> int:
        if kind is None:
            (n,) = self.conn.execute(
                "SELECT COUNT(*) FROM records"
            ).fetchone()
        else:
            (n,) = self.conn.execute(
                "SELECT COUNT(*) FROM records WHERE kind = ?", (kind,),
            ).fetchone()
        return n

    def count_by_kind(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT kind, COUNT(*) FROM records GROUP BY kind"
        ).fetchall()
        return {k: n for (k, n) in rows}

    def add_edges(
        self,
        edges: "list[tuple[str, str, str, str, str]]",
    ) -> int:
        """Bulk insert directed edges. Each tuple is
        `(src_kind, src_id, dst_kind, dst_id, edge_kind)`. Returns the
        count of rows inserted (post-dedupe — the PK on the 4-tuple of
        endpoints means re-emission collapses to a single row per
        anchor↔neighbor pair, regardless of label).

        Sets `meta.edges_emitted=1` on every call, including when the
        edge list is empty — the stamp marks the db as edge-aware so a
        consumer can distinguish a stage-2 run that legitimately
        produced zero edges (single chunk with no extracted artifacts,
        all batches gave up, etc.) from a pre-stage-2 db that never
        saw `add_edges` at all. Without the stamp on the empty path,
        the migration marker the docstring promises is undefined for
        the legitimately-empty case.
        """
        conn = self.conn
        cur = conn.cursor()
        inserted = 0
        if edges:
            cur.executemany(
                """INSERT OR IGNORE INTO edges
                   (src_kind, src_id, dst_kind, dst_id, edge_kind)
                   VALUES (?,?,?,?,?)""",
                edges,
            )
            inserted = cur.rowcount
        cur.execute(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            ("edges_emitted", "1"),
        )
        conn.commit()
        return inserted

    # ── Filter surface (slice-2 stage 3 dispatcher backend) ─────────────────
    #
    # Two read methods consumed by the chatbot dispatcher when the
    # model emits a multi-filter lookup. ``query_filtered`` is the
    # query-bearing path (vector KNN, ranked by cosine distance);
    # ``filter_select`` is the filter-only path (records-table SELECT,
    # no vec0). Both compose the same filter axes — ``entry_type``,
    # ``has_neighbor``, ``exact_match`` — as **bound** parameters; no
    # model output ever reaches an interpolated SQL string.
    #
    # Implementation note: ``has_neighbor`` is resolved into an in-memory
    # set of allowed ``(dst_kind, dst_id)`` tuples by a separate edges
    # query, then applied as a post-filter. Mixing the edges JOIN into
    # the vec0 MATCH is brittle (sqlite-vec's query planner restricts
    # what JOINs the KNN may sit alongside); a two-step lookup is
    # clearer, parameterizes the same, and the anchor sets are small.

    def neighbors_of(
        self,
        anchors: list[tuple[str, str]],
    ) -> set[tuple[str, str]]:
        """The set of ``(dst_kind, dst_id)`` reachable from any anchor
        in ``anchors`` via the persisted edges table. Empty when no
        edges have been written (or none match) — callers should treat
        the empty result as "no neighbors", not "no filter".
        """
        if not anchors:
            return set()
        placeholders = ",".join(["(?, ?)"] * len(anchors))
        flat: list[str] = []
        for kind, rid in anchors:
            flat.append(kind)
            flat.append(rid)
        sql = (
            f"SELECT dst_kind, dst_id FROM edges "
            f"WHERE (src_kind, src_id) IN ({placeholders})"
        )
        rows = self.conn.execute(sql, flat).fetchall()
        return {(r[0], r[1]) for r in rows}

    def query_filtered(
        self,
        vector: list[float],
        *,
        k: int,
        kinds: tuple[str, ...] = (),
        neighbor_ids: set[tuple[str, str]] | None = None,
        exact_match: tuple[str, ...] = (),
        file_ids: tuple[str, ...] = (),
        over_fetch: int = 200,
    ) -> list[tuple[StoredRecord, float]]:
        """Distance-ranked top-k constrained by the supplied filters.

        Two paths, picked by whether ANY records-side filter is set:

          * **Filter present** (``neighbor_ids`` / ``kinds`` /
            ``exact_match``) — load the matching records' vectors via a
            records-side SELECT compiled from the filter clauses, then
            compute cosine distance in Python over that pool, sort
            ascending, trim to ``k``. The bound is the matching
            records' size, not the whole store. This avoids the
            sqlite-vec MATCH semantics where a ``WHERE ... AND ...``
            filter is applied *after* the global top-k is computed
            (Codex P1 #612 / the earlier rowid-IN trap) — which
            silently drops valid matches when other rows dominate the
            distance ranking.
          * **No filter** — KNN over the full store via vec0
            ``embedding MATCH ? AND k = ?``. The straight global
            top-k; nothing to post-filter.

        ``neighbor_ids`` is the precomputed result of ``neighbors_of``
        — passed in already-resolved so multiple lookups in one tool
        call can share an anchors→neighbors lookup.
        """
        if len(vector) != self.dim:
            raise ValueError(
                f"query dim {len(vector)} != store dim {self.dim}"
            )
        if neighbor_ids is not None and not neighbor_ids:
            # Anchors resolved to no neighbors — return empty, never
            # fall through to vector hits the caller didn't ask for.
            return []
        import json as _json

        has_records_filter = (
            neighbor_ids is not None
            or bool(kinds)
            or bool(exact_match)
            or bool(file_ids)
        )

        if has_records_filter:
            # Records-first path: WHERE clause restricts the pool BEFORE
            # the rank, so a selective filter (e.g. a small entry_type
            # over a large store) can't be silently truncated by a
            # global top-k that didn't surface any matching rows.
            clause, params = _records_where(
                kinds=kinds,
                neighbor_ids=neighbor_ids,
                exact_match=exact_match,
                file_ids=file_ids,
            )
            sql = (
                "SELECT r.kind, r.record_id, r.text, r.file_id, "
                "r.source_ref, r.section_path, r.topic, r.char_offset, "
                "r.extra_json, r.display_text, v.embedding "
                "FROM records r JOIN vectors v ON v.rowid = r.rowid"
                + (f" WHERE {clause}" if clause else "")
            )
            rows = self.conn.execute(sql, params).fetchall()
            scored: list[tuple[StoredRecord, float]] = []
            for row in rows:
                stored_vec = _unpack(row[10], self.dim)
                # Cosine distance on unit-norm vectors:
                # 1 - dot(query, stored). Both sides are unit-norm by
                # contract (``_assert_unit_norm`` at embed-time + the
                # dispatcher's degenerate-query guard upstream); a
                # zero-norm row would zero the dot product → distance
                # 1.0, naturally sorted last.
                dot = 0.0
                for a, b in zip(vector, stored_vec):
                    dot += a * b
                distance = 1.0 - dot
                rec = StoredRecord(
                    kind=row[0], record_id=row[1], text=row[2],
                    file_id=row[3], source_ref=row[4],
                    section_path=row[5], topic=row[6],
                    char_offset=row[7], extra=_json.loads(row[8] or "{}"),
                    display_text=row[9] or "",
                )
                scored.append((rec, distance))
            scored.sort(key=lambda rd: rd[1])
            return scored[:k]

        # No records-side filter: standard vec0 KNN over the whole
        # store, over-fetched and trimmed.
        q = _pack(vector)
        pool_k = max(int(over_fetch), int(k))
        sql = (
            "SELECT r.kind, r.record_id, r.text, r.file_id, r.source_ref, "
            "r.section_path, r.topic, r.char_offset, r.extra_json, "
            "r.display_text, v.distance "
            "FROM vectors v JOIN records r ON r.rowid = v.rowid "
            "WHERE v.embedding MATCH ? AND k = ? "
            "ORDER BY v.distance"
        )
        rows = self.conn.execute(sql, (q, pool_k)).fetchall()
        out: list[tuple[StoredRecord, float]] = []
        for row in rows:
            rec = StoredRecord(
                kind=row[0], record_id=row[1], text=row[2],
                file_id=row[3], source_ref=row[4],
                section_path=row[5], topic=row[6],
                char_offset=row[7], extra=_json.loads(row[8] or "{}"),
                display_text=row[9] or "",
            )
            out.append((rec, float(row[10])))
            if len(out) >= k:
                break
        return out

    def filter_select(
        self,
        *,
        limit: int,
        kinds: tuple[str, ...] = (),
        neighbor_ids: set[tuple[str, str]] | None = None,
        exact_match: tuple[str, ...] = (),
        file_ids: tuple[str, ...] = (),
    ) -> list[StoredRecord]:
        """Filter-only records SELECT (no KNN, no distance). Returns
        the first ``limit`` records matching every supplied filter.

        Every supplied filter compiles into a ``WHERE`` clause and is
        enforced before the ``LIMIT``. Rows come back in records-table
        insertion order; the dispatcher salience-sorts after the
        union, so callers passing the per-turn cap as ``limit`` get a
        generous pool the salience pass can rank across (rather than
        a per-lookup ``count`` slice that would lock in
        insertion-order top-N before ranking).

        ``neighbor_ids`` is the precomputed neighbor set from
        ``neighbors_of``; an empty set short-circuits to ``[]`` (no
        neighbors known is distinct from "no neighbor filter")."""
        if neighbor_ids is not None and not neighbor_ids:
            return []
        clause, params = _records_where(
            kinds=kinds, neighbor_ids=neighbor_ids, exact_match=exact_match,
            file_ids=file_ids,
        )
        where = f" WHERE {clause}" if clause else ""
        sql = (
            "SELECT kind, record_id, text, file_id, source_ref, "
            "section_path, topic, char_offset, extra_json, display_text "
            f"FROM records{where} LIMIT ?"
        )
        params = list(params)
        params.append(int(limit))
        rows = self.conn.execute(sql, params).fetchall()
        import json as _json
        out: list[StoredRecord] = []
        for row in rows:
            out.append(StoredRecord(
                kind=row[0], record_id=row[1], text=row[2],
                file_id=row[3], source_ref=row[4],
                section_path=row[5], topic=row[6],
                char_offset=row[7], extra=_json.loads(row[8] or "{}"),
                display_text=row[9] or "",
            ))
        return out


def _records_where(
    *,
    kinds: tuple[str, ...] = (),
    neighbor_ids: set[tuple[str, str]] | None = None,
    exact_match: tuple[str, ...] = (),
    file_ids: tuple[str, ...] = (),
) -> tuple[str, list[object]]:
    """Compile the shared records-table WHERE clause for the filter
    surface — ``kinds`` → ``kind IN (?, ...)``, ``exact_match`` →
    parameterized ``LOWER(text) LIKE ?`` ORed across the list with
    ``%``/``_``/``\\`` ESCAPE-quoted, ``file_ids`` →
    ``LOWER(file_id) LIKE ?`` ORed the same way (the ``source`` filter:
    a case-insensitive substring on the per-record file identifier, so a
    user-supplied ``"meditations"`` anchors retrieval on
    ``aurelius-meditations.txt``), ``neighbor_ids`` →
    row-value ``(kind, record_id) IN ((?, ?), ...)``. AND-composed,
    all bound params, in-order. Empty input → empty clause + empty
    params (caller omits the WHERE)."""
    clauses: list[str] = []
    params: list[object] = []
    if kinds:
        clauses.append("kind IN (" + ",".join(["?"] * len(kinds)) + ")")
        params.extend(kinds)
    if exact_match:
        ors: list[str] = []
        for s in exact_match:
            ors.append("LOWER(text) LIKE ? ESCAPE '\\'")
            params.append("%" + _escape_like(s.lower()) + "%")
        clauses.append("(" + " OR ".join(ors) + ")")
    if file_ids:
        ors = []
        for s in file_ids:
            ors.append("LOWER(file_id) LIKE ? ESCAPE '\\'")
            params.append("%" + _escape_like(s.lower()) + "%")
        clauses.append("(" + " OR ".join(ors) + ")")
    if neighbor_ids is not None:
        clauses.append(
            "(kind, record_id) IN ("
            + ",".join(["(?, ?)"] * len(neighbor_ids))
            + ")"
        )
        for kind, rid in neighbor_ids:
            params.append(kind)
            params.append(rid)
    return (" AND ".join(clauses), params)


def _escape_like(s: str) -> str:
    """Backslash-escape SQL LIKE wildcards (``%`` and ``_``) and the
    escape character itself so a model-supplied substring matches
    literally inside a parameterized ``LIKE ? ESCAPE '\\'`` clause."""
    return (
        s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    )


@contextmanager
def open_store(path: Path, dim: int = EMBEDDING_DIM) -> Iterator[VectorStore]:
    """Convenience context-manager wrapper around `VectorStore.open()`."""
    store = VectorStore(path, dim=dim).open()
    try:
        yield store
    finally:
        store.close()

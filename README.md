# BaseVault

Local-first personal data pipeline. Takes heterogeneous personal files,
extracts structured facts, resolves entities + subject, detects
within-theme patterns, synthesizes cross-domain insights, and produces
actionable recommendations — all with deterministic source-ref
propagation back to the original file offsets.

The desktop shell is Tauri + React. The pipeline is Python, invoked as
a subprocess by the Rust backend.

## Run modes

| Mode  | Provider          | Visible when            | Notes |
|-------|-------------------|-------------------------|-------|
| Local | Ollama or MLX     | always                  | No user data leaves the machine. Sequential (1 worker). |
| TEE   | Tinfoil           | Tinfoil key in Settings | Trusted execution; encrypted in transit; parallel fan-out (multiple workers). |

The production binary routes user data only through TEE or LOCAL.
Vision models match the pipeline mode — images are never sent across
trust boundaries. The app ships no analytics, telemetry, or crash
reporting; the only outbound connections it ever makes are enumerated
under [Network activity](#network-activity) below.

## Trust model

BaseVault has two privacy postures: **Local** (Ollama / llama.cpp on
your machine) and **TEE** (Tinfoil-hosted enclave). Local is simpler
— your *data* never leaves your laptop (the app's only outbound
connections are the user-triggered update check and, if you've
configured a TEE model, its attestation checks; see
[Network activity](#network-activity)). TEE crosses a network boundary
to a remote machine, but uses hardware attestation + transparency logs
so you don't have to trust the operator with your data.

### Local mode

In Local mode your data stays on the machine. Inference runs in
Ollama (which we spawn or talk to over `localhost`); the pipeline
runs in the bundled Python sidecar. What you implicitly trust:

- **Your hardware + OS** — disk encryption, RAM, the kernel that
  boots Ollama and the pipeline. Same trust surface as anything else
  you run locally.
- **Ollama / llama.cpp** — the inference binary that actually loads
  the model. Open source, distributed via `ollama.com` and buildable
  from `github.com/ollama/ollama` + `github.com/ggerganov/llama.cpp`.
- **The model weights** — the GGUF files Ollama downloads. Publicly
  distributed; a corrupted model would require compromising the
  model registry, but the file hash is observable post-download.
- **The Python pipeline + its deps** — same code as TEE mode minus
  the network calls; same `requirements.txt` (so the PyPI integrity
  caveat below applies here too).

The trust surface is small because there's no remote party. The
tradeoff is throughput: Local runs one worker at a time and is much
slower than hosted inference, and quality is bounded by what fits in
your machine's RAM/VRAM.

### TEE mode

In TEE mode your data is sent to a remote enclave for inference. The
trust model is structured so that even though the data leaves your
machine, **no party along the path — including Tinfoil itself — can
read or substitute it**. The model runs inside a hardware-isolated
enclave: AMD-SEV-SNP or Intel TDX silicon enforces that the compute
stays sealed even from the cloud operator. Every call gets back a
cryptographic *attestation* — proof the enclave is running the exact
image we expected, not something that swapped your prompt out to
plaintext.

BaseVault's Settings panel has a **Verify** button per model. It runs
the embedded Tinfoil SDK's full verification and shows what the SDK
checked — not just "ok." If you click "View chain," you see this
expanded into raw evidence (hashes, signatures, transparency-log
entries) you can copy + audit yourself.

**Additional things you have to trust in TEE mode** (on top of Local
mode's baseline — these are the assumptions that come with crossing
the network boundary, and the ones to check externally if you're
being thorough):

- **AMD / Intel silicon** — the hardware attestation roots. The chip
  vouches for which code is loaded; you trust the silicon vendor's
  signing roots.
- **Tinfoil's published source** at `github.com/tinfoilsh/...` —
  that the open-source code that builds the enclave images, when
  audited, behaves as advertised.
- **GitHub identity** — that `tinfoilsh` on github.com is the real
  Tinfoil organization and not a compromised account.
- **Sigstore (Fulcio + Rekor)** — the public-good signing + append-
  only transparency log infrastructure that proves *which workflow
  built the image*. Operated by the open-source Sigstore project.
- **Linux core** — kernel, init, root filesystem of the enclave. We
  don't trust them blindly; their hashes are baked into the
  attestation chain so any change shows up.
- **GitHub Actions build toolchain** — the Ubuntu runner image,
  package mirrors (apt / cargo / pip), and compiler toolchain that
  GitHub provisions to build the enclave images. SLSA provenance
  proves *which workflow file ran*, but it doesn't prove the build
  was reproducible from public source — a compromised runner could,
  in theory, alter the output without changing the workflow. This is
  a real gap. Mitigations available today: independently rebuild from
  the same source commit and confirm the hash matches what's in the
  deployment manifest (Tinfoil doesn't ship reproducible builds yet,
  so this requires hands-on verification rather than a click).
- **Tinfoil + OpenAI Python clients (and their transitive deps)** —
  the `tinfoil` and `openai` PyPI packages that BaseVault uses to do
  the verification handshake and the inference calls. Both are
  open-source (`github.com/tinfoilsh/...` and
  `github.com/openai/openai-python`) and you can read the exact code
  that runs against your data: it's installed verbatim into the
  bundled Python at
  `~/Library/Application Support/BaseVault/.../python/lib/python3.14/site-packages/`.
  The trust gap that *would* exist here is **PyPI registry integrity**
  — the binaries came from PyPI at install time, and you'd otherwise
  be trusting that what PyPI served matches what's on GitHub.
  BaseVault closes this with full hash-pinning:
  `engine/requirements.txt` carries `--hash=sha256:...` lines
  for every wheel (generated via `pip-compile --generate-hashes`),
  and `scripts/setup-bundled-python.sh` installs them with
  hash enforcement. Pip refuses any wheel whose sha256 doesn't
  match the pinned value, so a registry-MITM or PyPI-side tampering
  attack on an installed version is detected at install time. You
  can also audit the actually-installed bytes against the public
  source yourself if you're being thorough.

**What you don't have to trust** (things the system protects against,
so you don't have to take anyone's word):

- **The network** — every quote is signed at hardware level, and the
  inference channel is end-to-end encrypted to inside the enclave.
  Nothing in transit (your ISP, Tinfoil's frontend, intermediate
  proxies, a hostile coffee-shop wifi) can read your prompt or
  alter what comes back.
- **Tinfoil as a company** — the chain replaces reputation-trust
  with cryptographic verification. Tinfoil could refuse to serve you
  (DoS), but they cannot read your prompts, substitute a different
  model behind the scenes, or quietly deploy modified enclave code
  without the attestation chain showing the change. You're trusting
  their *open source* (which any third party can audit), not their
  *promises*.
- **Tinfoil's own runtime servers** — they can't substitute a
  different model; the measurement is computed *inside* the chip and
  signed before it leaves.
- **A compromised release after the fact** — Sigstore's transparency
  log entries are append-only and globally consistent, so post-hoc
  edits show up.

**Local vs TEE at a glance**:

| Concern                       | Local              | TEE                                       |
|-------------------------------|--------------------|-------------------------------------------|
| Data leaves the machine       | No                 | Yes, but encrypted into the enclave       |
| Trust the network             | N/A                | No (e2e crypto, hardware-signed quotes)   |
| Trust the inference operator  | N/A                | No (chain replaces reputation-trust)      |
| Trust hardware silicon        | Your CPU           | AMD/Intel + your CPU                      |
| Trust the inference binary    | Ollama (local OSS) | Tinfoil enclave image (verified per call) |
| Throughput                    | 1 worker           | multiple parallel                         |
| Quality ceiling               | Local RAM/VRAM     | Hosted frontier models                    |

**The chain, end to end.** The verification is performed by the
**Tinfoil SDK** BaseVault embeds (the `tinfoil` PyPI package, source at
`github.com/tinfoilsh/tinfoil-python`) — not by code in this
repository. BaseVault triggers it, refuses inference if it fails, and
logs the evidence. What the SDK checks on each call:

1. The enclave hardware produces a cryptographic *quote* — a signed
   measurement of the loaded code. This is rooted in AMD/Intel's
   silicon CA.
2. The SDK compares the quote's measurement against the published
   deployment manifest (`tinfoil-deployment.json`) for the model on
   `github.com/tinfoilsh/confidential-<model>`. Match → the enclave
   is running what Tinfoil publicly committed to.
3. The SDK verifies the deployment's SLSA *provenance attestation* via
   Sigstore. Match → the artifact was built by the expected GitHub
   Actions workflow on Tinfoil's repo, signed at build time, recorded
   in Rekor's public transparency log.
4. The enclave's Linux toolchain (kernel, initrd, root filesystem)
   has its hashes embedded in the same quote — so the whole boot chain
   is verified, not just the model image.

If any step fails or returns an unexpected measurement, BaseVault turns
the trust bar red and refuses inference. Local mode runs offline and
has no attestation surface; the production binary carries no other mode
that crosses a trust boundary.

The full evidence (raw quote bytes, TUF metadata, Sigstore bundle,
Rekor inclusion proof) is logged to
`~/.basevault/attestations.jsonl` for every verification — you can
grep, audit, and feed it to other tools.

### Network activity

BaseVault has **no analytics, telemetry, crash reporting, or
phone-home**. Every outbound connection it makes falls into one of
these kinds, and never any other:

1. **TEE inference (when a TEE model is configured).** Your prompt is
   end-to-end encrypted into the attested enclave; a router-discovery
   request to `atc.tinfoil.sh` selects the enclave endpoint. Local-only
   runs make neither call.
2. **Attestation verification (automatic, when a TEE model is
   configured).** So the trust indicator reflects current reality
   rather than a one-time check, BaseVault re-runs the Tinfoil SDK's
   verification of the configured enclave's attestation **at startup and
   then once an hour** in the background — plus on demand when you
   re-check from Settings. Each
   verification fetches a live attestation quote from the enclave,
   certificate collateral from Intel's attestation service (PCS), and
   the deployment manifest from GitHub. This is *metadata* — which
   enclave, which manifest — and carries none of your data. If no TEE
   model is configured, none of this runs.
3. **Local inference (localhost).** Local runs talk to Ollama /
   llama.cpp over `localhost` only — never off the loopback interface.
4. **Update check (user-initiated).** The Settings panel has a
   **Check for updates** button. Clicking it fetches a small release
   manifest from `basevault-releases.s3.amazonaws.com`; installing
   downloads the signed build from the same host. This is **not
   automatic** — there is no background poll, no check on launch, and
   no update traffic unless you press the button. As with any update
   check, the request necessarily reveals your IP address and the
   time you checked to the host serving the manifest (AWS S3). It
   carries none of your data and no identifier BaseVault assigns.

With **no TEE model configured and no update check**, BaseVault makes
no outbound connection at all — Local inference stays on `localhost`,
and the pipeline's enclave-reaching imports are fenced off by a
build-time gate that fails if pipeline code so much as imports a
network library. You can confirm the whole picture with a network
monitor.

### What attestation does and does not prove

Attestation proves the **provenance** of the enclave build: that the
remote machine is running the specific, publicly-committed Tinfoil
enclave image, built by the expected workflow, recorded in a public
transparency log — cross-checkable yourself at
[visibility.tinfoil.sh](https://visibility.tinfoil.sh). It does **not**
prove that the operator is benevolent, that the model is good, or that
the published source is free of bugs. It replaces *"trust the operator's
promise"* with *"verify the operator ran the code they published"* —
a strictly narrower, checkable claim. The honest gaps in that chain
(non-reproducible builds, PyPI registry integrity, silicon vendor
roots) are spelled out explicitly in the TEE-mode trust list above
rather than glossed over.

## Pipeline

Input files flow through these stages:

1. **Ingest** — whitelist of supported formats (txt, md, pdf, docx, json,
   zip, images). Files > 40 MB are dropped. Zips are recursively
   extracted. Images are transcribed via a vision model that matches the
   pipeline mode (never cross-boundary).
2. **Split** — per-file document segmentation. Bundles (multiple
   self-contained documents in one file) are split; every sub-document
   carries `file_id` so downstream refs trace back to the original file.
3. **Extract** — per-document metadata (topics, people, events, review
   flags) + per-chunk content extraction into `ExtractedItem`s (fact,
   decision, event, emotion, signal, open_loop). Each `EvidenceSpan`
   carries `file_path`, `file_offset`, `file_length` — absolute refs
   into the original ingested file. Facts are grouped by topic on
   disk (`facts_by_topic.json`).
4. **Entities** — resolves entity mentions across facts, groups by
   `(normalized_name, entity_type)`, picks a single `subject` for the
   run (tiered resolver: LLM → alias-match → mention-count fallback,
   with a non-person-guard that rejects non-person LLM picks). Bundle
   inputs scrub `subject` to null and synthesize per-file narrator
   entities so cross-file contexts don't collapse.
5. **Patterns (within-theme)** — one LLM call per topic. Produces 2–4
   mechanistic patterns per topic. Each pattern carries `source_facts`
   (indices into the topic's fact list).
6. **Insights** — one LLM call. Produces ≤3 cross-domain and ≤2 critical
   insights across all topics. Each carries `source_patterns` — tuples
   of `(topic, local_index, weight)` back into within-theme patterns.
7. **Actions** — one LLM call. Emits actionable recommendations with a
   harm-gate filter; carries refs back into insights and patterns.

## Ref propagation

Each stage stores refs to the *previous* stage only. The full chain:

```
file content
    ↑ (file_path, file_offset, file_length) on each EvidenceSpan
ExtractedItem (fact)
    ↑ source_facts on each Pattern
Pattern
    ↑ source_patterns on each Insight
Insight
    ↑ source_insights on each Action
Action
```

Entities reference facts via `EntityRef` and do not sit on this linear
chain — they are a sibling projection of the facts layer used by
downstream stages for subject/person context.

No stage embeds refs deeper than one level. A walker traverses the DAG
on demand. LLM outputs with invalid prompt-local IDs are retained, but
the object records a `hallucinated_ref_count` so downstream code can
see how trustworthy the chain is.

## Output layout

Two roots, split along Unix conventions:

```
~/.basevault/logs/                             # app runs only (Tauri sets agent=app)
  <run-id>/                                    # flat per-run dirs
    run.log                pipeline log for this run
    config.json            static start-of-run snapshot (write-once)
    llm-calls.jsonl        append-only event log (begin/end/cycle markers)
    llm-stats.json         end-of-run rollup materialized from the jsonl
    llm-stats.txt          human-readable rollup sibling
    paused.flag            present iff Tauri-side paused (runtime marker)
    run.json               legacy; only on pre-#165 runs (read-fallback only)
    intermediate/          canonical JSON artifacts + preprocessed markdown
    stages/                per-stage on-disk markers + topic / entity files
    vault/                 only under experiment --emit-vault

~/.basevault/logs-dev/                         # everything else — scripts, smoke,
  <run-id>/                                    # tests, ad-hoc CLI (agent=experiment,
                                               # the default when BASEVAULT_AGENT is
                                               # unset). Not scanned by the GUI.

  sweeps/<sweep-id>/                           # eval namespace (run_benchmark_sweep)
    eval.json              sweep manifest (cases, modes, judge_model)
    report/judge.md        cross-case judge output
    <case>-<mode>/         one per planned run; same shape as above

  Pre-flatten corpora at <session>/eval-<ts>/<run>/ under either root are
  still readable — the Tauri walk and resume path both accept the legacy
  nested shape.
~/.basevault/config.json     app UI state (last-loaded inputs, mode)
~/.basevault/.migration-v1-done    one-shot migration marker

~/Documents/BaseVault/       user-facing vault root — flat
  <run-id>/
    0-inputs/ 1-facts/ 2-entities/ 3-patterns/ 4-insights.md 5-actions.md

~/Library/Application Support/BaseVault/.env   settings (name, Tinfoil key)
```

The 3-pane app renders runs and per-file output natively — no external
tool required. Power users can point any markdown editor (Obsidian,
Typora, etc.) at `~/Documents/BaseVault/<run-id>/` if they prefer that
flow.

Session IDs are `<ISO-8601-Z>-<suffix>[-<descriptor>]` where suffix is
`app` or `experiment`. App sessions have one session per app launch;
each "Run pipeline" click creates a fresh `eval-<ts>/<run-id>/` under
it. Experiment sweeps (from CLI) use their own session naming.

`run-id` format for app runs: `<ISO-8601-Z>-manual` (unique per click).
Used both as the log leaf-dir name and as the flat vault-dir name.

## Resume

```bash
python3 runner.py --resume-run-id <run-id>
# or, for experiment sweeps that predate run-ids:
python3 runner.py --resume-run-dir /abs/path/.logs/<session>/<eval>/<run>
```

Detects the latest checkpoint and re-runs only what's missing.
Checkpoint priority: `actions → insights → patterns → entities → extract`.
Within `extract` and `patterns`, intra-stage partials mean a hard-kill
mid-stage only loses the in-flight LLM call, not the whole stage —
completed parents / topics are loaded from disk on resume.

## App run lifecycle

The runner writes `config.json` once at run start (static fields:
mode, models, inputs, version stamps) and streams begin/end/cycle
events to `llm-calls.jsonl`. The Tauri shell derives live status +
progress (`running | paused | cancelled | failed | completed`) by
walking the jsonl + checking a `paused.flag` marker for the
runtime-paused state. Pause / cancel hard-kill the subprocess via
SIGTERM-then-SIGKILL (Python's signal handler flushes the rollup);
the Rust shell either appends a `cycle_cancelled` event to the jsonl
(cancel) or writes `paused.flag` (pause). Resume re-spawns
`runner.py --resume-run-id`, which clears the marker and emits a
fresh `cycle_start`.

One run per mode at a time; Local + TEE can run side-by-side for
latency/quality comparison.

## Dev setup

```bash
npm install
npm run tauri dev
```

The Rust backend spawns Python via the interpreter chosen by
`pipeline_dir()` / `python_bin()` in `src-tauri/src/lib.rs`.

### Editor / LSP

Point your editor's Python interpreter at the bundled sidecar:

```
src-tauri/binaries/python/bin/python3
```

This is the same interpreter the app, eval scripts, and `tauri dev` all
use. Build it once with `scripts/setup-bundled-python.sh` before
opening any pipeline file in your editor.

## Releases

Releases are distributed as macOS `.app`/`.dmg` bundles that are
recursively code-signed (including the bundled Python sidecar and
every nested `.dylib`/`.so`), Apple-notarized, and stapled. The
release build is tag-driven off `main`:

```bash
git tag v0.1.18
git push origin v0.1.18
```

`CFBundleShortVersionString` comes from the tag; `CFBundleVersion` is
the cumulative `v*` tag count. The signed artifacts are published as
GitHub Releases, and the in-app **Check for updates** button (see
[Network activity](#network-activity)) resolves against the same
signed manifest.

### Verifying a download

Every release artifact carries a GitHub-native **build-provenance
attestation** — a Sigstore-signed, transparency-log-backed record of the
exact workflow run and source commit that produced it. Verify a download
against it, pinned to this repository, with the GitHub CLI:

```bash
gh attestation verify BaseVault_<version>_aarch64.dmg \
  --repo basevault-ai/basevault
```

A pass means the bytes you have were built by this repo's release
workflow from a specific commit — not repackaged or swapped downstream.
The signed digests are also printed on each release run's summary page.

## License

Apache License 2.0 — see [`LICENSE`](LICENSE).

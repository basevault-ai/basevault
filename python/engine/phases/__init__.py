"""Production pipeline phases, migrated onto the kernel (issue #912).

One module per ``kernel.enums.PhaseName``. Each module holds a
``kernel.abstractions.Phase`` implementation that REUSES the production
prompt-build / parse logic from its stage module verbatim (e.g. the
extraction phase imports ``content_extractor._build_prompt`` /
``_parse_items``); the only thing the migration changes is the
retry / halve / cache PLUMBING, which the kernel owns
(``BoundExecutionEnv.run`` + ``RetryPolicy`` + ``CachingHook``).

The model + provider wiring lives in ``phases.model_specs`` (kernel
``ModelSpec`` wrappers around the production ``llm.ModelSpec`` registry,
plus the local ``ollama_provider`` / ``mlx_provider``). Tinfoil stays in
``kernel/``; the eval-only non-attested provider stays under
``testing/``.
"""

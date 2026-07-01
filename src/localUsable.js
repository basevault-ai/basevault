// Whether the "Local" inference mode is selectable. The configured local
// backend must be actually usable on this machine — not merely un-skipped.
// Single source of truth for both the mode-picker's disabled state and the
// startup mode-fallback, so the two can't drift.
//
// MLX (bundled, default backend): a model must be downloaded AND the
// running OS must meet the bundled dylib's floor (os_supported, from the
// local_model_status command). Below that floor `import mlx.core` crashes,
// so an enabled-but-unusable Local is exactly the dead-end being prevented.
// Ollama (opt-in): gated on its own detect-only verification, as before.
// Explicitly skipped local setup stays off regardless of either.
export function localModeUsable({ setupMode, backend, modelDownloaded, osSupported }) {
  if (setupMode === "skipped") return false;
  if (backend === "ollama") return setupMode === "verified";
  return !!modelDownloaded && !!osSupported;
}

// Whether the chosen local backend is set up this session: a bundled-MLX
// model needs only to be downloaded (the download IS the setup — there is
// no separate verify step), while Ollama is opt-in and gated on its
// detect-only verify. Shared by the Wizard and Settings so the two
// onboarding surfaces can't disagree on what "local is ready" means.
export function localBackendReady({ backend, mlxDownloaded, ollamaVerified }) {
  return backend === "ollama" ? !!ollamaVerified : !!mlxDownloaded;
}

import { describe, it, expect } from "vitest";
import { localModeUsable, localBackendReady } from "./localUsable";

const mlx = (over) => ({
  setupMode: null,
  backend: "mlx",
  modelDownloaded: false,
  osSupported: true,
  ...over,
});

describe("localModeUsable — MLX backend", () => {
  it("is unavailable by default on a fresh install (no model downloaded)", () => {
    // #731: Local must default OFF, not be selectable into the MLX crash.
    expect(localModeUsable(mlx())).toBe(false);
  });

  it("becomes available once the model is downloaded on a supported OS", () => {
    expect(localModeUsable(mlx({ modelDownloaded: true }))).toBe(true);
  });

  it("stays disabled until the model is actually downloaded", () => {
    expect(localModeUsable(mlx({ modelDownloaded: false }))).toBe(false);
  });

  it("stays disabled on an OS the bundled MLX can't load on", () => {
    // #732 coordination: model present but macOS below the floor → the
    // import would crash, so Local must not be selectable.
    expect(
      localModeUsable(mlx({ modelDownloaded: true, osSupported: false })),
    ).toBe(false);
  });

  it("respects an explicit skip even with a usable model present", () => {
    expect(
      localModeUsable(mlx({ setupMode: "skipped", modelDownloaded: true })),
    ).toBe(false);
  });
});

describe("localModeUsable — Ollama backend", () => {
  const ollama = (over) => ({
    setupMode: "verified",
    backend: "ollama",
    modelDownloaded: false,
    osSupported: false,
    ...over,
  });

  it("is available when verified, independent of the MLX OS floor", () => {
    expect(localModeUsable(ollama())).toBe(true);
  });

  it("is unavailable when not yet verified", () => {
    expect(localModeUsable(ollama({ setupMode: null }))).toBe(false);
  });

  it("respects an explicit skip", () => {
    expect(localModeUsable(ollama({ setupMode: "skipped" }))).toBe(false);
  });
});

describe("localBackendReady", () => {
  it("treats a downloaded MLX model as ready — the download is the setup", () => {
    // The bug this fixes: MLX has no separate verify step, so a downloaded
    // model must count as ready (Settings showed it stuck "pending").
    expect(localBackendReady({ backend: "mlx", mlxDownloaded: true })).toBe(true);
  });

  it("is not ready when the MLX model isn't downloaded", () => {
    expect(localBackendReady({ backend: "mlx", mlxDownloaded: false })).toBe(false);
  });

  it("ignores the Ollama verify flag for the MLX backend", () => {
    expect(
      localBackendReady({ backend: "mlx", mlxDownloaded: false, ollamaVerified: true }),
    ).toBe(false);
  });

  it("gates Ollama on its verify, not on a downloaded MLX model", () => {
    expect(localBackendReady({ backend: "ollama", ollamaVerified: true })).toBe(true);
    expect(
      localBackendReady({ backend: "ollama", ollamaVerified: false, mlxDownloaded: true }),
    ).toBe(false);
  });
});

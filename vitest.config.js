import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.js"],
    include: ["src/**/*.{test,spec}.{js,jsx}"],
    // Pin a non-UTC timezone so the local-time rendering tests are
    // deterministic on any machine/CI. Without this a UTC runner would
    // silently mask the very UTC-vs-local skew #984 fixed. America/
    // New_York (UTC−4/−5) is the reporter's zone in that issue.
    env: { TZ: "America/New_York" },
  },
});

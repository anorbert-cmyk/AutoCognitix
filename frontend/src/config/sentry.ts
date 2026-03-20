import * as Sentry from "@sentry/react";

export function initSentry(): void {
  const dsn = import.meta.env.VITE_SENTRY_DSN;
  if (!dsn) {
    console.warn("Sentry DSN not configured - error tracking disabled");
    return;
  }

  Sentry.init({
    dsn,
    environment: import.meta.env.MODE,
    enabled: import.meta.env.PROD,
    tracesSampleRate: 0.1,
    replaysSessionSampleRate: 0,
    replaysOnErrorSampleRate: 1.0,
  });
}

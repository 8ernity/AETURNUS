/**
 * Shared browser-side credential helper.
 *
 * The backend intentionally rejects unauthenticated mutations.  The static
 * demo has no login screen, so an operator can place an API key in
 * `localStorage` under `aeturnus_api_key` (or inject it at build time through
 * VITE_AETURNUS_API_KEY).  Read-only endpoints continue to work without it.
 */
export function getApiKey(): string {
  try {
    return (
      window.localStorage.getItem("aeturnus_api_key") ||
      (import.meta.env.VITE_AETURNUS_API_KEY as string | undefined) ||
      ""
    );
  } catch {
    return "";
  }
}

export function authHeaders(extra: Record<string, string> = {}): Record<string, string> {
  const headers: Record<string, string> = { ...extra };
  const key = getApiKey();
  if (key) headers["X-API-Key"] = key;
  return headers;
}

export function websocketUrl(path: string): string {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const key = getApiKey();
  const suffix = key ? `?token=${encodeURIComponent(key)}` : "";
  return `${protocol}//${window.location.host}${path}${suffix}`;
}

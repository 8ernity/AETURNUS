/**
 * Shared browser-side credential helper.
 *
 * The backend intentionally rejects unauthenticated mutations and WebSocket
 * connections.  The static UI therefore lets an operator store the key under
 * `localStorage["aeturnus_api_key"]`; deployments may also inject a build-time
 * `VITE_AETURNUS_API_KEY` for private demos.  Read-only endpoints continue to
 * work without it.
 */
const STORAGE_KEY = "aeturnus_api_key";

export function getApiKey(): string {
  try {
    return (
      window.localStorage.getItem(STORAGE_KEY) ||
      (import.meta.env.VITE_AETURNUS_API_KEY as string | undefined) ||
      ""
    );
  } catch {
    return "";
  }
}

export function hasApiKey(): boolean {
  return Boolean(getApiKey());
}

export function saveApiKey(key: string): void {
  try {
    const trimmed = key.trim();
    if (trimmed) {
      window.localStorage.setItem(STORAGE_KEY, trimmed);
    } else {
      window.localStorage.removeItem(STORAGE_KEY);
    }
  } catch {
    // localStorage can be unavailable in hardened browser contexts.
  }
}

export function clearApiKey(): void {
  try {
    window.localStorage.removeItem(STORAGE_KEY);
  } catch {
    // Ignore storage failures.
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

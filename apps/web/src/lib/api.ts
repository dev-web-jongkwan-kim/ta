const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:7101";

export async function fetchJSON(path: string) {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`API error ${res.status}`);
  }
  return res.json();
}

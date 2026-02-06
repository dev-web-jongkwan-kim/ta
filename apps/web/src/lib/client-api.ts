// 클라이언트(브라우저)에서 사용하는 API 헬퍼
// Docker 내부 URL이 아닌 localhost 사용

export function getClientApiBase(): string {
  // 브라우저에서는 항상 localhost 사용
  if (typeof window !== "undefined") {
    return "http://localhost:7101";
  }
  // 서버사이드에서는 Docker 내부 URL 사용
  return process.env.NEXT_PUBLIC_API_BASE || "http://localhost:7101";
}

export async function clientFetch(path: string) {
  const apiBase = getClientApiBase();
  const res = await fetch(`${apiBase}${path}`);
  if (!res.ok) {
    throw new Error(`API error ${res.status}`);
  }
  return res.json();
}

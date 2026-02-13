export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const BACKEND_URL =
  process.env.BACKEND_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export async function POST(req: Request) {
  const form = await req.formData();

  const upstream = await fetch(`${BACKEND_URL}/swap/single`, {
    method: "POST",
    body: form,
    // no headers needed; fetch will set multipart boundary automatically
  });

  const buf = await upstream.arrayBuffer();

  return new Response(buf, {
    status: upstream.status,
    headers: {
      "content-type": upstream.headers.get("content-type") || "image/png",
      "cache-control": "no-store",
    },
  });
}

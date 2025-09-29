import { NextRequest, NextResponse } from "next/server";

// Ensure Node runtime (not edge) for multipart/FormData proxying
export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const modalUrl = process.env.MODAL_URL;
    if (!modalUrl) {
      return NextResponse.json({ error: "MODAL_URL not set" }, { status: 500 });
    }

    // Forward incoming multipart/form-data to Modal
    const form = await req.formData();

    const resp = await fetch(`${modalUrl}/swap`, {
      method: "POST",
      body: form,
      // FastAPI reads the multipart boundary automatically
    });

    if (!resp.ok) {
      // Try to extract JSON error from backend
      let detail: any = null;
      try {
        detail = await resp.json();
      } catch {
        // Ignore
      }
      return NextResponse.json(
        { error: detail?.detail ?? `Modal error: ${resp.status}` },
        { status: resp.status }
      );
    }

    const blob = await resp.blob();
    return new NextResponse(blob, {
      headers: { "Content-Type": "image/png" },
      status: 200,
    });
  } catch (e: any) {
    return NextResponse.json(
      { error: e?.message ?? "Proxy failed" },
      { status: 500 }
    );
  }
}

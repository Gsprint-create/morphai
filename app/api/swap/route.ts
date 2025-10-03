import { NextRequest, NextResponse } from "next/server";

export const runtime = "nodejs";

export async function POST(req: NextRequest) {
  try {
    const modalUrl = process.env.MODAL_URL;

    if (!modalUrl) {
      // ✅ loud, structured logs in Vercel
      console.error(
        JSON.stringify({
          level: "error",
          where: "api/swap",
          msg: "Missing MODAL_URL env",
          hint: "Set MODAL_URL in Vercel Project → Settings → Environment Variables and redeploy.",
        })
      );
      return NextResponse.json(
        {
          error: "Server missing MODAL_URL. Set it in environment variables and redeploy.",
          hint: "Vercel → Project → Settings → Environment Variables → MODAL_URL",
        },
        { status: 500 }
      );
    }

    const form = await req.formData(); // keeps original multipart boundary
    const resp = await fetch(`${modalUrl}/swap`, {
      method: "POST",
      body: form,
    });

    if (!resp.ok) {
      let detail: any = null;
      try {
        detail = await resp.json();
      } catch {
        /* ignore */
      }
      console.error(
        JSON.stringify({
          level: "error",
          where: "api/swap",
          msg: "Modal backend returned non-200",
          status: resp.status,
          detail: detail?.detail ?? null,
        })
      );
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
    console.error(
      JSON.stringify({
        level: "error",
        where: "api/swap",
        msg: "Unhandled exception",
        error: e?.message ?? String(e),
      })
    );
    return NextResponse.json({ error: e?.message ?? "Proxy failed" }, { status: 500 });
  }
}

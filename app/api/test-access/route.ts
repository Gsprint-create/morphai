import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: Request) {
  const { password } = (await req.json().catch(() => ({}))) as { password?: string };

  const expected = process.env.MORPHAI_TEST_PASSWORD;
  if (!expected) {
    return NextResponse.json({ detail: "Server not configured" }, { status: 500 });
  }

  if (!password || password !== expected) {
    return NextResponse.json({ detail: "Invalid password" }, { status: 401 });
  }

  const res = NextResponse.json({ ok: true });

  // Cookie good for 30 days
  res.cookies.set("morphai_test", "1", {
    httpOnly: true,
    secure: true,
    sameSite: "lax",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });

  return res;
}

import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

export function middleware(req: NextRequest) {
  const { pathname } = req.nextUrl;

  // Protect everything under /tools
  if (pathname.startsWith("/tools")) {
    const ok = req.cookies.get("morphai_test")?.value === "1";
    if (!ok) {
      const url = req.nextUrl.clone();
      url.pathname = "/coming-soon";
      url.searchParams.set("next", pathname);
      return NextResponse.redirect(url);
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/tools/:path*"],
};

"use client";

import { useState } from "react";

export default function TestAccessPage() {
  const [password, setPassword] = useState("");
  const [msg, setMsg] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  async function unlock() {
    setMsg(null);
    setLoading(true);
    try {
      const res = await fetch("/api/test-access", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ password }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        setMsg(data?.detail || "Wrong password");
        return;
      }

      const next = new URLSearchParams(window.location.search).get("next") || "/tools";
      window.location.href = next;
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen bg-black text-white flex items-center justify-center px-6">
      <div className="w-full max-w-md rounded-2xl border border-white/10 bg-white/5 p-6">
        <h1 className="text-xl font-semibold">Private access</h1>
        <p className="mt-1 text-sm text-white/60">Enter your test password to unlock tools.</p>

        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          className="mt-4 w-full rounded-xl border border-white/10 bg-black/40 px-4 py-3 text-sm text-white outline-none"
        />

        <button
          onClick={unlock}
          disabled={!password || loading}
          className="mt-4 w-full rounded-xl bg-blue-600 px-4 py-3 text-sm font-semibold text-white hover:bg-blue-500 disabled:opacity-50"
        >
          {loading ? "Unlocking…" : "Unlock"}
        </button>

        {msg && <div className="mt-3 text-sm text-red-300">{msg}</div>}

        <a href="/" className="mt-4 inline-block text-xs text-white/50 hover:text-white/70">
          ← Back to home
        </a>
      </div>
    </main>
  );
}

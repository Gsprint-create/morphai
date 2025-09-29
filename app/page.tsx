"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import "./globals.css";

type FileState = {
  file: File | null;
  url: string;
};

export default function Home() {
  const [source, setSource] = useState<FileState>({ file: null, url: "" });
  const [target, setTarget] = useState<FileState>({ file: null, url: "" });
  const [resultUrl, setResultUrl] = useState<string>("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string>("");
  const [elapsed, setElapsed] = useState<number>(0);

  const timerRef = useRef<NodeJS.Timeout | null>(null);
  useEffect(() => {
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
      if (source.url) URL.revokeObjectURL(source.url);
      if (target.url) URL.revokeObjectURL(target.url);
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    };
  }, [source.url, target.url, resultUrl]);

  const canSubmit = useMemo(() => !!source.file && !!target.file && !busy, [source, target, busy]);

  const onPick = (setter: (s: FileState) => void) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] || null;
    if (!f) return setter({ file: null, url: "" });
    const url = URL.createObjectURL(f);
    setter({ file: f, url });
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setResultUrl("");
    if (!source.file || !target.file) {
      setError("Please choose both Source and Target images.");
      return;
    }

    try {
      setBusy(true);
      setElapsed(0);
      timerRef.current = setInterval(() => setElapsed((s) => s + 1), 1000);

      const form = new FormData();
      form.append("source", source.file);
      form.append("target", target.file);

      const res = await fetch("/api/swap", { method: "POST", body: form });
      if (!res.ok) {
        const j = await res.json().catch(() => ({} as any));
        throw new Error(j?.error || `HTTP ${res.status}`);
      }
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setResultUrl(url);
    } catch (err: any) {
      setError(err?.message ?? "Swap failed.");
    } finally {
      setBusy(false);
      if (timerRef.current) clearInterval(timerRef.current);
    }
  };

  const clearAll = () => {
    if (source.url) URL.revokeObjectURL(source.url);
    if (target.url) URL.revokeObjectURL(target.url);
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setSource({ file: null, url: "" });
    setTarget({ file: null, url: "" });
    setResultUrl("");
    setError("");
    setElapsed(0);
  };

  return (
    <main>
      <h1>Face Swap (Modal + Vercel)</h1>
      <p className="muted">Upload a Source face and a Target photo, then click <b>Swap Faces</b>.</p>

      <form className="grid grid-3">
        {/* Source */}
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="badge">Source (face to use)</span>
            {source.file && <span className="muted" style={{ fontSize: 12 }}>{Math.round(source.file.size / 1024)} KB</span>}
          </div>
          <div style={{ marginTop: 10, marginBottom: 10 }}>
            <input className="input" type="file" accept="image/*" onChange={onPick(setSource)} />
          </div>
          {source.url ? <img className="preview" src={source.url} alt="source" /> : <div className="preview" />}
        </div>

        {/* Target */}
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="badge">Target (photo to modify)</span>
            {target.file && <span className="muted" style={{ fontSize: 12 }}>{Math.round(target.file.size / 1024)} KB</span>}
          </div>
          <div style={{ marginTop: 10, marginBottom: 10 }}>
            <input className="input" type="file" accept="image/*" onChange={onPick(setTarget)} />
          </div>
          {target.url ? <img className="preview" src={target.url} alt="target" /> : <div className="preview" />}
        </div>

        {/* Result */}
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="badge">Result</span>
            {busy ? <span className="muted" style={{ fontSize: 12 }}>Processing… {elapsed}s</span> : null}
          </div>
          <div className="row" style={{ marginTop: 10, marginBottom: 10 }}>
            <button className="btn" onClick={onSubmit} disabled={!canSubmit}>
              {busy ? "Swapping…" : "Swap Faces"}
            </button>
            <button className="btn" type="button" onClick={clearAll} disabled={busy}>
              Clear
            </button>
          </div>

          {error && <div className="error">{error}</div>}
          {resultUrl ? (
            <>
              <img className="preview" src={resultUrl} alt="result" />
              <div className="row" style={{ marginTop: 10 }}>
                <a className="link" href={resultUrl} download="swap.png">Download PNG</a>
              </div>
            </>
          ) : (
            <div className="preview" />
          )}
        </div>
      </form>

      <div className="footer">
        Backend: <code>Modal FastAPI</code> at server proxy <code>/api/swap</code>. Make sure <code>MODAL_URL</code> is set in your env.
      </div>
    </main>
  );
}

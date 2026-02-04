"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Picked = {
  id: string;
  file: File;
  url: string;
  name: string;
};

type TargetFace = {
  faceId: string; // numeric index as string: "0","1","2"...
  label: string;
};

type Focus = { x: number; y: number; ok: boolean; reason?: string };

function uid(prefix = "id") {
  return `${prefix}-${Math.random().toString(16).slice(2)}-${Date.now()}`;
}

function formatBytes(bytes: number) {
  if (!bytes) return "0 B";
  const k = 1024;
  const sizes = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function makePicked(file: File, prefix: string): Picked {
  return {
    id: uid(prefix),
    file,
    url: URL.createObjectURL(file),
    name: file.name,
  };
}

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
}

// Uses Chrome Shape Detection API (FaceDetector). If unavailable, returns ok:false.
async function detectFaceFocusFromUrl(url: string): Promise<Focus> {
  try {
    const FaceDetectorCtor = (globalThis as any).FaceDetector;
    if (!FaceDetectorCtor) {
      return { x: 50, y: 30, ok: false, reason: "FaceDetector not supported" };
    }

    const detector = new FaceDetectorCtor({ fastMode: true, maxDetectedFaces: 1 });

    const img = new Image();
    img.src = url;
    await img.decode();

    const faces = await detector.detect(img);
    if (!faces || faces.length === 0) {
      return { x: 50, y: 30, ok: false, reason: "No face detected" };
    }

    const box = faces[0].boundingBox as DOMRectReadOnly;
    const cx = box.x + box.width / 2;
    const cy = box.y + box.height / 2;

    const xPct = clamp((cx / img.naturalWidth) * 100, 0, 100);
    const yPct = clamp((cy / img.naturalHeight) * 100, 0, 100);

    return { x: xPct, y: yPct, ok: true };
  } catch (e: any) {
    return { x: 50, y: 30, ok: false, reason: String(e?.message || e) };
  }
}

export default function FaceSwapPage() {
  // Mode
  const [multiMode, setMultiMode] = useState(false);

  // Auto face focus
  const [autoFaceFocus, setAutoFaceFocus] = useState(true);

  // Source + Target (single mode)
  const [singleSource, setSingleSource] = useState<Picked | null>(null);
  const [target, setTarget] = useState<Picked | null>(null);

  // Multi mode sources
  const [sources, setSources] = useState<Picked[]>([]);

  // Detected faces from backend
  const [faces, setFaces] = useState<TargetFace[]>([]);
  const [assignments, setAssignments] = useState<Record<string, string>>({});

  // DnD UI state
  const [dragSourceId, setDragSourceId] = useState<string | null>(null);
  const [dragOverFaceId, setDragOverFaceId] = useState<string | null>(null);

  // Face focus states
  const [targetFocus, setTargetFocus] = useState<Focus>({ x: 50, y: 30, ok: false });
  const [sourceFocus, setSourceFocus] = useState<Focus>({ x: 50, y: 30, ok: false });
  const [manualTargetFocus, setManualTargetFocus] = useState({ x: 50, y: 30 });
  const [manualSourceFocus, setManualSourceFocus] = useState({ x: 50, y: 30 });

  // Result
  const [isProcessing, setIsProcessing] = useState(false);
  const [resultUrl, setResultUrl] = useState<string | null>(null);

  // Progress + cancel
  const abortRef = useRef<AbortController | null>(null);
  const [progress, setProgress] = useState(0);
  const [stage, setStage] = useState<
    "idle" | "upload" | "swap" | "finalize" | "done" | "error" | "canceled"
  >("idle");

  // ---------- cleanup helpers ----------
  function revoke(p: Picked | null) {
    if (p?.url) URL.revokeObjectURL(p.url);
  }

  function revokeMany(list: Picked[]) {
    for (const p of list) if (p.url) URL.revokeObjectURL(p.url);
  }

  function clearResult() {
    if (resultUrl) URL.revokeObjectURL(resultUrl);
    setResultUrl(null);
  }

  function resetDetectedFaces() {
    setFaces([]);
    setAssignments({});
    setDragOverFaceId(null);
  }

  function cancelInFlight() {
    abortRef.current?.abort();
    abortRef.current = null;
  }

  useEffect(() => {
    return () => {
      cancelInFlight();
      revoke(target);
      revoke(singleSource);
      revokeMany(sources);
      if (resultUrl) URL.revokeObjectURL(resultUrl);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ---------- auto face focus effects ----------
  useEffect(() => {
    let cancelled = false;
    async function run() {
      if (!autoFaceFocus || !target?.url) return;
      const f = await detectFaceFocusFromUrl(target.url);
      if (!cancelled) {
        setTargetFocus(f);
        if (!f.ok) setManualTargetFocus({ x: 50, y: 30 });
      }
    }
    run();
    return () => {
      cancelled = true;
    };
  }, [autoFaceFocus, target?.url]);

  useEffect(() => {
    let cancelled = false;
    async function run() {
      if (!autoFaceFocus || !singleSource?.url) return;
      const f = await detectFaceFocusFromUrl(singleSource.url);
      if (!cancelled) {
        setSourceFocus(f);
        if (!f.ok) setManualSourceFocus({ x: 50, y: 30 });
      }
    }
    run();
    return () => {
      cancelled = true;
    };
  }, [autoFaceFocus, singleSource?.url]);

  // ---------- pickers ----------
  function pickSingle(setter: (p: Picked | null) => void, prefix: string) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = () => {
      const file = input.files?.[0];
      if (!file) return;
      setter(makePicked(file, prefix));
    };
    input.click();
  }

  function pickMany(add: (items: Picked[]) => void, prefix: string) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.multiple = true;
    input.onchange = () => {
      const files = Array.from(input.files || []);
      if (!files.length) return;
      add(files.map((f) => makePicked(f, prefix)));
    };
    input.click();
  }

  // ---------- computed ----------
  const hasDetectedFaces = faces.length > 0;
  const hasSources = sources.length > 0;

  const assignedCount = useMemo(() => {
    return Object.values(assignments).filter((v) => v && v !== "none").length;
  }, [assignments]);

  const canSwapSingle = !!target && !!singleSource && !isProcessing;
  const canDetect = !!target && !isProcessing;

  const canSwapMulti =
    !!target && hasDetectedFaces && hasSources && assignedCount > 0 && !isProcessing;

  const sourceById = useMemo(() => {
    const m = new Map<string, Picked>();
    for (const s of sources) m.set(s.id, s);
    return m;
  }, [sources]);

  // ---------- progress helpers ----------
  function stageLabel(s: typeof stage) {
    switch (s) {
      case "upload":
        return "Uploading images…";
      case "swap":
        return multiMode ? "Swapping multiple faces…" : "Swapping face…";
      case "finalize":
        return "Finalizing result…";
      case "done":
        return "Done!";
      case "canceled":
        return "Canceled.";
      case "error":
        return "Something went wrong.";
      default:
        return "";
    }
  }

  // Smooth “fake progress” that climbs to 90% while waiting on backend.
  function startProgressPump() {
    setProgress(5);
    setStage("upload");

    const started = Date.now();
    const id = window.setInterval(() => {
      setProgress((p) => {
        if (Date.now() - started > 1200) setStage("swap");
        const next = p + Math.max(0.2, (90 - p) * 0.06);
        return Math.min(90, Number(next.toFixed(1)));
      });
    }, 120);

    return () => window.clearInterval(id);
  }

  function stopProgressDone() {
    setStage("finalize");
    setProgress(95);
    window.setTimeout(() => {
      setProgress(100);
      setStage("done");
      window.setTimeout(() => {
        setStage("idle");
        setProgress(0);
      }, 900);
    }, 250);
  }

  function stopProgressError(kind: "error" | "canceled") {
    setStage(kind);
    window.setTimeout(() => {
      setStage("idle");
      setProgress(0);
    }, 1200);
  }

  // ---------- REAL backend actions ----------
  async function detectFacesReal() {
    if (!target || isProcessing) return;

    setIsProcessing(true);
    clearResult();

    // cancel any in-flight request
    cancelInFlight();
    const controller = new AbortController();
    abortRef.current = controller;

    const stopPump = startProgressPump();

    try {
      const fd = new FormData();
      fd.append("target", target.file);

      const res = await fetch("http://127.0.0.1:8000/swap/single", {
  method: "POST",
  body: fd,
});


      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        alert(data?.detail || "Face detection failed");
        stopPump();
        stopProgressError("error");
        return;
      }

      const detected: TargetFace[] = (data.faces || []).map((f: any) => ({
        faceId: String(f.index),
        label: `Face #${Number(f.index) + 1}`,
      }));

      setFaces(detected);

      const init: Record<string, string> = {};
      for (const f of detected) init[f.faceId] = "none";
      setAssignments(init);

      stopPump();
      stopProgressDone();
    } catch (err: any) {
      stopPump();
      if (err?.name === "AbortError") stopProgressError("canceled");
      else {
        alert(err?.message || "Face detection failed");
        stopProgressError("error");
      }
    } finally {
      setIsProcessing(false);
      abortRef.current = null;
    }
  }

async function handleSingleSwap() {
  if (isProcessing) return;

  // Capture stable references (so TS knows they're not null below)
  const src = singleSource;
  const tgt = target;

  if (!src || !tgt) {
    alert("Please select both Source and Target images.");
    return;
  }

  setIsProcessing(true);
  clearResult();
  setProgress(1);
  setStage("upload");

  cancelInFlight();
  const controller = new AbortController();
  abortRef.current = controller;

  try {
    const fd = new FormData();
    fd.append("source", src.file);  // source first
    fd.append("target", tgt.file);  // target second

    const res = await fetch(`${process.env.NEXT_PUBLIC_BACKEND_URL || "http://127.0.0.1:8000"}/swap/single`, {
      method: "POST",
      body: fd,
      signal: controller.signal,
    });

    if (!res.ok) {
      const txt = await res.text();
      alert(txt || "Swap failed");
      stopProgressError("error");
      return;
    }

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    setResultUrl(url);

    stopProgressDone();
  } catch (err: any) {
    if (err?.name === "AbortError") stopProgressError("canceled");
    else {
      alert(err?.message || "Swap failed");
      stopProgressError("error");
    }
  } finally {
    setIsProcessing(false);
    abortRef.current = null;
  }
}




  async function handleMultiSwap() {
    if (!target || isProcessing) return;

    const sourceIndexById = new Map<string, number>();
    sources.forEach((s, i) => sourceIndexById.set(s.id, i));

    const map: Record<string, number> = {};
    for (const [faceId, srcId] of Object.entries(assignments)) {
      if (!srcId || srcId === "none") continue;
      const idx = sourceIndexById.get(srcId);
      if (idx === undefined) continue;
      map[faceId] = idx;
    }

    if (Object.keys(map).length === 0) {
      alert("Assign at least one target face to a source first.");
      return;
    }

    setIsProcessing(true);
    clearResult();

    cancelInFlight();
    setIsProcessing(false);
    stopProgressError("canceled");


    const stopPump = startProgressPump();

    try {
      const fd = new FormData();
fd.append("source", src.file);
fd.append("target", tgt.file);

const res = await fetch("/api/swap/single", {
  method: "POST",
  body: fd,
  signal: controller.signal,
});


if (!res.ok) {
  const txt = await res.text();
  alert(txt || "Swap failed");
  return;
}

const blob = await res.blob();
setResultUrl(URL.createObjectURL(blob));

      setResultUrl(url);

      stopProgressDone();
    } catch (err: any) {
      stopPump();
      if (err?.name === "AbortError") stopProgressError("canceled");
      else {
        alert(err?.message || "Multi swap failed");
        stopProgressError("error");
      }
    } finally {
      setIsProcessing(false);
      abortRef.current = null;
    }
  }

  function autoAssign() {
    if (!faces.length || !sources.length) return;
    const next: Record<string, string> = { ...assignments };
    faces.forEach((f, idx) => {
      next[f.faceId] = sources[idx]?.id || "none";
    });
    setAssignments(next);
    clearResult();
  }

  function removeSource(sourceId: string) {
    setSources((prev) => {
      const found = prev.find((s) => s.id === sourceId);
      if (found?.url) URL.revokeObjectURL(found.url);
      return prev.filter((s) => s.id !== sourceId);
    });

    setAssignments((prev) => {
      const next = { ...prev };
      for (const k of Object.keys(next)) {
        if (next[k] === sourceId) next[k] = "none";
      }
      return next;
    });

    if (dragSourceId === sourceId) setDragSourceId(null);
    clearResult();
  }

  function onDragStartSource(sourceId: string) {
    setDragSourceId(sourceId);
  }

  function assignFace(faceId: string, sourceId: string) {
    setAssignments((prev) => ({ ...prev, [faceId]: sourceId }));
    clearResult();
  }

  function onDropOnFace(faceId: string) {
    if (!dragSourceId) return;
    assignFace(faceId, dragSourceId);
    setDragOverFaceId(null);
    setDragSourceId(null);
  }

  function toggleMode() {
    setMultiMode((prev) => {
      const next = !prev;
      clearResult();
      resetDetectedFaces();
      return next;
    });
  }

  function focusPosition(focus: Focus, manual: { x: number; y: number }) {
    const x = focus.ok ? focus.x : manual.x;
    const y = focus.ok ? focus.y : manual.y;
    return `${x}% ${y}%`;
  }

  return (
    <main className="min-h-screen bg-black text-white px-6 py-10">
      <div className="mx-auto max-w-6xl">
        <header className="text-center">
          <h1 className="text-3xl font-semibold">FaceSwap</h1>
          <p className="mt-2 text-white/70">Single swap by default. Multi-Face mode is optional.</p>

          <div className="mt-6 flex items-center justify-center gap-3">
            <span className="text-sm text-white/70">Single</span>
            <button
              onClick={toggleMode}
              className={`relative inline-flex h-6 w-11 items-center rounded-full transition ${
                multiMode ? "bg-blue-500" : "bg-white/20"
              }`}
              aria-label="Toggle multi-face mode"
            >
              <span
                className={`inline-block h-4 w-4 transform rounded-full bg-white transition ${
                  multiMode ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
            <span className="text-sm text-white/70">Multi-Face</span>
          </div>

          <div className="mt-3 flex items-center justify-center gap-3">
            <span className="text-xs text-white/60">Auto zoom to face</span>
            <button
              onClick={() => setAutoFaceFocus((v) => !v)}
              className={`relative inline-flex h-5 w-10 items-center rounded-full transition ${
                autoFaceFocus ? "bg-emerald-500" : "bg-white/20"
              }`}
              aria-label="Toggle auto face focus"
            >
              <span
                className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition ${
                  autoFaceFocus ? "translate-x-6" : "translate-x-1"
                }`}
              />
            </button>
          </div>

          <p className="mt-2 text-center text-xs text-white/50">
            Multi-Face tip: drag a source card onto a face row to assign it.
          </p>
        </header>

        {/* Source first, Target second */}
        <div className="mt-10 grid gap-6 md:grid-cols-2">
          {/* Source */}
          <section className="rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold">Source Face</h2>
              {singleSource ? (
                <span className="text-xs text-emerald-300">Selected</span>
              ) : (
                <span className="text-xs text-white/50">Required</span>
              )}
            </div>

            <button
              onClick={() =>
                pickSingle((p) => {
                  revoke(singleSource);
                  setSingleSource(p);
                  clearResult();
                  setSourceFocus({ x: 50, y: 30, ok: false });
                  setManualSourceFocus({ x: 50, y: 30 });
                }, "src")
              }
              className="mt-3 w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-left text-sm hover:bg-white/10"
            >
              {singleSource ? "Change source face" : "Choose source face"}
              {singleSource && (
                <div className="mt-1 text-xs text-emerald-300">Selected: {singleSource.name}</div>
              )}
            </button>

            {singleSource && (
              <div className="mt-3 text-xs text-white/50">
                {singleSource.file.type || "image"} • {formatBytes(singleSource.file.size)}
              </div>
            )}

            {singleSource?.url ? (
              <div className="mt-4 overflow-hidden rounded-xl border border-white/10 bg-black/30">
                <img
                  src={singleSource.url}
                  alt="Source preview"
                  className="h-56 w-full bg-black"
                  style={{
                    objectFit: autoFaceFocus ? "cover" : "contain",
                    objectPosition: autoFaceFocus
                      ? focusPosition(sourceFocus, manualSourceFocus)
                      : "50% 50%",
                  }}
                />
              </div>
            ) : (
              <div className="mt-4 flex h-56 items-center justify-center rounded-xl border border-white/10 bg-black/30 text-sm text-white/40">
                No source yet
              </div>
            )}

            {autoFaceFocus && singleSource?.url && !sourceFocus.ok && (
              <div className="mt-3 space-y-2">
                <div className="text-xs text-white/50">
                  Auto focus fallback ({sourceFocus.reason || "unknown"}) — adjust:
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-white/50 w-10">X</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={manualSourceFocus.x}
                    onChange={(e) =>
                      setManualSourceFocus((p) => ({ ...p, x: Number(e.target.value) }))
                    }
                    className="w-full"
                  />
                  <span className="text-xs text-white/50 w-10 text-right">{manualSourceFocus.x}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-white/50 w-10">Y</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={manualSourceFocus.y}
                    onChange={(e) =>
                      setManualSourceFocus((p) => ({ ...p, y: Number(e.target.value) }))
                    }
                    className="w-full"
                  />
                  <span className="text-xs text-white/50 w-10 text-right">{manualSourceFocus.y}</span>
                </div>
              </div>
            )}

            {singleSource && (
              <button
                onClick={() => {
                  revoke(singleSource);
                  setSingleSource(null);
                  clearResult();
                  setSourceFocus({ x: 50, y: 30, ok: false });
                }}
                className="mt-3 text-xs text-white/60 hover:text-white"
              >
                Remove source
              </button>
            )}
          </section>

          {/* Target */}
          <section className="rounded-2xl border border-white/10 bg-white/5 p-5">
            <div className="flex items-center justify-between">
              <h2 className="text-sm font-semibold">Target Image</h2>
              {target ? (
                <span className="text-xs text-emerald-300">Selected</span>
              ) : (
                <span className="text-xs text-white/50">Required</span>
              )}
            </div>

            <button
              onClick={() =>
                pickSingle((p) => {
                  revoke(target);
                  setTarget(p);
                  clearResult();
                  resetDetectedFaces();
                  setTargetFocus({ x: 50, y: 30, ok: false });
                  setManualTargetFocus({ x: 50, y: 30 });
                }, "tgt")
              }
              className="mt-3 w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-left text-sm hover:bg-white/10"
            >
              {target ? "Change target image" : "Choose target image"}
              {target && (
                <div className="mt-1 text-xs text-emerald-300">Selected: {target.name}</div>
              )}
            </button>

            {target && (
              <div className="mt-3 text-xs text-white/50">
                {target.file.type || "image"} • {formatBytes(target.file.size)}
              </div>
            )}

            {target?.url ? (
              <div className="mt-4 overflow-hidden rounded-xl border border-white/10 bg-black/30">
                <img
                  src={target.url}
                  alt="Target preview"
                  className="h-56 w-full bg-black"
                  style={{
                    objectFit: autoFaceFocus ? "cover" : "contain",
                    objectPosition: autoFaceFocus
                      ? focusPosition(targetFocus, manualTargetFocus)
                      : "50% 50%",
                  }}
                />
              </div>
            ) : (
              <div className="mt-4 flex h-56 items-center justify-center rounded-xl border border-white/10 bg-black/30 text-sm text-white/40">
                No target yet
              </div>
            )}

            {autoFaceFocus && target?.url && !targetFocus.ok && (
              <div className="mt-3 space-y-2">
                <div className="text-xs text-white/50">
                  Auto focus fallback ({targetFocus.reason || "unknown"}) — adjust:
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-white/50 w-10">X</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={manualTargetFocus.x}
                    onChange={(e) =>
                      setManualTargetFocus((p) => ({ ...p, x: Number(e.target.value) }))
                    }
                    className="w-full"
                  />
                  <span className="text-xs text-white/50 w-10 text-right">{manualTargetFocus.x}</span>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs text-white/50 w-10">Y</span>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={manualTargetFocus.y}
                    onChange={(e) =>
                      setManualTargetFocus((p) => ({ ...p, y: Number(e.target.value) }))
                    }
                    className="w-full"
                  />
                  <span className="text-xs text-white/50 w-10 text-right">{manualTargetFocus.y}</span>
                </div>
              </div>
            )}

            {target && (
              <button
                onClick={() => {
                  revoke(target);
                  setTarget(null);
                  resetDetectedFaces();
                  clearResult();
                  setTargetFocus({ x: 50, y: 30, ok: false });
                }}
                className="mt-3 text-xs text-white/60 hover:text-white"
              >
                Remove target
              </button>
            )}
          </section>
        </div>

        {/* MULTI MODE */}
        {multiMode && (
          <>
            <div className="mt-6 grid gap-6 lg:grid-cols-3">
              <section className="rounded-2xl border border-white/10 bg-white/5 p-5">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold">Detect Faces</h3>
                  {hasDetectedFaces ? (
                    <span className="text-xs text-emerald-300">{faces.length} found</span>
                  ) : (
                    <span className="text-xs text-white/50">Required</span>
                  )}
                </div>

                <div className="mt-3 text-sm text-white/70">Real detection via backend.</div>

                <button
                  onClick={detectFacesReal}
                  disabled={!canDetect}
                  className={`mt-4 w-full rounded-xl px-4 py-3 text-sm font-medium ${
                    canDetect
                      ? "bg-blue-500/90 hover:bg-blue-500"
                      : "bg-white/10 text-white/40 cursor-not-allowed"
                  }`}
                >
                  Detect Faces
                </button>

                {hasDetectedFaces && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {faces.map((f) => (
                      <span
                        key={f.faceId}
                        className="rounded-full border border-white/10 bg-white/5 px-2 py-1 text-xs text-white/70"
                      >
                        {f.label}
                      </span>
                    ))}
                  </div>
                )}

                {hasDetectedFaces && (
                  <button
                    onClick={() => {
                      resetDetectedFaces();
                      clearResult();
                    }}
                    className="mt-3 text-xs text-white/60 hover:text-white"
                  >
                    Reset detections
                  </button>
                )}
              </section>

              <section className="rounded-2xl border border-white/10 bg-white/5 p-5 lg:col-span-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold">Source Faces (Multi)</h3>
                  {hasSources ? (
                    <span className="text-xs text-emerald-300">{sources.length} added</span>
                  ) : (
                    <span className="text-xs text-white/50">Add at least 1</span>
                  )}
                </div>

                <div className="mt-3 text-sm text-white/70">
                  Drag a source card onto a face row to assign it.
                </div>

                <button
                  onClick={() =>
                    pickMany((items) => {
                      setSources((prev) => [...prev, ...items]);
                      clearResult();
                    }, "msrc")
                  }
                  disabled={isProcessing}
                  className={`mt-4 w-full rounded-xl px-4 py-3 text-sm font-medium ${
                    !isProcessing
                      ? "bg-white/10 hover:bg-white/15"
                      : "bg-white/10 text-white/40 cursor-not-allowed"
                  }`}
                >
                  Add source faces
                </button>

                {sources.length > 0 && (
                  <div className="mt-4 grid gap-3 md:grid-cols-2">
                    {sources.map((s) => (
                      <div
                        key={s.id}
                        draggable={!isProcessing}
                        onDragStart={() => onDragStartSource(s.id)}
                        onDragEnd={() => {
                          setDragSourceId(null);
                          setDragOverFaceId(null);
                        }}
                        className={`flex items-center gap-3 rounded-xl border p-3 transition ${
                          dragSourceId === s.id
                            ? "border-blue-500/50 bg-blue-500/10"
                            : "border-white/10 bg-black/25"
                        } ${isProcessing ? "opacity-50" : "cursor-grab active:cursor-grabbing"}`}
                        title="Drag me onto a target face row"
                      >
                        <img
                          src={s.url}
                          alt=""
                          className="h-12 w-12 rounded-lg object-cover object-center"
                        />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-sm text-white/80">{s.name}</div>
                          <div className="text-xs text-white/40">{formatBytes(s.file.size)}</div>
                        </div>
                        <button
                          onClick={() => removeSource(s.id)}
                          className="text-xs text-white/60 hover:text-white"
                          disabled={isProcessing}
                        >
                          Remove
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </section>
            </div>

            <section className="mt-6 rounded-2xl border border-white/10 bg-white/5 p-6">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div>
                  <h3 className="text-sm font-semibold">Assign sources to target faces</h3>
                  <p className="mt-1 text-sm text-white/70">
                    Drag & drop a source onto a face row, or use the dropdown.
                  </p>
                </div>

                <div className="flex gap-3">
                  <button
                    onClick={autoAssign}
                    disabled={!hasDetectedFaces || !hasSources || isProcessing}
                    className={`rounded-xl border border-white/10 px-4 py-2 text-sm ${
                      hasDetectedFaces && hasSources && !isProcessing
                        ? "bg-white/5 hover:bg-white/10"
                        : "bg-white/5 text-white/30 cursor-not-allowed"
                    }`}
                  >
                    Auto-assign
                  </button>

                  <button
                    onClick={() => {
                      const next: Record<string, string> = {};
                      faces.forEach((f) => (next[f.faceId] = "none"));
                      setAssignments(next);
                      clearResult();
                    }}
                    disabled={!hasDetectedFaces || isProcessing}
                    className={`rounded-xl border border-white/10 px-4 py-2 text-sm ${
                      hasDetectedFaces && !isProcessing
                        ? "bg-white/5 hover:bg-white/10"
                        : "bg-white/5 text-white/30 cursor-not-allowed"
                    }`}
                  >
                    Clear map
                  </button>
                </div>
              </div>

              {!hasDetectedFaces ? (
                <div className="mt-5 rounded-xl border border-white/10 bg-black/30 p-4 text-sm text-white/60">
                  Detect faces first to enable mapping.
                </div>
              ) : (
                <div className="mt-5 overflow-hidden rounded-xl border border-white/10">
                  <div className="grid grid-cols-12 bg-white/5 px-4 py-3 text-xs text-white/60">
                    <div className="col-span-4">Target Face</div>
                    <div className="col-span-8">Assigned Source</div>
                  </div>

                  <div className="divide-y divide-white/10">
                    {faces.map((f) => {
                      const assignedId = assignments[f.faceId] ?? "none";
                      const assigned = assignedId !== "none" ? sourceById.get(assignedId) : null;
                      const isOver = dragOverFaceId === f.faceId && !!dragSourceId;

                      return (
                        <div
                          key={f.faceId}
                          className={`grid grid-cols-12 items-center px-4 py-3 transition ${
                            isOver ? "bg-blue-500/10" : ""
                          }`}
                          onDragOver={(e) => {
                            if (!dragSourceId) return;
                            e.preventDefault();
                            setDragOverFaceId(f.faceId);
                          }}
                          onDragLeave={() => {
                            setDragOverFaceId((cur) => (cur === f.faceId ? null : cur));
                          }}
                          onDrop={(e) => {
                            e.preventDefault();
                            onDropOnFace(f.faceId);
                          }}
                        >
                          <div className="col-span-4 text-sm">{f.label}</div>

                          <div className="col-span-8">
                            <div className="flex flex-col gap-2">
                              <div
                                className={`rounded-xl border border-dashed px-3 py-2 text-sm ${
                                  isOver
                                    ? "border-blue-500/60 bg-blue-500/10 text-white/80"
                                    : "border-white/15 bg-black/30 text-white/60"
                                }`}
                              >
                                {assigned ? (
                                  <div className="flex items-center gap-2">
                                    <img
                                      src={assigned.url}
                                      alt=""
                                      className="h-8 w-8 rounded-lg object-cover object-center"
                                    />
                                    <div className="min-w-0">
                                      <div className="truncate text-sm text-white/80">{assigned.name}</div>
                                      <div className="text-xs text-white/40">Assigned</div>
                                    </div>
                                    <button
                                      onClick={() => assignFace(f.faceId, "none")}
                                      className="ml-auto text-xs text-white/60 hover:text-white"
                                      disabled={isProcessing}
                                    >
                                      Clear
                                    </button>
                                  </div>
                                ) : (
                                  <div className="text-sm">
                                    Drop a source here{" "}
                                    <span className="text-white/40">(or use dropdown)</span>
                                  </div>
                                )}
                              </div>

                              <select
                                className="w-full rounded-xl border border-white/10 bg-black/30 px-3 py-2 text-sm text-white/80"
                                value={assignedId}
                                onChange={(e) => {
                                  setAssignments((prev) => ({ ...prev, [f.faceId]: e.target.value }));
                                  clearResult();
                                }}
                                disabled={!hasSources || isProcessing}
                              >
                                <option value="none">None (leave as is)</option>
                                {sources.map((s) => (
                                  <option key={s.id} value={s.id}>
                                    {s.name}
                                  </option>
                                ))}
                              </select>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              <div className="mt-4 text-xs text-white/50">
                Assigned: {assignedCount} / {faces.length}
              </div>
            </section>
          </>
        )}

        {/* Run + Result */}
        <section className="mt-6 grid gap-6 lg:grid-cols-2">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
            <h3 className="text-sm font-semibold">{multiMode ? "Run Multi-Face Swap" : "Run Single Swap"}</h3>
            <p className="mt-2 text-sm text-white/70">
              {multiMode ? "Target + many sources + mapping" : "Source → Target"}
            </p>

            <div className="mt-4 space-y-3">
              <button
                onClick={multiMode ? handleMultiSwap : handleSingleSwap}
                disabled={multiMode ? !canSwapMulti : !canSwapSingle}
                className={`w-full rounded-xl px-4 py-3 text-sm font-semibold ${
                  (multiMode ? canSwapMulti : canSwapSingle)
                    ? "bg-blue-500/90 hover:bg-blue-500"
                    : "bg-white/10 text-white/40 cursor-not-allowed"
                }`}
              >
                {isProcessing ? "Processing…" : multiMode ? "Swap Multiple Faces" : "Swap Face"}
              </button>

              {isProcessing && (
                <button
                  onClick={() => {
                    cancelInFlight();
                    setIsProcessing(false);
                    stopProgressError("canceled");
                  }}
                  className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-white/80 hover:bg-white/10"
                >
                  Cancel
                </button>
              )}

              {(isProcessing || stage !== "idle") && (
                <div className="rounded-xl border border-white/10 bg-black/30 p-3">
                  <div className="flex items-center justify-between text-xs text-white/60">
                    <span>{stageLabel(stage)}</span>
                    <span>{Math.round(progress)}%</span>
                  </div>
                  <div className="mt-2 h-2 w-full overflow-hidden rounded-full bg-white/10">
                    <div
                      className="h-full rounded-full bg-blue-500/90 transition-all"
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>
              )}
            </div>

            <div className="mt-3 text-xs text-white/50">Connected to real backend via /api proxy.</div>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold">Result</h3>
              {isProcessing ? (
                <span className="text-xs text-blue-300">Working…</span>
              ) : resultUrl ? (
                <span className="text-xs text-emerald-300">Ready</span>
              ) : (
                <span className="text-xs text-white/50">No result</span>
              )}
            </div>

            <div className="mt-4 overflow-hidden rounded-xl border border-white/10 bg-black/30">
              {resultUrl ? (
                <img src={resultUrl} alt="Result" className="h-64 w-full bg-black object-contain" />
              ) : (
                <div className="flex h-64 items-center justify-center text-sm text-white/50">
                  {isProcessing ? "Swapping faces…" : "Your swapped image will appear here"}
                </div>
              )}
            </div>

            <div className="mt-4 flex flex-wrap gap-3">
              <a
                href={resultUrl || "#"}
                download={multiMode ? "morphai-multi-result.png" : "morphai-result.png"}
                className={`rounded-xl border border-white/10 px-4 py-2 text-sm ${
                  resultUrl ? "bg-white/5 hover:bg-white/10" : "bg-white/5 text-white/30 pointer-events-none"
                }`}
              >
                Download
              </a>

              <button
                onClick={clearResult}
                disabled={!resultUrl || isProcessing}
                className={`rounded-xl border border-white/10 px-4 py-2 text-sm ${
                  resultUrl && !isProcessing
                    ? "bg-white/5 hover:bg-white/10"
                    : "bg-white/5 text-white/30 cursor-not-allowed"
                }`}
              >
                Clear
              </button>
            </div>
          </div>
        </section>

        <div className="mt-10 text-center text-xs text-white/40">
          Auto zoom uses browser FaceDetector. If unsupported, use the sliders.
        </div>
      </div>
    </main>
  );
}

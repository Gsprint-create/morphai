export default function ComingSoonPage() {
  return (
    <main className="min-h-screen bg-black text-white flex items-center justify-center px-6">
      <div className="max-w-xl text-center">
        <h1 className="text-3xl font-semibold">MorphAI</h1>
        <p className="mt-2 text-white/70">
          Tools are in private testing. Launching soon.
        </p>

        <a
          href="/test-access"
          className="inline-block mt-6 rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-white hover:bg-white/10"
        >
          Private access →
        </a>
      </div>
    </main>
  );
}

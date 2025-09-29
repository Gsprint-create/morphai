export const metadata = {
  title: "Face Swap (Modal + Vercel)",
  description: "Simple face swap frontend proxying to a Modal FastAPI backend",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

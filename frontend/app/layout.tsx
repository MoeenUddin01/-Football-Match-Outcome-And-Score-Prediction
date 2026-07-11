import type { Metadata } from "next";
import "./globals.css";
import NavBar from "@/components/NavBar";
import SiteBackground from "@/components/SiteBackground";

export const metadata: Metadata = {
  title: "ChronoPitch",
  description: "Football match outcome and score prediction",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-background text-white font-sans">
        {/* Site-wide background: drifting flag particles */}
        <SiteBackground />

        {/* Grain / noise texture overlay */}
        <svg className="grain-overlay" aria-hidden="true">
          <filter id="grain-filter">
            <feTurbulence
              type="fractalNoise"
              baseFrequency="0.65"
              numOctaves="3"
              stitchTiles="stitch"
            />
            <feColorMatrix type="saturate" values="0" />
          </filter>
          <rect width="100%" height="100%" filter="url(#grain-filter)" />
        </svg>

        <NavBar />
        <main className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
      </body>
    </html>
  );
}

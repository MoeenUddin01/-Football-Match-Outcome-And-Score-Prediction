"use client";

import { useMemo } from "react";
import teamFlags from "@/lib/teamFlags";

const SAMPLE_TEAMS = [
  "Brazil", "Argentina", "Germany", "France", "England",
  "Spain", "Italy", "Japan", "Nigeria", "Mexico",
  "Morocco", "South Korea", "Portugal", "Netherlands", "Senegal",
];

/**
 * Site-wide decorative background: 12 sparse flag particles at very low
 * opacity, slowly drifting with staggered timing. Purely decorative,
 * behind all content (low z-index). Disabled by prefers-reduced-motion.
 */
export default function SiteBackground() {
  const particles = useMemo(() => {
    return Array.from({ length: 12 }, (_, i) => {
      const team = SAMPLE_TEAMS[i % SAMPLE_TEAMS.length];
      const code = teamFlags[team] ?? null;
      return {
        key: `site-bg-${i}`,
        code,
        left: `${5 + (i * 7.5) % 90}%`,
        top: `${8 + ((i * 13) % 84)}%`,
        size: 18 + (i % 4) * 4,
        duration: 10 + (i % 5) * 2,
        delay: i * 0.8,
      };
    });
  }, []);

  return (
    <div
      className="fixed inset-0 z-0 overflow-hidden pointer-events-none"
      aria-hidden="true"
    >
      {particles.map((p) =>
        p.code ? (
          <img
            key={p.key}
            src={`https://flagcdn.com/w40/${p.code}.png`}
            alt=""
            className="flag-particle"
            style={{
              left: p.left,
              top: p.top,
              width: p.size,
              height: Math.round(p.size * 0.75),
              "--drift-duration": `${p.duration}s`,
              "--drift-delay": `${p.delay}s`,
            } as React.CSSProperties}
          />
        ) : (
          <div
            key={p.key}
            className="flag-particle rounded-full"
            style={{
              left: p.left,
              top: p.top,
              width: p.size * 0.5,
              height: p.size * 0.5,
              backgroundColor: "#555",
              "--drift-duration": `${p.duration}s`,
              "--drift-delay": `${p.delay}s`,
            } as React.CSSProperties}
          />
        )
      )}
    </div>
  );
}

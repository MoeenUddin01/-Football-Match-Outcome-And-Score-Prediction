"use client";

import teamFlags from "@/lib/teamFlags";

interface TeamFlagProps {
  team: string;
  size?: number;
  className?: string;
}

export default function TeamFlag({ team, size = 24, className = "" }: TeamFlagProps) {
  const code = teamFlags[team];

  if (code) {
    return (
      <img
        src={`https://flagcdn.com/w80/${code}.png`}
        alt={`${team} flag`}
        width={size}
        height={Math.round(size * 0.75)}
        className={`inline-block rounded-sm object-cover ${className}`}
        style={{ width: size, height: Math.round(size * 0.75) }}
        loading="lazy"
      />
    );
  }

  // Placeholder: circle with first letter
  const letter = team.charAt(0).toUpperCase();
  return (
    <span
      className={`inline-flex items-center justify-center rounded-full bg-white/10 text-white/60 font-bold ${className}`}
      style={{ width: size, height: size, fontSize: size * 0.45 }}
      title={team}
    >
      {letter}
    </span>
  );
}

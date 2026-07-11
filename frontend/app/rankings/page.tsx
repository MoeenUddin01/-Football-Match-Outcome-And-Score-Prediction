"use client";

import { useEffect, useState } from "react";
import { fetchRankings } from "@/lib/api";
import type { TeamRanking } from "@/lib/api";
import TeamFlag from "@/components/TeamFlag";

export default function RankingsPage() {
  const [rankings, setRankings] = useState<TeamRanking[]>([]);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchRankings()
      .then(setRankings)
      .catch(() => setError("Failed to load rankings. Is the backend running?"))
      .finally(() => setLoading(false));
  }, []);

  const filtered = rankings.filter((r) =>
    r.team.toLowerCase().includes(search.toLowerCase())
  );

  if (loading) {
    return <div className="flex justify-center items-center h-64"><div className="animate-pulse text-muted">Loading rankings...</div></div>;
  }

  if (error) {
    return <div className="text-center py-12 text-primary">{error}</div>;
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Elo Rankings</h1>
        <p className="text-muted">{rankings.length} teams ranked by current Elo rating</p>
      </div>

      <input
        type="text"
        placeholder="Search team..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full sm:w-80 bg-card border border-white/10 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:border-primary"
      />

      <div className="space-y-1">
        {filtered.map((r) => (
          <div
            key={r.team}
            className="flex items-center gap-4 bg-card border border-white/5 rounded-card px-4 py-3 hover:border-white/10 transition-colors card-hover"
          >
            <div className="w-8 h-8 rounded-full bg-background flex items-center justify-center text-sm font-bold text-muted flex-shrink-0">
              {r.rank}
            </div>
            <TeamFlag team={r.team} size={24} />
            <span className="flex-1 font-medium text-sm">{r.team}</span>
            <span className="text-sm font-mono text-primary">{r.elo.toFixed(0)}</span>
          </div>
        ))}
        {filtered.length === 0 && (
          <p className="text-center text-muted py-8">No teams match &ldquo;{search}&rdquo;</p>
        )}
      </div>
    </div>
  );
}

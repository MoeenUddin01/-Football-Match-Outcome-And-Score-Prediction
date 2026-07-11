"use client";

import { useEffect, useState } from "react";
import { fetchTeams, fetchTournaments, fetchTopScorers } from "@/lib/api";
import type { Scorer, Tournament } from "@/lib/api";
import TeamFlag from "@/components/TeamFlag";

export default function ScorersPage() {
  const [scorers, setScorers] = useState<Scorer[]>([]);
  const [teams, setTeams] = useState<string[]>([]);
  const [tournaments, setTournaments] = useState<Tournament[]>([]);
  const [teamFilter, setTeamFilter] = useState("");
  const [tournamentFilter, setTournamentFilter] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([fetchTeams(), fetchTournaments()])
      .then(([t, tourn]) => { setTeams(t); setTournaments(tourn); })
      .catch(() => {});
  }, []);

  useEffect(() => {
    setLoading(true);
    setError("");
    fetchTopScorers({
      team: teamFilter || undefined,
      tournament: tournamentFilter || undefined,
      limit: 25,
    })
      .then(setScorers)
      .catch(() => setError("Failed to load scorers. Is the backend running?"))
      .finally(() => setLoading(false));
  }, [teamFilter, tournamentFilter]);

  if (loading && scorers.length === 0) {
    return <div className="flex justify-center items-center h-64"><div className="animate-pulse text-muted">Loading scorers...</div></div>;
  }

  if (error) {
    return <div className="text-center py-12 text-primary">{error}</div>;
  }

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Top Scorers</h1>
        <p className="text-muted">All-time top scorers across 150+ years of international football</p>
      </div>

      {/* Filters */}
      <div className="flex gap-4 flex-wrap">
        <div className="space-y-1">
          <label className="text-sm text-muted">Team</label>
          <select
            value={teamFilter}
            onChange={(e) => setTeamFilter(e.target.value)}
            className="w-full sm:w-56 bg-card border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary"
          >
            <option value="">All teams</option>
            {teams.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>
        </div>
        <div className="space-y-1">
          <label className="text-sm text-muted">Tournament</label>
          <select
            value={tournamentFilter}
            onChange={(e) => setTournamentFilter(e.target.value)}
            className="w-full sm:w-56 bg-card border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary"
          >
            <option value="">All tournaments</option>
            {tournaments.map((t) => <option key={t.name} value={t.name}>{t.name}</option>)}
          </select>
        </div>
      </div>

      {/* Leaderboard */}
      <div className="space-y-1">
        {scorers.map((s, i) => {
          const penaltyShare = s.goal_count > 0 ? s.penalty_count / s.goal_count : 0;
          return (
            <div
              key={`${s.scorer}-${i}`}
              className="flex items-center gap-4 bg-card border border-white/5 rounded-card px-4 py-3 hover:border-white/10 transition-colors card-hover"
            >
              <div className="w-8 h-8 rounded-full bg-background flex items-center justify-center text-sm font-bold text-muted flex-shrink-0">
                {i + 1}
              </div>
              <TeamFlag team={s.team} size={24} />
              <div className="flex-1 min-w-0">
                <p className="font-medium text-sm truncate">{s.scorer}</p>
                <p className="text-xs text-muted">{s.team}</p>
              </div>
              <div className="flex items-center gap-2">
                {penaltyShare > 0.2 && (
                  <span className="text-[10px] bg-secondary/30 text-secondary px-1.5 py-0.5 rounded font-medium">
                    P:{s.penalty_count}
                  </span>
                )}
                <span className="text-lg font-bold font-mono text-primary min-w-[3ch] text-right">
                  {s.goal_count}
                </span>
              </div>
            </div>
          );
        })}
        {scorers.length === 0 && (
          <p className="text-center text-muted py-8">No scorers found for the selected filters.</p>
        )}
      </div>
    </div>
  );
}

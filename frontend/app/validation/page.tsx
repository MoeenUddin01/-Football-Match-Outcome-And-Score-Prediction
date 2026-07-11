"use client";

import { useEffect, useState, useMemo } from "react";
import { fetchValidationReport } from "@/lib/api";
import type { ValidationMatch } from "@/lib/api";
import TeamFlag from "@/components/TeamFlag";

export default function ValidationPage() {
  const [matches, setMatches] = useState<ValidationMatch[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetchValidationReport()
      .then(setMatches)
      .catch(() => setError("Failed to load validation report. Is the backend running?"))
      .finally(() => setLoading(false));
  }, []);

  const stats = useMemo(() => {
    if (matches.length === 0) return null;
    const correct = matches.filter((m) => m.xgb_correct).length;
    const exact = matches.filter((m) => m.poisson_exact).length;
    const n = matches.length;
    return {
      total: n,
      accuracy: correct / n,
      exactScore: exact / n,
      correct,
      exact,
    };
  }, [matches]);

  if (loading) {
    return <div className="flex justify-center items-center h-64"><div className="animate-pulse text-muted">Loading validation report...</div></div>;
  }

  if (error) {
    return <div className="text-center py-12 text-primary">{error}</div>;
  }

  return (
    <div className="space-y-8">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold">Validation Report</h1>
        <p className="text-muted">World Cup 2022 & Euro 2024 backtest results ({stats?.total} matches)</p>
      </div>

      {/* Stat Cards */}
      {stats && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="bg-card border border-white/5 rounded-card p-5 text-center space-y-1 card-hover">
            <p className="text-3xl font-bold text-primary">{(stats.accuracy * 100).toFixed(1)}%</p>
            <p className="text-sm text-muted">Outcome Accuracy</p>
            <p className="text-xs text-muted">{stats.correct}/{stats.total} correct</p>
          </div>
          <div className="bg-card border border-white/5 rounded-card p-5 text-center space-y-1 card-hover">
            <p className="text-3xl font-bold text-secondary">{(stats.exactScore * 100).toFixed(1)}%</p>
            <p className="text-sm text-muted">Exact Scoreline</p>
            <p className="text-xs text-muted">{stats.exact}/{stats.total} matches</p>
          </div>
          <div className="bg-card border border-white/5 rounded-card p-5 text-center space-y-1 card-hover">
            <p className="text-3xl font-bold text-muted">
              {matches.filter((m) => m.xgb_correct && !m.poisson_exact).length}
            </p>
            <p className="text-sm text-muted">Right Winner, Wrong Score</p>
          </div>
        </div>
      )}

      {/* Matches Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-white/10 text-muted text-left">
              <th className="pb-3 pr-4">Date</th>
              <th className="pb-3 pr-4">Match</th>
              <th className="pb-3 pr-4">Result</th>
              <th className="pb-3 pr-4">XGB Pick</th>
              <th className="pb-3 pr-4">Probs (H/D/A)</th>
              <th className="pb-3 pr-4">Poisson</th>
              <th className="pb-3">Status</th>
            </tr>
          </thead>
          <tbody>
            {matches.map((m, i) => (
              <tr
                key={i}
                className={`border-b border-white/5 ${
                  m.xgb_correct
                    ? "border-l-2 border-l-green-500/40"
                    : "border-l-2 border-l-red-500/40"
                }`}
              >
                <td className="py-3 pr-4 text-muted whitespace-nowrap">{m.date}</td>
                <td className="py-3 pr-4">
                  <span className="font-medium flex items-center gap-1.5">
                    <TeamFlag team={m.home_team} size={16} />
                    {m.home_team}
                  </span>
                  <span className="text-muted"> vs </span>
                  <span className="font-medium flex items-center gap-1.5">
                    <TeamFlag team={m.away_team} size={16} />
                    {m.away_team}
                  </span>
                </td>
                <td className="py-3 pr-4 font-mono">{m.home_score}-{m.away_score}</td>
                <td className="py-3 pr-4 capitalize">{m.xgb_pick.replace("_", " ")}</td>
                <td className="py-3 pr-4 text-muted font-mono text-xs">
                  {(m.xgb_probs.home_win_prob * 100).toFixed(0)}/
                  {(m.xgb_probs.draw_prob * 100).toFixed(0)}/
                  {(m.xgb_probs.away_win_prob * 100).toFixed(0)}
                </td>
                <td className="py-3 pr-4 font-mono">
                  {m.poisson_home_goals}-{m.poisson_away_goals}
                </td>
                <td className="py-3">
                  {m.xgb_correct ? (
                    <span className="text-green-400 text-xs font-medium">Correct</span>
                  ) : (
                    <span className="text-red-400 text-xs font-medium">Wrong</span>
                  )}
                  {m.poisson_exact && (
                    <span className="text-secondary text-xs ml-2">(exact)</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

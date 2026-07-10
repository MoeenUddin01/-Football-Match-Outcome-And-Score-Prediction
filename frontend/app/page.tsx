"use client";

import { useEffect, useState } from "react";
import { fetchTeams, fetchTournaments, postPredict } from "@/lib/api";
import type { PredictResponse, Tournament } from "@/lib/api";
import TeamFlag from "@/components/TeamFlag";

const SAMPLE_MATCHES = [
  { home: "Brazil", away: "Argentina", tournament: "FIFA World Cup", neutral: true },
  { home: "England", away: "Spain", tournament: "UEFA Euro", neutral: true },
  { home: "France", away: "Germany", tournament: "FIFA World Cup", neutral: true },
  { home: "Japan", away: "Brazil", tournament: "Friendly", neutral: false },
];

export default function HomePage() {
  const [teams, setTeams] = useState<string[]>([]);
  const [tournaments, setTournaments] = useState<Tournament[]>([]);
  const [homeTeam, setHomeTeam] = useState("");
  const [awayTeam, setAwayTeam] = useState("");
  const [tournament, setTournament] = useState("");
  const [neutral, setNeutral] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [dataLoading, setDataLoading] = useState(true);

  useEffect(() => {
    Promise.all([fetchTeams(), fetchTournaments()])
      .then(([t, tourn]) => { setTeams(t); setTournaments(tourn); })
      .catch(() => setError("Failed to load data. Is the backend running?"))
      .finally(() => setDataLoading(false));
  }, []);

  const handlePredict = async () => {
    if (!homeTeam || !awayTeam || !tournament) {
      setError("Please select both teams and a tournament.");
      return;
    }
    setError("");
    setResult(null);
    setLoading(true);
    try {
      const res = await postPredict({ home_team: homeTeam, away_team: awayTeam, tournament, neutral });
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Prediction failed");
    } finally {
      setLoading(false);
    }
  };

  const fillSample = (s: typeof SAMPLE_MATCHES[0]) => {
    setHomeTeam(s.home);
    setAwayTeam(s.away);
    setTournament(s.tournament);
    setNeutral(s.neutral);
    setResult(null);
    setError("");
  };

  if (dataLoading) {
    return <div className="flex justify-center items-center h-64"><div className="animate-pulse text-muted">Loading...</div></div>;
  }

  return (
    <div className="space-y-12">
      {/* Hero */}
      <section className="relative -mx-6 -mt-6 overflow-hidden" style={{ minHeight: 340 }}>
        {/* Background image */}
        <img
          src="https://images.unsplash.com/photo-1522778119026-d647f0596c20?w=1600&q=80"
          alt=""
          className="absolute inset-0 w-full h-full object-cover"
          loading="eager"
        />
        {/* Dark gradient overlay */}
        <div className="absolute inset-0 bg-gradient-to-r from-black/90 via-black/70 to-black/40" />
        {/* Content */}
        <div className="relative z-10 px-6 py-16 flex flex-col justify-center" style={{ minHeight: 340 }}>
          <h1 className="text-5xl font-bold tracking-tight mb-4">
            <span className="text-primary">Chrono</span>Pitch
          </h1>
          <p className="text-muted text-lg max-w-2xl">
            Predict international football match outcomes and scorelines using
            Elo ratings, rolling form statistics, and machine learning.
            Trained on 49,000+ matches from 1872 to 2026.
          </p>
        </div>
      </section>

      {/* Predictor Form */}
      <section className="bg-card rounded-card p-6 border border-white/5 space-y-6">
        <h2 className="text-xl font-semibold">Match Predictor</h2>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div className="space-y-1">
            <label className="text-sm text-muted">Home Team</label>
            <div className="relative">
              <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none z-10">
                <TeamFlag team={homeTeam} size={18} />
              </div>
              <select
                value={homeTeam}
                onChange={(e) => setHomeTeam(e.target.value)}
                className="w-full bg-background border border-white/10 rounded-lg pl-9 pr-3 py-2 text-sm focus:outline-none focus:border-primary appearance-none"
              >
                <option value="">Select team...</option>
                {teams.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-sm text-muted">Away Team</label>
            <div className="relative">
              <div className="absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none z-10">
                <TeamFlag team={awayTeam} size={18} />
              </div>
              <select
                value={awayTeam}
                onChange={(e) => setAwayTeam(e.target.value)}
                className="w-full bg-background border border-white/10 rounded-lg pl-9 pr-3 py-2 text-sm focus:outline-none focus:border-primary appearance-none"
              >
                <option value="">Select team...</option>
                {teams.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
          </div>
          <div className="space-y-1">
            <label className="text-sm text-muted">Tournament</label>
            <select
              value={tournament}
              onChange={(e) => setTournament(e.target.value)}
              className="w-full bg-background border border-white/10 rounded-lg px-3 py-2 text-sm focus:outline-none focus:border-primary"
            >
              <option value="">Select tournament...</option>
              {tournaments.map((t) => <option key={t.name} value={t.name}>{t.name}</option>)}
            </select>
          </div>
          <div className="space-y-1 flex items-end">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={neutral}
                onChange={(e) => setNeutral(e.target.checked)}
                className="w-4 h-4 rounded border-white/20"
              />
              <span className="text-sm text-muted">Neutral venue</span>
            </label>
          </div>
        </div>

        {error && <p className="text-primary text-sm">{error}</p>}

        <button
          onClick={handlePredict}
          disabled={loading}
          className="bg-primary hover:bg-primary/90 disabled:opacity-50 text-white font-medium px-6 py-2.5 rounded-lg text-sm transition-colors"
        >
          {loading ? "Predicting..." : "Predict"}
        </button>
      </section>

      {/* Result Card */}
      {result && (
        <section className="bg-card rounded-card p-6 border border-white/5 space-y-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-semibold flex items-center gap-3">
              <span className="flex items-center gap-2">
                <TeamFlag team={result.home_team} size={28} />
                {result.home_team}
              </span>
              <span className="text-muted font-normal">vs</span>
              <span className="flex items-center gap-2">
                <TeamFlag team={result.away_team} size={28} />
                {result.away_team}
              </span>
            </h2>
            <span className="text-xs text-muted bg-background px-2 py-1 rounded">
              {result.tournament}{result.neutral ? " (neutral)" : ""}
            </span>
          </div>

          {/* XGBoost Probs Bar */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-muted">XGBoost Classifier</h3>
            <div className="flex h-8 rounded-lg overflow-hidden text-xs font-medium">
              <div
                className="bg-primary flex items-center justify-center text-white transition-all"
                style={{ width: `${result.xgb_probs.home_win * 100}%` }}
              >
                {result.xgb_probs.home_win > 0.1 && `${(result.xgb_probs.home_win * 100).toFixed(0)}%`}
              </div>
              <div
                className="bg-secondary flex items-center justify-center text-white transition-all"
                style={{ width: `${result.xgb_probs.draw * 100}%` }}
              >
                {result.xgb_probs.draw > 0.1 && `${(result.xgb_probs.draw * 100).toFixed(0)}%`}
              </div>
              <div
                className="bg-muted flex items-center justify-center text-white transition-all"
                style={{ width: `${result.xgb_probs.away_win * 100}%` }}
              >
                {result.xgb_probs.away_win > 0.1 && `${(result.xgb_probs.away_win * 100).toFixed(0)}%`}
              </div>
            </div>
            <div className="flex justify-between text-xs text-muted">
              <span className="flex items-center gap-1.5">
                <TeamFlag team={result.home_team} size={14} />
                {result.home_team}
              </span>
              <span>Draw</span>
              <span className="flex items-center gap-1.5">
                <TeamFlag team={result.away_team} size={14} />
                {result.away_team}
              </span>
            </div>
            <p className="text-sm">
              Pick: <span className="font-semibold text-primary">{result.xgb_pick.replace("_", " ")}</span>
            </p>
          </div>

          {/* Poisson Scoreline */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-muted">Poisson Scoreline</h3>
            <p className="text-3xl font-bold flex items-center gap-3">
              <span className="flex items-center gap-2">
                <TeamFlag team={result.home_team} size={32} />
                {result.home_team}
              </span>
              <span>{result.poisson_home_goals} - {result.poisson_away_goals}</span>
              <span className="flex items-center gap-2">
                <TeamFlag team={result.away_team} size={32} />
                {result.away_team}
              </span>
            </p>
            <p className="text-xs text-muted">
              Expected goals: {result.poisson_home_lambda.toFixed(2)} - {result.poisson_away_lambda.toFixed(2)}
            </p>
          </div>

          {/* Poisson Probs */}
          <div className="space-y-1">
            <h3 className="text-sm font-medium text-muted">Poisson-derived probabilities</h3>
            <div className="flex gap-6 text-sm">
              <span className="flex items-center gap-1.5">
                <TeamFlag team={result.home_team} size={14} />
                {result.home_team}: {(result.poisson_probs.home_win_prob * 100).toFixed(0)}%
              </span>
              <span>Draw: {(result.poisson_probs.draw_prob * 100).toFixed(0)}%</span>
              <span className="flex items-center gap-1.5">
                <TeamFlag team={result.away_team} size={14} />
                {result.away_team}: {(result.poisson_probs.away_win_prob * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          {/* Disagreement note */}
          {(() => {
            const xgbPick = result.xgb_pick;
            const poissonPick = result.poisson_probs.home_win_prob > result.poisson_probs.away_win_prob
              ? "home_win"
              : result.poisson_probs.away_win_prob > result.poisson_probs.home_win_prob
                ? "away_win"
                : "draw";
            if (xgbPick !== poissonPick) {
              return (
                <div className="bg-secondary/20 border border-secondary/30 rounded-lg px-4 py-2 text-sm text-secondary">
                  Models disagree on this one — XGBoost picks {xgbPick.replace("_", " ")}, Poisson picks {poissonPick.replace("_", " ")}.
                </div>
              );
            }
            return null;
          })()}
        </section>
      )}

      {/* Sample Matches */}
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-muted">Quick Predict</h2>
        <div className="flex gap-4 overflow-x-auto pb-2">
          {SAMPLE_MATCHES.map((s, i) => (
            <button
              key={i}
              onClick={() => fillSample(s)}
              className="bg-card border border-white/5 rounded-card p-4 min-w-[220px] text-left hover:border-primary/50 transition-colors flex-shrink-0"
            >
              <div className="flex items-center gap-2 mb-1">
                <TeamFlag team={s.home} size={18} />
                <span className="font-medium text-sm">{s.home}</span>
              </div>
              <div className="flex items-center gap-2">
                <TeamFlag team={s.away} size={18} />
                <span className="font-medium text-sm">{s.away}</span>
              </div>
              <p className="text-xs text-muted mt-2">{s.tournament}{s.neutral ? " (neutral)" : ""}</p>
            </button>
          ))}
        </div>
      </section>
    </div>
  );
}

"use client";

import { useEffect, useState } from "react";
import { fetchTeams, fetchTournaments, postPredict } from "@/lib/api";
import type { PredictResponse, Tournament } from "@/lib/api";
import TeamFlag from "@/components/TeamFlag";
import { getTeamColors } from "@/lib/teamColors";
import confetti from "canvas-confetti";

const SAMPLE_MATCHES = [
  { home: "Brazil", away: "Argentina", tournament: "FIFA World Cup", neutral: true },
  { home: "England", away: "Spain", tournament: "UEFA Euro", neutral: true },
  { home: "France", away: "Germany", tournament: "FIFA World Cup", neutral: true },
  { home: "Japan", away: "Brazil", tournament: "Friendly", neutral: false },
];

function fireConfetti(teamColors: [string, string]) {
  const [c1, c2] = teamColors;
  const colors = [`#${c1}`, `#${c2}`];
  confetti({ particleCount: 60, spread: 55, origin: { y: 0.7, x: 0.4 }, colors });
  confetti({ particleCount: 40, spread: 50, origin: { y: 0.7, x: 0.6 }, colors });
}

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function blendColors(c1: string, c2: string, t: number): string {
  const [r1, g1, b1] = hexToRgb(c1);
  const [r2, g2, b2] = hexToRgb(c2);
  const r = Math.round(r1 + (r2 - r1) * t);
  const g = Math.round(g1 + (g2 - g1) * t);
  const b = Math.round(b1 + (b2 - b1) * t);
  return `rgb(${r}, ${g}, ${b})`;
}

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
  const [barsActive, setBarsActive] = useState(false);

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
    setBarsActive(false);
    setLoading(true);
    try {
      const res = await postPredict({ home_team: homeTeam, away_team: awayTeam, tournament, neutral });
      setResult(res);

      // Trigger bar animation on next frame
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          setBarsActive(true);
        });
      });

      // Confetti if top probability > 70%
      const maxProb = Math.max(res.xgb_probs.home_win, res.xgb_probs.draw, res.xgb_probs.away_win);
      if (maxProb > 0.7) {
        const winnerTeam = res.xgb_probs.home_win > res.xgb_probs.away_win
          ? res.home_team
          : res.xgb_probs.away_win > res.xgb_probs.home_win
            ? res.away_team
            : null;
        if (winnerTeam) {
          setTimeout(() => fireConfetti(getTeamColors(winnerTeam)), 700);
        }
      }
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
    setBarsActive(false);
    setError("");
  };

  // --- Dynamic radial glow colors ---
  const hasTeams = homeTeam && awayTeam;
  const [homeColors, awayColors] = [
    getTeamColors(homeTeam || ""),
    getTeamColors(awayTeam || ""),
  ];
  // Blend both teams' primary colors for the glow center
  const glowPrimary = hasTeams
    ? blendColors(homeColors[0], awayColors[0], 0.5)
    : "#D63915"; // default primary orange
  const glowSecondary = hasTeams
    ? blendColors(homeColors[1], awayColors[1], 0.5)
    : "#554099"; // default secondary purple

  // --- Winning outcome highlight helper ---
  const getWinner = (p: { home_win: number; draw: number; away_win: number }) => {
    if (p.home_win >= p.draw && p.home_win >= p.away_win) return "home";
    if (p.away_win >= p.draw && p.away_win >= p.home_win) return "away";
    return "draw";
  };

  if (dataLoading) {
    return <div className="flex justify-center items-center h-64"><div className="animate-pulse text-muted">Loading...</div></div>;
  }

  return (
    <div className="space-y-12">
      {/* Hero with radial glows and headline */}
      <section className="relative -mx-6 -mt-6 overflow-hidden" style={{ minHeight: 340 }}>
        {/* Stadium background image */}
        <img
          src="/hero-bg.jpg"
          alt=""
          className="absolute inset-0 w-full h-full object-cover"
          loading="eager"
        />

        {/* Smoky blue overlay */}
        <div
          className="absolute inset-0"
          style={{
            background: "radial-gradient(ellipse at 60% 40%, rgba(30,60,120,0.35) 0%, rgba(15,30,80,0.5) 40%, rgba(5,10,30,0.6) 100%)",
          }}
        />

        {/* Radial glow: large primary (orange / blended) */}
        <div
          className="radial-glow"
          style={{
            width: 600,
            height: 600,
            top: "-15%",
            left: "10%",
            backgroundColor: glowPrimary,
            opacity: 0.18,
          }}
        />
        {/* Radial glow: smaller secondary (purple / blended), offset right */}
        <div
          className="radial-glow"
          style={{
            width: 400,
            height: 400,
            top: "10%",
            right: "5%",
            backgroundColor: glowSecondary,
            opacity: 0.15,
          }}
        />

        {/* Dark overlay for text readability */}
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
      <section className="bg-card rounded-card p-6 border border-white/5 space-y-6 card-hover">
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
        <section className="bg-card rounded-card p-6 border border-white/5 space-y-6 card-hover">
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

          {/* XGBoost Probs Bar — animated + winner highlight */}
          <div className="space-y-2">
            <h3 className="text-sm font-medium text-muted">XGBoost Classifier</h3>
            <div className="flex h-8 rounded-lg overflow-hidden text-xs font-medium">
              {(["home_win", "draw", "away_win"] as const).map((key, idx) => {
                const val = result.xgb_probs[key];
                const barClass = idx === 0 ? "prob-bar-home" : idx === 1 ? "prob-bar-draw" : "prob-bar-away";
                const colorClass = idx === 0 ? "bg-primary" : idx === 1 ? "bg-secondary" : "bg-muted";
                const winner = getWinner(result.xgb_probs);
                const isWinner = (key === "home_win" && winner === "home")
                  || (key === "away_win" && winner === "away")
                  || (key === "draw" && winner === "draw");
                return (
                  <div
                    key={key}
                    className={`${barClass} ${colorClass} flex items-center justify-center text-white ${barsActive ? "prob-bar-active" : ""} ${isWinner && val > 0.4 ? "winner-glow" : ""}`}
                    style={{
                      width: barsActive ? `${val * 100}%` : "0%",
                      "--glow-color": idx === 0 ? "rgba(214,57,21,0.5)" : idx === 1 ? "rgba(85,64,153,0.5)" : "rgba(136,136,136,0.5)",
                      fontWeight: isWinner ? 700 : 500,
                    } as React.CSSProperties}
                  >
                    {val > 0.1 && `${(val * 100).toFixed(0)}%`}
                  </div>
                );
              })}
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
              className="bg-card border border-white/5 rounded-card p-4 min-w-[220px] text-left hover:border-primary/50 transition-colors flex-shrink-0 card-hover"
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

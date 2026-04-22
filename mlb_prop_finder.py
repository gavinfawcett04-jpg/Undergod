#!/usr/bin/env python3
"""
MLB Player Prop Finder — v1 (Pitcher Strikeouts)

Pulls today's MLB schedule + probable starting pitchers, projects each pitcher's
expected strikeout total, and (optionally) compares those projections to user-
supplied sportsbook prop lines to surface the best +EV over/under bets.

Usage:
    python mlb_prop_finder.py                              # Today's projections
    python mlb_prop_finder.py --date 2026-04-22            # Specific date
    python mlb_prop_finder.py --lines lines.csv            # Add EV vs prop lines
    python mlb_prop_finder.py --lines lines.csv --min-edge 0.04

Methodology:
    expected_Ks = expected_BF * blended_K_per_PA
    blended_K_per_PA = pitcher_K%  *  (opponent_K%_vs_handedness / league_K%)
    P(over L) ≈ 1 - Poisson.cdf(ceil(L) - 1, expected_Ks)

Honest caveats:
    • Sportsbooks model this too. Edges of 1-3% are noise; ≥5% is interesting.
    • Probable pitchers get scratched. Always re-check before placing.
    • IP projections are crude (rolling avg). A real model needs game state,
      bullpen usage, weather, ump K-zone, park factors.
    • Validate with 2-4 weeks of paper-betting before risking actual money.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import requests
from scipy.stats import poisson

try:
    from pybaseball import statcast_pitcher, cache
    cache.enable()
except ImportError:
    print("ERROR: pybaseball not installed.  Run:  pip install -r requirements.txt",
          file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Constants — tune these once per season
# ---------------------------------------------------------------------------
MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
LEAGUE_K_PCT     = 0.222   # league-average K% (update each spring)
DEFAULT_IP       = 5.5     # fallback IP projection if no recent data
PA_PER_IP        = 4.3     # league-average batters faced per inning
LOOKBACK_DAYS    = 45      # rolling window for K% & IP estimates


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class GamePitcher:
    pitcher_id: int          # MLBAM id
    pitcher_name: str
    team: str
    opponent: str
    opponent_id: int
    handedness: str          # 'R' or 'L'
    home: bool
    game_pk: int


@dataclass
class StrikeoutProjection:
    pitcher: GamePitcher
    proj_ip: float
    proj_bf: float
    pitcher_k_pct: float
    opp_k_pct: float
    blended_k_pct: float
    expected_ks: float

    def prob_over(self, line: float) -> float:
        """P(strikeouts ≥ ceil(line)).  Line 6.5 → need ≥7."""
        threshold = int(np.ceil(line))
        return float(1 - poisson.cdf(threshold - 1, self.expected_ks))

    def prob_under(self, line: float) -> float:
        return 1.0 - self.prob_over(line)


@dataclass
class PropEvaluation:
    proj: StrikeoutProjection
    line: float
    over_odds: int           # American odds, e.g. -115
    under_odds: int
    p_over: float
    p_under: float
    over_edge: float         # model_prob - implied_prob
    under_edge: float
    best_side: str           # 'OVER', 'UNDER', or 'PASS'
    best_edge: float
    best_ev_per_unit: float  # EV in units on a 1-unit stake


# ---------------------------------------------------------------------------
# Schedule + probable pitchers (free MLB Stats API, no key required)
# ---------------------------------------------------------------------------
def get_todays_pitchers(target_date: date) -> list[GamePitcher]:
    params = {
        "sportId":  1,
        "date":     target_date.strftime("%Y-%m-%d"),
        "hydrate":  "probablePitcher,team",
    }
    r = requests.get(MLB_SCHEDULE_URL, params=params, timeout=15)
    r.raise_for_status()
    data = r.json()

    out: list[GamePitcher] = []
    for date_block in data.get("dates", []):
        for g in date_block.get("games", []):
            teams = g.get("teams", {})
            for side, opp_side in [("home", "away"), ("away", "home")]:
                t   = teams.get(side, {})
                opp = teams.get(opp_side, {})
                pp  = t.get("probablePitcher")
                if not pp:
                    continue
                out.append(GamePitcher(
                    pitcher_id  = pp["id"],
                    pitcher_name= pp.get("fullName", "Unknown"),
                    team        = t.get("team", {}).get("abbreviation", "???"),
                    opponent    = opp.get("team", {}).get("abbreviation", "???"),
                    opponent_id = opp.get("team", {}).get("id", 0),
                    handedness  = pp.get("pitchHand", {}).get("code", "R"),
                    home        = (side == "home"),
                    game_pk     = g.get("gamePk"),
                ))
    return out


# ---------------------------------------------------------------------------
# Pitcher rate stats from Statcast pitch-by-pitch
# ---------------------------------------------------------------------------
_K_EVENTS = {"strikeout", "strikeout_double_play"}

def _pa_terminals(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the final pitch of each PA (where 'events' is populated)."""
    return df[df["events"].notna() & (df["events"] != "")]


def get_pitcher_k_pct(pitcher_id: int, end: date,
                      window: int = LOOKBACK_DAYS) -> tuple[float, float]:
    """
    Returns (k_pct, recent_avg_ip).
    Falls back to (LEAGUE_K_PCT, DEFAULT_IP) on any failure.
    """
    start = (end - timedelta(days=window)).strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")
    try:
        df = statcast_pitcher(start, end_s, pitcher_id)
    except Exception as e:
        print(f"  [warn] statcast pull failed for {pitcher_id}: {e}",
              file=sys.stderr)
        return LEAGUE_K_PCT, DEFAULT_IP

    if df is None or df.empty:
        return LEAGUE_K_PCT, DEFAULT_IP

    pa = _pa_terminals(df)
    if pa.empty:
        return LEAGUE_K_PCT, DEFAULT_IP

    k_pct = pa["events"].isin(_K_EVENTS).mean()

    # Crude IP projection: outs recorded / 3, averaged across recent games
    games = pa.groupby("game_date")
    outs_per_game = games.apply(lambda g: _outs_in_game(g))
    avg_ip = outs_per_game.mean() / 3.0 if len(outs_per_game) else DEFAULT_IP

    return float(k_pct), float(avg_ip)


def _outs_in_game(pa_df: pd.DataFrame) -> int:
    """Approximate outs recorded in a single appearance from PA terminals."""
    out_events = {
        "strikeout", "strikeout_double_play", "field_out", "force_out",
        "grounded_into_double_play", "double_play", "sac_fly",
        "sac_bunt", "sac_fly_double_play", "fielders_choice_out",
        "triple_play",
    }
    outs = pa_df["events"].isin(out_events).sum()
    # Add 1 extra out for each double play, 2 for triple play
    outs += (pa_df["events"] == "grounded_into_double_play").sum()
    outs += (pa_df["events"] == "strikeout_double_play").sum()
    outs += (pa_df["events"] == "double_play").sum()
    outs += 2 * (pa_df["events"] == "triple_play").sum()
    return int(outs)


# ---------------------------------------------------------------------------
# Opponent K% vs handedness (also from Statcast — pulls all pitches in window
# by the requested hand, then aggregates by batting team)
# ---------------------------------------------------------------------------
_team_k_cache: dict[tuple[str, str, str], float] = {}

def get_opponent_k_pct(opp_team_id: int, pitcher_hand: str,
                       end: date, window: int = LOOKBACK_DAYS) -> float:
    """
    K% of `opp_team_id`'s batters vs pitchers of `pitcher_hand` over window.
    Uses the league-wide statcast pull and filters; cached per call.
    Falls back to LEAGUE_K_PCT on failure.
    """
    start_s = (end - timedelta(days=window)).strftime("%Y-%m-%d")
    end_s   = end.strftime("%Y-%m-%d")
    key = (start_s, end_s, pitcher_hand)

    # Lazy import to avoid pulling huge data unless needed
    if key not in _team_k_cache_full:
        try:
            from pybaseball import statcast
            df = statcast(start_dt=start_s, end_dt=end_s)
        except Exception as e:
            print(f"  [warn] league statcast pull failed: {e}", file=sys.stderr)
            _team_k_cache_full[key] = pd.DataFrame()
        else:
            _team_k_cache_full[key] = df

    df = _team_k_cache_full[key]
    if df is None or df.empty:
        return LEAGUE_K_PCT

    sub = df[df["p_throws"] == pitcher_hand]
    pa  = _pa_terminals(sub)
    if pa.empty:
        return LEAGUE_K_PCT

    # In statcast, batting team is whichever team is at-bat:
    #   inning_topbot == 'Top' -> away batting; 'Bot' -> home batting
    pa = pa.copy()
    pa["bat_team_id"] = np.where(pa["inning_topbot"] == "Top",
                                 pa["away_team"], pa["home_team"])
    # away_team / home_team are 3-letter codes, not IDs.  Need the abbr instead.
    # We'll pass abbreviation in caller for matching.  But here we got an ID,
    # so we map id → abbr via a lightweight call.
    abbr = _team_id_to_abbr(opp_team_id)
    team_pa = pa[pa["bat_team_id"] == abbr]
    if team_pa.empty:
        return LEAGUE_K_PCT
    return float(team_pa["events"].isin(_K_EVENTS).mean())


_team_k_cache_full: dict[tuple[str, str, str], pd.DataFrame] = {}
_team_abbr_cache: dict[int, str] = {}

def _team_id_to_abbr(team_id: int) -> str:
    if team_id in _team_abbr_cache:
        return _team_abbr_cache[team_id]
    try:
        r = requests.get(
            f"https://statsapi.mlb.com/api/v1/teams/{team_id}", timeout=10)
        r.raise_for_status()
        abbr = r.json()["teams"][0]["abbreviation"]
    except Exception:
        abbr = "???"
    _team_abbr_cache[team_id] = abbr
    return abbr


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------
def project_pitcher(p: GamePitcher, target_date: date) -> StrikeoutProjection:
    k_pct, proj_ip = get_pitcher_k_pct(p.pitcher_id, target_date)
    opp_k_pct      = get_opponent_k_pct(p.opponent_id, p.handedness, target_date)

    # Log5-style blend, clamped
    blended = k_pct * (opp_k_pct / LEAGUE_K_PCT)
    blended = float(np.clip(blended, 0.05, 0.55))

    proj_bf     = proj_ip * PA_PER_IP
    expected_ks = proj_bf * blended

    return StrikeoutProjection(
        pitcher        = p,
        proj_ip        = proj_ip,
        proj_bf        = proj_bf,
        pitcher_k_pct  = k_pct,
        opp_k_pct      = opp_k_pct,
        blended_k_pct  = blended,
        expected_ks    = expected_ks,
    )


# ---------------------------------------------------------------------------
# EV math
# ---------------------------------------------------------------------------
def american_to_decimal(odds: int) -> float:
    return 1 + (odds / 100 if odds > 0 else 100 / -odds)

def american_to_implied(odds: int) -> float:
    d = american_to_decimal(odds)
    return 1.0 / d


def evaluate_prop(proj: StrikeoutProjection,
                  line: float,
                  over_odds: int,
                  under_odds: int) -> PropEvaluation:
    p_over  = proj.prob_over(line)
    p_under = 1.0 - p_over

    over_edge  = p_over  - american_to_implied(over_odds)
    under_edge = p_under - american_to_implied(under_odds)

    if over_edge >= under_edge and over_edge > 0:
        best_side = "OVER"
        best_edge = over_edge
        ev = p_over * (american_to_decimal(over_odds) - 1) - (1 - p_over)
    elif under_edge > 0:
        best_side = "UNDER"
        best_edge = under_edge
        ev = p_under * (american_to_decimal(under_odds) - 1) - (1 - p_under)
    else:
        best_side = "PASS"
        best_edge = max(over_edge, under_edge)
        ev = 0.0

    return PropEvaluation(
        proj=proj, line=line,
        over_odds=over_odds, under_odds=under_odds,
        p_over=p_over, p_under=p_under,
        over_edge=over_edge, under_edge=under_edge,
        best_side=best_side, best_edge=best_edge,
        best_ev_per_unit=ev,
    )


# ---------------------------------------------------------------------------
# Lines CSV loader
# ---------------------------------------------------------------------------
def load_lines(path: str) -> dict[int, tuple[float, int, int]]:
    """
    Expected CSV columns: pitcher_id, line, over_odds, under_odds
    Returns {pitcher_id: (line, over_odds, under_odds)}
    """
    out: dict[int, tuple[float, int, int]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                pid = int(row["pitcher_id"])
                out[pid] = (
                    float(row["line"]),
                    int(row.get("over_odds")  or -110),
                    int(row.get("under_odds") or -110),
                )
            except (KeyError, ValueError) as e:
                print(f"  [warn] skipping bad row {row}: {e}", file=sys.stderr)
    return out


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
def print_projections(projs: list[StrikeoutProjection]) -> None:
    projs = sorted(projs, key=lambda x: x.expected_ks, reverse=True)
    print(f"\n{'PITCHER':<24}{'TM':<5}{'OPP':<5}{'H':<3}"
          f"{'IP':>5}{'K%':>7}{'oppK%':>7}{'xK':>6}")
    print("-" * 62)
    for p in projs:
        print(f"{p.pitcher.pitcher_name:<24}"
              f"{p.pitcher.team:<5}{p.pitcher.opponent:<5}"
              f"{p.pitcher.handedness:<3}"
              f"{p.proj_ip:>5.1f}"
              f"{p.pitcher_k_pct:>6.1%}"
              f"{p.opp_k_pct:>7.1%}"
              f"{p.expected_ks:>6.2f}")


def print_evaluations(evals: list[PropEvaluation], min_edge: float) -> None:
    evals = sorted(evals, key=lambda e: e.best_edge, reverse=True)
    print(f"\n{'PITCHER':<22}{'LINE':>5}{'xK':>6}{'SIDE':>7}"
          f"{'EDGE':>8}{'EV':>8}")
    print("-" * 56)
    for e in evals:
        flag = "  ★" if e.best_edge >= min_edge and e.best_side != "PASS" else ""
        print(f"{e.proj.pitcher.pitcher_name:<22}"
              f"{e.line:>5.1f}{e.proj.expected_ks:>6.2f}"
              f"{e.best_side:>7}"
              f"{e.best_edge:>+7.1%}"
              f"{e.best_ev_per_unit:>+7.2f}{flag}")
    print(f"\n★ = edge >= {min_edge:.0%} (your threshold)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="MLB pitcher-K prop finder")
    ap.add_argument("--date",  help="YYYY-MM-DD (default: today)")
    ap.add_argument("--lines", help="CSV of prop lines for EV calc")
    ap.add_argument("--min-edge", type=float, default=0.05,
                    help="Edge threshold to flag (default 0.05 = 5%%)")
    args = ap.parse_args()

    target = (datetime.strptime(args.date, "%Y-%m-%d").date()
              if args.date else date.today())

    print(f"Fetching MLB schedule for {target} ...")
    pitchers = get_todays_pitchers(target)
    if not pitchers:
        print("No probable pitchers found (off-day, or lineups not posted yet).")
        return 0
    print(f"Found {len(pitchers)} probable starters.")

    print("Pulling Statcast & projecting ...")
    projs = [project_pitcher(p, target) for p in pitchers]

    print_projections(projs)

    if args.lines:
        lines = load_lines(args.lines)
        evals = []
        for proj in projs:
            entry = lines.get(proj.pitcher.pitcher_id)
            if entry is None:
                continue
            line, over_odds, under_odds = entry
            evals.append(evaluate_prop(proj, line, over_odds, under_odds))
        if evals:
            print_evaluations(evals, args.min_edge)
        else:
            print("\nNo line entries matched today's pitcher_ids. "
                  "Use the IDs in the projections table above.")

    return 0


if __name__ == "__main__":
    sys.exit(main())

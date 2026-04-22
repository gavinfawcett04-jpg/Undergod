# MLB Prop Finder (v1 — Pitcher Strikeouts)

A Python tool that pulls today's MLB schedule, projects each starting pitcher's
strikeout total, and (optionally) compares the projection to sportsbook prop
lines to surface +EV over/under bets.

Built for fast iteration. Single file. No frameworks. Easy to refactor into a
package once the model proves out.

---

## Setup

```bash
git clone <your-repo>
cd mlb_prop_finder
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

**Projections only** (no lines required):
```bash
python mlb_prop_finder.py
```

**Specific date**:
```bash
python mlb_prop_finder.py --date 2026-04-22
```

**With lines for EV calculation**:
```bash
python mlb_prop_finder.py --lines lines.csv --min-edge 0.05
```

The `lines.csv` format (see `sample_lines.csv`):
```
pitcher_id,line,over_odds,under_odds
605483,6.5,-115,-105
```

`pitcher_id` is the MLBAM ID — printed in the projections table when you
run the projections-only command, so the workflow is:

1. Run projections.
2. Open DraftKings/FanDuel/whatever, find the K props for those pitchers.
3. Fill in `lines.csv` with the MLBAM IDs from step 1.
4. Re-run with `--lines lines.csv`.

---

## How the model works

For each probable starter:

```
expected_Ks  =  expected_BF  ×  blended_K%
blended_K%   =  pitcher_K%  ×  (opp_K%_vs_handedness  /  league_K%)
expected_BF  =  expected_IP  ×  4.3
```

- `pitcher_K%` — pulled from the pitcher's last 45 days of Statcast.
- `opp_K%_vs_handedness` — the opposing team's K% against pitchers of the
  starter's hand over the same window.
- `expected_IP` — rolling average IP from recent appearances.
- `league_K%` — constant, update each spring (`LEAGUE_K_PCT` at top of file).

Then strikeouts are modeled as Poisson(expected_Ks), so:

```
P(over L)  =  1  -  Poisson.cdf(ceil(L) - 1,  expected_Ks)
```

EV per 1-unit stake:
```
EV  =  P(win) × (decimal_odds - 1)  -  (1 - P(win))
edge =  P(model)  -  P(implied from odds)
```

---

## Honest caveats — read these

1. **Sportsbooks model this too.** Their lines are sharp. Edges of 1–3% are
   usually noise from your own modeling error. Treat ≥5% as "interesting,"
   not "free money."
2. **Probable pitchers get scratched.** Always re-check lineups within an hour
   of first pitch.
3. **The IP projection is crude.** A real model needs game state (blowout
   risk → early hook), bullpen rest, weather (cold = fewer Ks), umpire K-zone,
   and park factors.
4. **Poisson is an approximation.** It assumes equal mean and variance. Real
   K distributions are slightly overdispersed.
5. **Validate before risking money.** Run for 2–4 weeks, log every projection
   and the actual outcome, and check your model's calibration (do your "60%
   over" picks actually hit ~60% of the time?). If not, the model is broken.

---

## Roadmap

**v1.1** (next week)
- Cache the league-wide Statcast pull to disk so re-runs are instant.
- Park factor adjustment (Coors / Yankee Stadium vs pitcher's parks).
- Umpire K-zone factor (look up the home plate ump's career called-strike rate).

**v1.2**
- Batter props: hits, total bases, HR. Harder — variance is brutal on small N.
- Use `pybaseball.batting_stats_range` + opposing pitcher xwOBA-by-pitch-type.

**v2**
- Auto-pull lines from [The Odds API](https://the-odds-api.com/) so you skip
  the manual CSV step.
- Track every projection vs result in SQLite for calibration analysis.
- Calibration plot (matplotlib). This is the portfolio gold — recruiters love
  seeing a model that's been measured against reality.

**v3** (the actual GM-track move)
- Move beyond betting framing. Same projection engine becomes a pitcher
  matchup evaluator for lineup construction. That's a CSULB internship deliverable
  worth showing Joey.

---

## File map

```
mlb_prop_finder/
├── mlb_prop_finder.py    # everything (refactor to package once it's stable)
├── requirements.txt
├── sample_lines.csv
└── README.md
```

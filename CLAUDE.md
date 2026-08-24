# Scorecast

A football match-prediction site. A Dixon-Coles bivariate Poisson is fitted on real
league results, every published number is read off the resulting score matrix, and the
track record is measured by walking the model forward through matches it never saw.

Built in a Claude web session that could not make ordinary HTTP requests. Two planned
features are blocked by that limitation alone and drop straight in here — see
**Unblocked by running locally** below. That is the main reason this project moved.

## Build

```bash
python3 model/build.py    # fit, backtest, calibrate, predict -> data/payload.json
python3 model/site.py     # payload + src/ parts          -> index.html
```

Or `./build.sh` for both. Requires `numpy`, `scipy`, `scikit-learn`.

`index.html` and `src/02-data.js.html` are **generated** — never hand-edit them. Edit
`src/01-head.html` (CSS + shell), `src/03-ui.js.html` (atoms), `src/04-views.js.html`
(views), `src/05-app.js.html` (router), then re-run `model/site.py`.

## Layout

```
model/fit.py      Dixon-Coles likelihood, score matrix, markets, confidence, Platt calibration
model/build.py    loads data, fits per league, backtests, grades, emits data/payload.json
model/site.py     assembles index.html
model/names.py    team-name bridge between the two data sources + display labels
data/2025-26_*    real results, one row per match: date,home,away,homeGoals,awayGoals
data/cur_2627_*   real 2026/27 results so far, dd/mm/yyyy
data/fixtures_odds.csv  the fixture card with average bookmaker odds
src/              the site, in load order
```

## Data integrity — read this before touching the data

The result files were extracted through a summarising fetch tool that **silently drops
and sometimes fabricates rows**. Every rule below was learned by getting it wrong.

1. **Never ask a fetch tool for rows "which have a full-time score".** That phrasing makes
   the summariser treat 0-0 as *no score* and drop those rows. It produced five leagues
   with **zero goalless draws in 250+ matches** — a statistically impossible signature that
   biased goals-per-game, both-teams-to-score and over/under.
2. **Never ask it to "list every 0-0".** Those passes fabricate: one returned a 1-0 as
   goalless and invented two more that direct reads showed were 1-2 and 1-1.
3. **Never name the teams you are checking.** The summariser reorders home and away to
   echo the question back at you.
4. **Never ask for named columns from a wide CSV.** It drifts onto neighbouring columns —
   `AvgCH` came back holding `MaxCH` values, inconsistently, and differently across repeated
   calls. Fetch complete raw lines and index the columns yourself.
5. **Reconcile before trusting.** The check that actually works is per-team appearance
   counts: sum of appearances must equal `2 × rows`, and in a complete season every team
   must have the same total. A dropped row shows up as a team with a missing fixture.
   Sanity band for real football: 0-0 in 3-9% of matches, 2.6-3.2 goals per game,
   42-46% home wins, 22-27% draws.

La Liga and Serie A were **dropped from the product** because their extractions failed
these checks and could not be re-pulled in time. Re-adding them means re-fetching cleanly
and passing the reconciliation, not relaxing the standard.

## Modelling decisions worth keeping

- **Time decay**, half-life 200 days. This season's results therefore dominate the fit.
- **Ridge shrinkage** (`RIDGE = 0.60`) so a side with two matches played cannot take an
  extreme rating off one result. Well-observed teams barely move.
- **Cross-fitted Platt calibration** on the goal markets. The raw fit gets total goals right
  to within 0.03 of a goal but understates both-teams-to-score by 8 points, because
  near-independent Poissons over-weight one-sided scorelines. The correction is fitted on
  one half of the history and applied to the other, so the record page is not the correction
  grading itself.
- **Confidence** is a rescaling of the leading outcome probability, discounted when the
  inputs are thin. It is *not* the probability of being right, and the record page's
  calibration table is the direct check on it.
- **Nothing is tuned against the backtest.** Keep it that way.

## Honesty rules for this product

These are not decoration — they are the reason the site is worth anything.

- Publish the unflattering numbers. The model **fails its own always-home baseline in the
  Championship** (37.6%) and the site says so on the page.
- Never present a partial league table as a final one. Tables carry an "as of" date, the
  match count, and the games-in-hand spread.
- A gap against a bookmaker's price is **two estimates disagreeing, not a profit**. No
  staking advice, no claims about returns, 18+ notice stays.
- If data cannot be verified, drop the league rather than ship it.

## Unblocked by running locally

Both of these failed in the web sandbox purely because it had no ordinary HTTP access.
Claude Code does, so they are now straightforward.

**1. Complete the 2025/26 seasons.** `https://www.football-data.co.uk/mmz4281/2526/E0.csv`
(and `E1`, `D1`, `F1`, `N1`, plus `SP1`/`I1` to restore Spain and Italy) carries full
seasons — results, half-time scores, shots, cards and closing odds. The web tool truncated
these 132-column files at ~45 of 380 rows. `curl` them, parse with `csv`, and the current
files go from 51-91% of a season to 100%. Extract `Date,HomeTeam,AwayTeam,FTHG,FTAG` into
the existing `data/2025-26_*.csv` format and everything downstream just works.

**2. Grade against the market.** The same files carry `AvgCH,AvgCD,AvgCA` and
`AvgC>2.5,AvgC<2.5` — average closing odds for every historical match. With those, the
backtest can answer the question that matters: did the model beat the bookmakers' price,
not just an always-home baseline. Expect it to lose. Publish that.

**3. Keep it current.** `data/fixtures_odds.csv` is a snapshot. Re-fetch
`https://www.football-data.co.uk/fixtures.csv` and the 2026/27 season files
(`mmz4281/2627/E1.csv`, `N1.csv` — E0, D1 and F1 did not exist yet as those seasons had not
started) on a schedule, re-run the two build steps, and the site refreshes.

## Still open from the original brief

- The match card lost its evidence panel — recent form, head-to-head, home/away splits.
  All computable from `data/2025-26_*.csv` with no fetching at all. This is the best
  next task.
- The leaderboard has no week/month/all-time filter; there is only one backtest window.
- The empty states in `04-views.js.html` are unreachable, because every date on the card
  has fixtures and league chips only render when they have matches.

## Checks before publishing

- No horizontal overflow from 320px up, on every view.
- Fixture rows keyboard-reachable, visible focus ring, `prefers-reduced-motion` respected.
- Text contrast at least 4.5:1 — the `--faint` token is the one that nearly failed.
- Run the distribution sanity check on any data you touch before it reaches the model.

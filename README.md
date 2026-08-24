# Scorecast

A football match-prediction site. A Dixon-Coles bivariate Poisson fitted on real league
results; predicted scorelines, outcome probabilities, market tips priced against real
bookmaker odds, and a track record measured on matches the model never saw.

## Run it

```bash
pip install numpy scipy scikit-learn
./build.sh
```

That produces `index.html` — a single self-contained file. Open it directly, or:

```bash
python3 -m http.server 8000
# http://localhost:8000/index.html
```

To pull fresh fixtures, odds, and current-season results before building, run
`python3 model/refresh.py` instead — it re-fetches everything and rebuilds in one go.
A scheduled task can run this daily; see `model/refresh.py`'s own docstring.

## What's in it

| View | What it shows |
|---|---|
| Today | Fixtures with predicted scoreline, probabilities, confidence, and a top-pick-per-market line; this season's played matches graded |
| Tips | Selections ranked by gap against the bookmakers' de-vigged price, and by raw probability |
| Track record | 2,251 out-of-sample predictions: outcomes, exact scores, by market, by league, calibration |
| League tables | Real standings from the same results the model is fitted on, labelled with an "as of" date |
| Team strength | The fitted attack and defence ratings every prediction is built from |
| How it works | The model, the data, and what it cannot do |

## Leagues

**Priced** (fixture card, live bookmaker odds, tips) — Premier League, Championship,
Bundesliga, Ligue 1, Eredivisie, La Liga, Serie A.

**Backtest-only** (team strength and a real walk-forward backtest, but no daily fixture
card — the source carries no forward-odds feed for these) — MLS, Liga MX, Brazil's
Série A, Argentina's Liga Profesional, Japan's J1 League.

## Where the numbers come from

- **Full-season results** — [football-data.co.uk](https://www.football-data.co.uk/), one
  complete 2025/26 (or equivalent) season per league, reconciled before use.
- **Current-season results** — [openfootball](https://github.com/openfootball) for the
  European leagues it covers (fresher than football-data.co.uk early in a season),
  football-data.co.uk directly for the rest.
- **Fixtures and odds** — football-data.co.uk's live fixture feed, priced leagues only.

## Honest limits

- The model clears its own always-home baseline in 9 of 12 leagues, and falls short in
  La Liga, Brazil's Série A, and Argentina's Liga Profesional — published on the record
  page, not hidden.
- A gap against a bookmaker's price is two estimates disagreeing. It is not a profit, this
  is not betting advice, and no claim is made about returns. 18+.

`CLAUDE.md` has the full engineering context, the data-integrity rules, and the next tasks.

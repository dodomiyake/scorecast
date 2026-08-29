"""
Fit, backtest, predict, and emit the payload the site is built from.

Run:  python3 model/build.py
Out:  data/payload.json
"""

import csv, json, math, os, sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import names as N
from fit import (load, fit, rates, score_matrix, markets, confidence,
                 promoted_prior, second_tier_rank, Platt, MAXG, HALF_LIFE_DAYS,
                 PROMOTED_PCTL)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
TODAY = datetime.now().date()

# La Liga and Serie A were dropped once, for a bad extraction (0.0% and 0.4%
# goalless against a real 3-9%). Re-pulled cleanly with a direct curl straight
# into the reconciliation check this time — both passed (SP1: 3.9% goalless,
# I1: 9.5%) — so they're back.
FILES = ["2025-26_en.1.csv", "2025-26_en.2.csv", "2025-26_de.1.csv",
         "2025-26_fr.1.csv", "2025-26_nl.1.csv",
         "2025-26_sp.1.csv", "2025-26_it.1.csv",
         # backtest-only: MLS, Liga MX, Brazil, Argentina, J1 League — real
         # results, reconciled the same way, but no bookmaker-odds feed for
         # their upcoming fixtures, so they get a fit and a backtest and
         # nothing on the daily card. See names.py's ALIAS comment.
         "2025_usa1.csv", "2025-26_mex1.csv", "2025_bra1.csv",
         "2025_arg1.csv", "2025_jpn1.csv"]

RESULTS = {f: load(f) for f in FILES}
for f, r in RESULTS.items():
    print(f"  {f:22s} {len(r):4d} matches  {r[0][0]} -> {r[-1][0]}")

# ---- 2026/27 so far -------------------------------------------------------
# Real results from whatever this season has produced. A division whose
# current-season file doesn't exist yet — because it hasn't kicked off, or
# because the source hasn't published it yet — is auto-detected rather than
# hand-listed, so the refresh script writing a new file is all it takes for
# the next build to pick it up. No code change needed when a season opens.

CURRENT = {}
print("\n2026/27 results so far")
for div in N.DIV_PRIMARY:
    fname = N.CURRENT_STEM.get(div)
    path = os.path.join(DATA, fname) if fname else None
    if not fname or not os.path.exists(path) or os.path.getsize(path) == 0:
        print(f"  {div}: 0 matches — season starts on this card")
        continue
    rows = []
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) != 5:
                continue
            d = datetime.strptime(row[0], "%d/%m/%Y").date()
            h, a = N.historic_key(row[1].strip()), N.historic_key(row[2].strip())
            if not h or not a:
                raise SystemExit(f"unmapped team in {fname}: {row[1]} / {row[2]}")
            rows.append((d, h, a, int(row[3]), int(row[4])))
    if not rows:
        print(f"  {div}: 0 matches — season starts on this card")
        continue
    rows.sort(key=lambda r: r[0])
    CURRENT[div] = rows
    print(f"  {div}: {len(rows)} matches  {rows[0][0]} -> {rows[-1][0]}")

def training_for(div):
    """Last season plus whatever this season has produced, for one division."""
    return sorted(RESULTS[N.DIV_PRIMARY[div]] + CURRENT.get(div, []), key=lambda r: r[0])

# openfootball long name -> the label the interface should use, derived from
# the curated short-name map rather than guessed at with regexes
LONG_TO_DISPLAY = {}
for short_name, long_name in N.ALIAS.items():
    LONG_TO_DISPLAY[long_name] = N.display(short_name)

# ---------------------------------------------------------------- 1b. evidence
#
# Recent form, head-to-head and home/away splits for the match card. Pooled
# across every division's file plus this season's results so far — the same
# pool the fit itself trains on — and keyed on the openfootball long name, so
# a team promoted or relegated mid-history is still found under its one key.

ALL_MATCHES = [m for rows in RESULTS.values() for m in rows] + \
              [m for rows in CURRENT.values() for m in rows]


def team_matches(key):
    return sorted((m for m in ALL_MATCHES if key in (m[1], m[2])), key=lambda m: m[0])


def recent_form(key, n=5):
    out = []
    for d, h, a, hg, ag in team_matches(key)[-n:][::-1]:
        home = h == key
        gf, ga = (hg, ag) if home else (ag, hg)
        result = "W" if gf > ga else ("L" if gf < ga else "D")
        opp = a if home else h
        out.append({"date": d.isoformat(), "venue": "H" if home else "A",
                     "opp": LONG_TO_DISPLAY.get(opp, opp), "gf": gf, "ga": ga, "result": result})
    return out


def head_to_head(key_a, key_b, n=5):
    meetings = sorted((m for m in ALL_MATCHES if {m[1], m[2]} == {key_a, key_b}),
                       key=lambda m: m[0], reverse=True)[:n]
    return [{"date": d.isoformat(), "home": LONG_TO_DISPLAY.get(h, h),
              "away": LONG_TO_DISPLAY.get(a, a), "hg": hg, "ag": ag} for d, h, a, hg, ag in meetings]


def venue_split(key, venue):
    """Full record at one venue ('home' or 'away') across the pooled history."""
    matches = [m for m in ALL_MATCHES if m[1 if venue == "home" else 2] == key]
    w = d = l = gf = ga = 0
    for _, h, a, hg, ag in matches:
        f, against = (hg, ag) if venue == "home" else (ag, hg)
        gf += f; ga += against
        if f > against: w += 1
        elif f < against: l += 1
        else: d += 1
    p = len(matches)
    return {"played": p, "w": w, "d": d, "l": l, "gf": gf, "ga": ga,
            "ppg": round((3 * w + d) / p, 2) if p else None}

# ---------------------------------------------------------------- 1. fit


print("\nfitting per division on last season + this season")
FULL, PLAYED = {}, {}
for div, primary in N.DIV_PRIMARY.items():
    pool = training_for(div)
    p = fit(pool, TODAY)
    FULL[primary] = p
    counts = defaultdict(int)
    for _, h, a, _, _ in pool:
        counts[h] += 1; counts[a] += 1
    PLAYED[primary] = counts
    print(f"  {div:3s} {len(pool):4d} matches  home adv {math.exp(p['home']):.3f}x  "
          f"rho {p['rho']:+.3f}  {'ok' if p['converged'] else 'NOT CONVERGED'}")


def resolve(div, short_name):
    """(attack, defence, thin, provenance) for a team on its division's scale.

    thin: 0 = fitted directly, 1 = carried across a tier, 2 = no history.
    """
    primary = N.DIV_PRIMARY[div]
    pp = FULL[primary]
    key = N.historic_key(short_name)

    if key and key in pp["attack"]:
        n_played = PLAYED[primary][key]
        # how much evidence there actually is, rather than a promoted/not flag
        thin = 0 if n_played >= 20 else (1 if n_played >= 5 else 2)
        why = (f"fitted on {n_played} league matches" if n_played >= 20
               else f"only {n_played} match{'es' if n_played != 1 else ''} in the fit — "
                    f"shrunk toward the league average")
        return pp["attack"][key], pp["defence"][key], thin, why

    # look for the team one tier away
    for other in N.DIV_SOURCES[div][1:]:
        op = FULL.get(other)
        if key and op and key in op["attack"]:
            rank = second_tier_rank(key, op)
            if other.endswith("en.1.csv"):          # dropped down a tier
                q = min(0.92, 0.55 + 0.35 * (rank or 0.5))
                att, dfn = promoted_prior(pp, pctl=q)
                why = "relegated — prior from top-flight strength"
            else:                                    # came up a tier
                att, dfn = promoted_prior(pp, rank=rank)
                why = "promoted — prior from second-tier finish"
            return att, dfn, 1, why

    att, dfn = promoted_prior(pp)
    return att, dfn, 2, "no prior-season data — league-floor prior"


# ---------------------------------------------------------------- 2. backtest


def outcome(h, a):
    return "H" if h > a else ("A" if h < a else "D")


print("\nwalking the model forward through unseen matches")
BT = []
for f, r in RESULTS.items():
    dates = sorted({m[0] for m in r})
    warm = dates[int(len(dates) * 0.50)]
    blocks, cur = [], []
    for d in [x for x in dates if x > warm]:
        cur.append(d)
        if len(cur) == 2:
            blocks.append(cur); cur = []
    if cur:
        blocks.append(cur)

    for blk in blocks:
        train = [m for m in r if m[0] < blk[0]]
        test = [m for m in r if m[0] in blk]
        if len(train) < 60 or not test:
            continue
        p = fit(train, blk[0])
        if not p:
            continue
        for d, h, a, hg, ag in test:
            if h not in p["attack"] or a not in p["attack"]:
                continue
            lam, mu = rates(p, h, a)
            mk = markets(score_matrix(lam, mu, p["rho"]))
            BT.append({
                "file": f, "date": d.isoformat(), "home": h, "away": a,
                "hg": hg, "ag": ag, "act": outcome(hg, ag),
                "p": [mk["home"], mk["draw"], mk["away"]],
                "pred": mk["pred"], "conf": confidence(mk, 0),
                "o25": mk["o25"], "btts": mk["btts"],
                "cs": max(mk["cs_home"], mk["cs_away"]),
                "cs_side": "H" if mk["cs_home"] >= mk["cs_away"] else "A",
                "dc": max(mk["dc_1x"], mk["dc_12"], mk["dc_x2"]),
                "dc_which": max((mk["dc_1x"], "1X"), (mk["dc_12"], "12"), (mk["dc_x2"], "X2"))[1],
                "m1": mk["margin_1"], "m2": mk["margin_2p"],
                "raw_o25": mk["o25"], "raw_btts": mk["btts"],
            })
    print(f"  {f:22s} {sum(1 for b in BT if b['file']==f):4d} out-of-sample predictions")

print(f"  TOTAL {len(BT)} out-of-sample predictions")


def called(b):
    return ["H", "D", "A"][int(np.argmax(b["p"]))]


def hit_market(b):
    """Did each market call land? Returns dict market -> bool (or None if no call)."""
    hg, ag, tot = b["hg"], b["ag"], b["hg"] + b["ag"]
    out = {}
    out["outcome"] = called(b) == b["act"]
    out["exact"] = b["pred"][0] == hg and b["pred"][1] == ag
    out["ou25"] = (b["o25"] >= 0.5) == (tot > 2.5)
    out["btts"] = (b["btts"] >= 0.5) == (hg > 0 and ag > 0)
    cs_actual = (ag == 0) if b["cs_side"] == "H" else (hg == 0)
    out["cs"] = (b["cs"] >= 0.5) == cs_actual
    dc_actual = {"1X": b["act"] in "HD", "12": b["act"] in "HA", "X2": b["act"] in "DA"}[b["dc_which"]]
    out["dc"] = dc_actual
    margin = abs(hg - ag)
    out["margin"] = (margin >= 2) if b["m2"] >= b["m1"] else (margin <= 1)
    return out


# ---------------------------------------------------------------- 2b. calibration
#
# The fit gets total goals right (-0.03 of a goal across 676 matches) but
# understates both-teams-to-score by 8 points, because near-independent
# Poissons put too much weight on one-sided scorelines. A two-parameter
# logistic correction per market fixes the shape.
#
# To keep the published record honest the correction is CROSS-FITTED: the
# backtest is split in two by date, each half is calibrated by a model fitted
# only on the other half, and the reported figures use those out-of-fold
# values. The card uses a correction fitted on the whole history.

BT.sort(key=lambda b: b["date"])
mid = len(BT) // 2
folds = [(list(range(0, mid)), list(range(mid, len(BT)))),
         (list(range(mid, len(BT))), list(range(0, mid)))]

CAL_MARKETS = {
    "o25":  lambda b: 1 if (b["hg"] + b["ag"]) > 2.5 else 0,
    "btts": lambda b: 1 if (b["hg"] > 0 and b["ag"] > 0) else 0,
}

print("\ncalibrating goal markets (cross-fitted)")
for key, truth in CAL_MARKETS.items():
    raw = "raw_" + key
    for train_idx, test_idx in folds:
        cal = Platt().fit([BT[i][raw] for i in train_idx],
                          [truth(BT[i]) for i in train_idx])
        for i in test_idx:
            BT[i][key] = cal.apply(BT[i][raw])
    before = np.mean([b[raw] for b in BT]) * 100
    after = np.mean([b[key] for b in BT]) * 100
    actual = np.mean([truth(b) for b in BT]) * 100
    print(f"  {key:5s} mean {before:5.1f}% -> {after:5.1f}%   actual {actual:5.1f}%")

# correction used on the live card, fitted on everything
CAL = {}
for key, truth in CAL_MARKETS.items():
    CAL[key] = Platt().fit([b["raw_" + key] for b in BT], [truth(b) for b in BT])

HITS = [hit_market(b) for b in BT]


def rate(key, subset=None):
    idx = range(len(BT)) if subset is None else subset
    vals = [HITS[i][key] for i in idx if HITS[i][key] is not None]
    return (100.0 * sum(vals) / len(vals)) if vals else 0.0, len(vals)


overall = {k: rate(k) for k in ["outcome", "exact", "ou25", "btts", "cs", "dc", "margin"]}
home_baseline = 100.0 * sum(1 for b in BT if b["act"] == "H") / len(BT)
fav_baseline = 100.0 * sum(1 for b in BT if called(b) == b["act"] and max(b["p"]) > 0.5) / len(BT)
logloss = -sum(math.log(max(1e-9, b["p"]["HDA".index(b["act"])])) for b in BT) / len(BT)
brier = sum(sum((b["p"][k] - (1 if "HDA"[k] == b["act"] else 0)) ** 2 for k in range(3))
            for b in BT) / len(BT)

print("\n--- out-of-sample performance ---")
for k, (v, n) in overall.items():
    print(f"  {k:8s} {v:5.1f}%   n={n}")
print(f"  always-home baseline {home_baseline:.1f}%")
print(f"  log loss {logloss:.4f}   Brier {brier:.4f}")

# per league
BY_LEAGUE = {}
for f in FILES:
    idx = [i for i, b in enumerate(BT) if b["file"] == f]
    if not idx:
        continue
    BY_LEAGUE[f] = {
        "outcome": rate("outcome", idx)[0],
        "exact": rate("exact", idx)[0],
        "ou25": rate("ou25", idx)[0],
        "btts": rate("btts", idx)[0],
        "n": len(idx),
        "home_base": 100.0 * sum(1 for i in idx if BT[i]["act"] == "H") / len(idx),
    }

# calibration by confidence band, measured not asserted
BANDS = [(80, 101, "80–100"), (65, 80, "65–79"), (50, 65, "50–64"),
         (35, 50, "35–49"), (0, 35, "0–34")]
CALIB = []
for lo, hi, label in BANDS:
    idx = [i for i, b in enumerate(BT) if lo <= b["conf"] < hi]
    if not idx:
        CALIB.append({"band": label, "hit": None, "n": 0, "avg_top": None})
        continue
    CALIB.append({
        "band": label,
        "hit": rate("outcome", idx)[0],
        "n": len(idx),
        "avg_top": 100.0 * float(np.mean([max(BT[i]["p"]) for i in idx])),
    })

# weekly trend
weeks = defaultdict(list)
for i, b in enumerate(BT):
    d = datetime.fromisoformat(b["date"]).date()
    weeks[d - timedelta(days=d.weekday())].append(i)
TREND = []
for wk in sorted(weeks):
    idx = weeks[wk]
    if len(idx) < 6:
        continue
    TREND.append({
        "week": wk.isoformat(),
        "outcome": rate("outcome", idx)[0],
        "exact": rate("exact", idx)[0],
        "n": len(idx),
    })
print(f"  {len(TREND)} weeks with enough settled matches to plot")

# ---------------------------------------------------------------- 3. card


def devig(odds):
    """Bookmaker odds -> probabilities with the margin divided out proportionally."""
    inv = [1.0 / o for o in odds if o and o > 1.0]
    if len(inv) != len(odds):
        return None
    s = sum(inv)
    return [x / s for x in inv], s - 1.0


fixtures = []
with open(os.path.join(DATA, "fixtures_odds.csv")) as fh:
    for row in csv.DictReader(fh):
        if row["Div"] not in N.DIV_PRIMARY:
            continue                     # league not covered — see FILES above
        d = datetime.strptime(row["Date"], "%d/%m/%Y").date()
        if d < TODAY:
            continue                     # already kicked off; we hold no result for it
        fixtures.append((d, row))
fixtures.sort(key=lambda r: (r[0], r[1]["Time"]))
if fixtures:
    print(f"\n{len(fixtures)} fixtures on the card from {fixtures[0][0]} to {fixtures[-1][0]}")
else:
    print("\n0 fixtures on the card — none of the covered leagues have upcoming fixtures published yet")

CARD = []
for d, row in fixtures:
    div = row["Div"]
    hs, as_ = row["HomeTeam"].strip(), row["AwayTeam"].strip()
    ha, hd, h_thin, h_why = resolve(div, hs)
    aa, ad, a_thin, a_why = resolve(div, as_)
    pp = FULL[N.DIV_PRIMARY[div]]

    lam = math.exp(min(3.0, ha - ad + pp["home"]))
    mu = math.exp(min(3.0, aa - hd))
    mk = markets(score_matrix(lam, mu, pp["rho"]))

    # published goal markets are the calibrated ones; the pair stays summing to 1
    mk["o25"] = CAL["o25"].apply(mk["o25"]); mk["u25"] = 1.0 - mk["o25"]
    mk["btts"] = CAL["btts"].apply(mk["btts"]); mk["nobtts"] = 1.0 - mk["btts"]

    thin = max(h_thin, a_thin)
    conf = confidence(mk, thin)

    h_key, a_key = N.historic_key(hs), N.historic_key(as_)
    evidence = {
        "form": {"home": recent_form(h_key) if h_key else [],
                 "away": recent_form(a_key) if a_key else []},
        "h2h": head_to_head(h_key, a_key) if h_key and a_key else [],
        "split": {"home": venue_split(h_key, "home") if h_key else None,
                  "away": venue_split(a_key, "away") if a_key else None},
    }

    def f(x):
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    o = [f(row["AvgH"]), f(row["AvgD"]), f(row["AvgA"])]
    ou = [f(row["AvgO25"]), f(row["AvgU25"])]
    mkt = devig(o) if all(o) else None
    mkt_ou = devig(ou) if all(ou) else None

    edges = []
    if mkt:
        for k, i, label in [("home", 0, "home"), ("draw", 1, "draw"), ("away", 2, "away")]:
            edges.append({"m": "1x2", "sel": label, "model": mk[k],
                          "book": mkt[0][i], "odds": o[i],
                          "edge": mk[k] - mkt[0][i]})
    if mkt_ou:
        edges.append({"m": "ou25", "sel": "over", "model": mk["o25"],
                      "book": mkt_ou[0][0], "odds": ou[0], "edge": mk["o25"] - mkt_ou[0][0]})
        edges.append({"m": "ou25", "sel": "under", "model": mk["u25"],
                      "book": mkt_ou[0][1], "odds": ou[1], "edge": mk["u25"] - mkt_ou[0][1]})

    CARD.append({
        "id": f"{div}-{hs}-{as_}".replace(" ", "").replace("'", ""),
        "div": div, "date": d.isoformat(), "time": row["Time"],
        "home": N.display(hs), "away": N.display(as_),
        "homeShort": N.short(N.display(hs)), "awayShort": N.short(N.display(as_)),
        "lam": lam, "mu": mu,
        "pred": mk["pred"],
        "probs": [mk["home"], mk["draw"], mk["away"]],
        "mk": {k: mk[k] for k in ["o15", "u15", "o25", "u25", "o35", "u35", "btts", "nobtts",
                                  "dc_1x", "dc_12", "dc_x2", "cs_home", "cs_away",
                                  "margin_1", "margin_2p", "exp_home", "exp_away"]},
        "scores": mk["scores"],
        "conf": conf, "thin": thin,
        "why": {"home": h_why, "away": a_why},
        "odds": {"h": o[0], "d": o[1], "a": o[2], "o25": ou[0], "u25": ou[1]},
        "book": {"probs": mkt[0] if mkt else None, "overround": mkt[1] if mkt else None,
                 "ou": mkt_ou[0] if mkt_ou else None},
        "edges": edges,
        "evidence": evidence,
    })

print(f"  {sum(1 for c in CARD if c['thin']==0)} fixtures with both sides fitted, "
      f"{sum(1 for c in CARD if c['thin']==1)} carried across a tier, "
      f"{sum(1 for c in CARD if c['thin']==2)} on a floor prior")

# ---------------------------------------------------------------- 4. emit

# ---------------------------------------------------------------- 3a. this season, graded
#
# Every 2026/27 match that has been played, predicted by a model fitted only on
# what was known before it kicked off. Small sample, but it is the only fully
# current evidence that exists, and it is genuinely out of sample.

print("\ngrading this season's played matches")
GRADED = []
for div, rows in CURRENT.items():
    pool_all = training_for(div)
    for d, h, a, hg, ag in rows:
        train = [m for m in pool_all if m[0] < d]
        p = fit(train, d)
        if not p or h not in p["attack"] or a not in p["attack"]:
            continue
        lam, mu = rates(p, h, a)
        mk = markets(score_matrix(lam, mu, p["rho"]))
        mk["o25"] = CAL["o25"].apply(mk["o25"])
        mk["btts"] = CAL["btts"].apply(mk["btts"])
        probs = [mk["home"], mk["draw"], mk["away"]]
        called = "HDA"[int(np.argmax(probs))]
        actual = outcome(hg, ag)
        GRADED.append({
            "div": div, "date": d.isoformat(),
            "home": N.short(N.display(next(k for k, v in N.ALIAS.items() if v == h))),
            "away": N.short(N.display(next(k for k, v in N.ALIAS.items() if v == a))),
            "hg": hg, "ag": ag,
            "pred": mk["pred"], "probs": probs,
            "called": called, "actual": actual,
            "hit": called == actual,
            "exact": mk["pred"][0] == hg and mk["pred"][1] == ag,
            "o25": mk["o25"], "o25_hit": (mk["o25"] >= 0.5) == ((hg + ag) > 2.5),
        })
GRADED.sort(key=lambda g: g["date"], reverse=True)
if GRADED:
    hits = sum(1 for g in GRADED if g["hit"])
    ex = sum(1 for g in GRADED if g["exact"])
    ou = sum(1 for g in GRADED if g["o25_hit"])
    print(f"  {len(GRADED)} matches: outcome {100*hits/len(GRADED):.1f}%  "
          f"exact {100*ex/len(GRADED):.1f}%  over/under {100*ou/len(GRADED):.1f}%")

# ---------------------------------------------------------------- 3b. tables
#
# Real standings computed from the real results on disk. These files are
# genuine mid-season snapshots -- every team's played count reconciles and the
# spread is ordinary games-in-hand -- so the table is true AS OF the last date
# the source carries. It is not a final table and the site never calls it one.

ZONES = {  # (european places, relegation places) as normally applied
    "E0": (4, 3), "E1": (2, 3), "D1": (4, 2), "F1": (3, 2), "N1": (2, 2),
    "SP1": (4, 3), "I1": (4, 3),
    # MLS has no relegation and splits into conferences; Liga MX runs two
    # short tournaments with a play-in bracket, not a single table; Brazil,
    # Argentina and Japan do promote/relegate but the site isn't confident
    # enough in the exact cutoffs to highlight them. No zone shading rather
    # than a wrong one.
    "USA": (0, 0), "MEX": (0, 0), "BRA": (0, 0), "ARG": (0, 0), "JPN": (0, 0),
}

def build_table(file):
    rows = {}
    for d, h, a, hg, ag in RESULTS[file]:
        for t in (h, a):
            rows.setdefault(t, {"team": t, "p":0,"w":0,"d":0,"l":0,"gf":0,"ga":0,"pts":0,"form":[]})
        H, A = rows[h], rows[a]
        H["p"] += 1; A["p"] += 1
        H["gf"] += hg; H["ga"] += ag; A["gf"] += ag; A["ga"] += hg
        if hg > ag:
            H["w"] += 1; A["l"] += 1; H["pts"] += 3; H["form"].append("W"); A["form"].append("L")
        elif hg < ag:
            A["w"] += 1; H["l"] += 1; A["pts"] += 3; A["form"].append("W"); H["form"].append("L")
        else:
            H["d"] += 1; A["d"] += 1; H["pts"] += 1; A["pts"] += 1
            H["form"].append("D"); A["form"].append("D")
    out = sorted(rows.values(),
                 key=lambda r: (-r["pts"], -(r["gf"]-r["ga"]), -r["gf"], r["team"]))
    for r in out:
        r["form"] = r["form"][-5:]          # results are already in date order
    return out


TABLES = {}
for div, file in N.DIV_PRIMARY.items():
    res = RESULTS[file]
    last_date = res[-1][0]
    recent = [m for m in res if (last_date - m[0]).days <= 4]
    TABLES[div] = {
        "rows": build_table(file),
        "asOf": last_date.isoformat(),
        "matches": len(res),
        "euro": ZONES[div][0], "releg": ZONES[div][1],
        "recent": [{"date": m[0].isoformat(), "home": m[1], "away": m[2],
                    "hg": m[3], "ag": m[4]} for m in recent[-10:]],
    }
    played = sorted({r["p"] for r in TABLES[div]["rows"]})
    print(f"  {div:3s} table: {len(TABLES[div]['rows'])} teams, played {played[0]}-{played[-1]}, "
          f"as of {last_date}")

print("\nbuilding league tables")

STRENGTH = {}
for div, primary in N.DIV_PRIMARY.items():
    p = FULL[primary]
    rows = []
    for t in p["teams"]:
        rows.append({"team": t, "att": p["attack"][t], "dfn": p["defence"][t],
                     "played": PLAYED[primary][t]})
    rows.sort(key=lambda r: -(r["att"] + r["dfn"]))
    STRENGTH[div] = {"home_adv": math.exp(p["home"]), "rho": p["rho"],
                     "n": p["n_matches"], "rows": rows}

payload = {
    "built": TODAY.isoformat(),
    "names": LONG_TO_DISPLAY,
    "card": CARD,
    "backtest": {
        "n": len(BT),
        "overall": {k: {"rate": v[0], "n": v[1]} for k, v in overall.items()},
        "home_baseline": home_baseline,
        "logloss": logloss, "brier": brier,
        "by_league": BY_LEAGUE,
        "calibration": CALIB,
        "trend": TREND,
        "window": [min(b["date"] for b in BT), max(b["date"] for b in BT)],
    },
    "strength": STRENGTH,
    "tables": TABLES,
    "graded": GRADED,
    "season": {
        "started": {d: (len(CURRENT.get(d, [])) > 0) for d in N.DIV_PRIMARY},
        "played": {d: len(CURRENT.get(d, [])) for d in N.DIV_PRIMARY},
        "lastResult": (GRADED[0]["date"] if GRADED else None),
    },
    "calibration_fit": {k: CAL[k].params() for k in CAL},
    "meta": {
        "half_life_days": HALF_LIFE_DAYS,
        "promoted_pctl": PROMOTED_PCTL,
        "maxg": MAXG,
        "train": {f: {"n": len(r), "from": r[0][0].isoformat(), "to": r[-1][0].isoformat()}
                  for f, r in RESULTS.items()},
    },
}

out = os.path.join(DATA, "payload.json")
with open(out, "w") as fh:
    json.dump(payload, fh, separators=(",", ":"))
print(f"\nwrote {out}  ({os.path.getsize(out)/1024:.0f} KB)")

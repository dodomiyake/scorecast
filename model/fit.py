"""
Scorecast — the actual model.

A Dixon-Coles bivariate Poisson fitted by maximum likelihood, per league, on
real 2025/26 results. Every number the site publishes comes out of this file:
predicted scorelines, outcome and market probabilities, the confidence rating,
and the whole track record, which is measured by walking the model forward
through matches it was not fitted on.

Nothing here is tuned against the backtest. The decay constant and the
promoted-team prior were fixed before the first evaluation run.
"""

import csv, json, math, os, sys
from collections import defaultdict
from datetime import date, datetime

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import names as N

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

MAXG = 10                    # goals per side considered in the score matrix
HALF_LIFE_DAYS = 200.0       # weight of a result halves after this long
PROMOTED_PCTL = 0.20         # promoted sides start at this percentile of the league
RHO_BOUND = 0.20             # Dixon-Coles low-score correction, kept small
RIDGE = 0.60                 # shrinks thinly-observed teams toward the league mean

# ---------------------------------------------------------------- data


def load(fname):
    out = []
    with open(os.path.join(DATA, fname)) as fh:
        for row in csv.reader(fh):
            if len(row) != 5:
                continue
            try:
                d = datetime.strptime(row[0], "%Y-%m-%d").date()
                out.append((d, row[1].strip(), row[2].strip(), int(row[3]), int(row[4])))
            except ValueError:
                continue
    out.sort(key=lambda r: r[0])
    return out


# ---------------------------------------------------------------- model


def tau(x, y, lam, mu, rho):
    """Dixon-Coles adjustment: real football has more 0-0 and 1-1 than
    independent Poissons predict, and fewer 1-0 / 0-1."""
    if x == 0 and y == 0:
        return 1.0 - lam * mu * rho
    if x == 0 and y == 1:
        return 1.0 + lam * rho
    if x == 1 and y == 0:
        return 1.0 + mu * rho
    if x == 1 and y == 1:
        return 1.0 - rho
    return 1.0


def censored_00(matches, min_n=150, threshold=0.02):
    """True when goalless draws are missing from the sample.

    The upstream extraction dropped rows on exactly the condition
    homeGoals == 0 and awayGoals == 0, leaving leagues with a 0-0 rate of
    zero across 250+ matches. Real football runs 7-9%. Where that signature
    is present the sample is not a random subset, so the likelihood has to be
    conditioned on the match not having finished 0-0."""
    if len(matches) < min_n:
        return False
    z = sum(1 for m in matches if m[3] == 0 and m[4] == 0)
    return (z / len(matches)) < threshold


def fit(matches, ref_date, half_life=HALF_LIFE_DAYS, censor=None):
    """Maximum-likelihood attack/defence strengths, home advantage and rho.

    With censor=True the sample is treated as drawn from P(x, y | not 0-0),
    so each contribution is divided by 1 - P(0,0). Predictions still come
    from the full, uncensored distribution."""
    teams = sorted({m[1] for m in matches} | {m[2] for m in matches})
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    if n < 4 or len(matches) < n:
        return None

    if censor is None:
        censor = censored_00(matches)

    hi = np.array([idx[m[1]] for m in matches])
    ai = np.array([idx[m[2]] for m in matches])
    hg = np.array([m[3] for m in matches], dtype=float)
    ag = np.array([m[4] for m in matches], dtype=float)
    age = np.array([(ref_date - m[0]).days for m in matches], dtype=float)
    w = np.exp(-math.log(2) * np.maximum(age, 0) / half_life)

    # log-factorials for the Poisson normaliser
    lf_h = np.array([math.lgamma(g + 1) for g in hg])
    lf_a = np.array([math.lgamma(g + 1) for g in ag])

    # params: attack[0..n-1] (last one implied), defence[0..n-1], home, rho
    def unpack(p):
        att = np.empty(n)
        att[:-1] = p[: n - 1]
        att[-1] = -att[:-1].sum()          # sum-to-zero keeps it identifiable
        dfn = p[n - 1 : 2 * n - 1]
        return att, dfn, p[-2], p[-1]

    low_mask = (hg <= 1) & (ag <= 1)

    def nll(p):
        att, dfn, home, rho = unpack(p)
        log_lam = att[hi] - dfn[ai] + home
        log_mu = att[ai] - dfn[hi]
        log_lam = np.clip(log_lam, -4, 3)
        log_mu = np.clip(log_mu, -4, 3)
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        ll = hg * log_lam - lam - lf_h + ag * log_mu - mu - lf_a

        if low_mask.any():
            l_, m_ = lam[low_mask], mu[low_mask]
            x_, y_ = hg[low_mask], ag[low_mask]
            t = np.ones_like(l_)
            t = np.where((x_ == 0) & (y_ == 0), 1 - l_ * m_ * rho, t)
            t = np.where((x_ == 0) & (y_ == 1), 1 + l_ * rho, t)
            t = np.where((x_ == 1) & (y_ == 0), 1 + m_ * rho, t)
            t = np.where((x_ == 1) & (y_ == 1), 1 - rho, t)
            ll[low_mask] += np.log(np.clip(t, 1e-9, None))

        if censor:
            # P(0-0) under the current parameters, removed from the support
            p00 = (1.0 - lam * mu * rho) * np.exp(-lam - mu)
            ll -= np.log(np.clip(1.0 - p00, 1e-6, None))

        # Without this a side with two matches played can take an extreme
        # rating off a single result. Well-observed teams barely move.
        penalty = RIDGE * (float((att ** 2).sum()) + float((dfn ** 2).sum()))
        return -float((w * ll).sum()) + penalty

    p0 = np.concatenate([np.zeros(n - 1), np.full(n, 0.0), [0.25], [0.0]])
    bounds = [(-2.5, 2.5)] * (n - 1) + [(-2.5, 2.5)] * n + [(-0.5, 1.2), (-RHO_BOUND, RHO_BOUND)]
    res = minimize(nll, p0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 900, "maxfun": 90000})

    att, dfn, home, rho = unpack(res.x)
    return {
        "teams": teams,
        "attack": {t: float(att[i]) for t, i in idx.items()},
        "defence": {t: float(dfn[i]) for t, i in idx.items()},
        "home": float(home),
        "rho": float(rho),
        "n_matches": len(matches),
        "converged": bool(res.success),
        "censored": bool(censor),
    }


def rates(params, home_team, away_team):
    a, d = params["attack"], params["defence"]
    lam = math.exp(min(3.0, a[home_team] - d[away_team] + params["home"]))
    mu = math.exp(min(3.0, a[away_team] - d[home_team]))
    return lam, mu


def score_matrix(lam, mu, rho, maxg=MAXG):
    """Joint distribution over scorelines, Dixon-Coles adjusted."""
    ph = np.array([math.exp(-lam) * lam ** k / math.factorial(k) for k in range(maxg + 1)])
    pa = np.array([math.exp(-mu) * mu ** k / math.factorial(k) for k in range(maxg + 1)])
    m = np.outer(ph, pa)
    m[0, 0] *= 1 - lam * mu * rho
    m[0, 1] *= 1 + lam * rho
    m[1, 0] *= 1 + mu * rho
    m[1, 1] *= 1 - rho
    m = np.clip(m, 0, None)
    return m / m.sum()


# ---------------------------------------------------------------- markets


def markets(m):
    """Every published market, read straight off the score matrix."""
    n = m.shape[0]
    idx = np.arange(n)
    total = idx[:, None] + idx[None, :]
    diff = idx[:, None] - idx[None, :]

    home = float(np.tril(m, -1).sum())
    draw = float(np.trace(m))
    away = float(np.triu(m, 1).sum())

    out = {
        "home": home, "draw": draw, "away": away,
        "o15": float(m[total > 1.5].sum()), "u15": float(m[total < 1.5].sum()),
        "o25": float(m[total > 2.5].sum()), "u25": float(m[total < 2.5].sum()),
        "o35": float(m[total > 3.5].sum()), "u35": float(m[total < 3.5].sum()),
        "btts": float(m[1:, 1:].sum()), "nobtts": float(m[0, :].sum() + m[:, 0].sum() - m[0, 0]),
        "dc_1x": home + draw, "dc_12": home + away, "dc_x2": draw + away,
        "cs_home": float(m[:, 0].sum()), "cs_away": float(m[0, :].sum()),
        "margin_1": float(m[np.abs(diff) == 1].sum()),
        "margin_2p": float(m[np.abs(diff) >= 2].sum()),
        "exp_home": float((m.sum(axis=1) * idx).sum()),
        "exp_away": float((m.sum(axis=0) * idx).sum()),
    }

    flat = [((i, j), float(m[i, j])) for i in range(n) for j in range(n)]
    flat.sort(key=lambda kv: -kv[1])
    out["scores"] = [{"h": i, "a": j, "p": p} for (i, j), p in flat[:6]]
    out["pred"] = [flat[0][0][0], flat[0][0][1]]
    return out


def confidence(mk, thin):
    """How far the most likely outcome sits above a three-way coin toss,
    discounted when the inputs are thin.

    Rescaled so the bands actually spread: 1/3 (a completely open fixture)
    maps to 0 and 0.80 (about as one-sided as football gets) maps to 100.
    This is a monotone transform of the leading probability, so the record
    page's calibration table is a direct check on it."""
    top = max(mk["home"], mk["draw"], mk["away"])
    raw = (top - 1.0 / 3.0) / (0.80 - 1.0 / 3.0)
    raw = max(0.0, min(1.0, raw))
    raw *= {0: 1.0, 1: 0.84, 2: 0.70}[thin]
    return int(round(max(4, min(96, raw * 100))))


# ---------------------------------------------------------------- priors


def promoted_prior(params, pctl=PROMOTED_PCTL, rank=None):
    """A side with no top-flight history starts near the bottom of the league.
    `rank` in [0,1] (1 = ran away with the division below) nudges it up."""
    att = np.array(list(params["attack"].values()))
    dfn = np.array(list(params["defence"].values()))
    q = pctl if rank is None else min(0.45, pctl + 0.30 * rank)
    return float(np.quantile(att, q)), float(np.quantile(dfn, q))


def second_tier_rank(hist_key, lower_params):
    """Where a promoted side finished on the strength scale of the tier below."""
    if not lower_params or hist_key not in lower_params["attack"]:
        return None
    net = {t: lower_params["attack"][t] + lower_params["defence"][t]
           for t in lower_params["attack"]}
    vals = sorted(net.values())
    return vals.index(net[hist_key]) / max(1, len(vals) - 1)


# ---------------------------------------------------------------- calibration


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


class Platt:
    """Logistic recalibration for one binary market.

    A Dixon-Coles model gets the total number of goals right but, because it
    treats the two scorelines as near-independent, it concentrates probability
    on one-sided results and understates both-teams-to-score. That is a shape
    error, not a level error, and a two-parameter logistic correction fixes it
    without touching the underlying fit."""

    def __init__(self):
        self.a = 1.0
        self.b = 0.0
        self.fitted = False

    def fit(self, probs, outcomes):
        from sklearn.linear_model import LogisticRegression
        x = _logit(probs).reshape(-1, 1)
        y = np.asarray(outcomes, dtype=int)
        if len(set(y.tolist())) < 2:
            return self
        lr = LogisticRegression(C=1e6, solver="lbfgs")
        lr.fit(x, y)
        self.a = float(lr.coef_[0][0])
        self.b = float(lr.intercept_[0])
        self.fitted = True
        return self

    def apply(self, p):
        if not self.fitted:
            return float(p)
        z = self.a * _logit([p])[0] + self.b
        return float(1.0 / (1.0 + math.exp(-z)))

    def params(self):
        return {"a": self.a, "b": self.b, "fitted": self.fitted}

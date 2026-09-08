"""
Re-fetch the fixture card and every division's current-season results, then
rebuild the site.

Run:  python3 model/refresh.py
Out:  refreshed data/cur_*.csv + data/fixtures_odds.csv, then build.py +
      finalize_payload.py + site.py

What this does NOT do: touch the 2025/26 (or equivalent) full-season files
that the model is fitted on. Those are a one-time pull, verified once, and
this script doesn't re-derive that verification — see CLAUDE.md's data
integrity rules if a new full season ever needs pulling in.

Season paths that need a manual bump once a year, in August, when the new
season opens: OF_SEASON (openfootball) and FD_SEASON (football-data.co.uk's
mmz4281 archive). Everything else — which current-season file exists, which
division is priced, which season label counts as "current" for the
calendar-year leagues — is detected at run time, not hand-maintained.
"""

import csv
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import names as N

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
STATUS_PATH = os.path.join(DATA, "refresh_status.json")

# every cur_*.csv on disk is keyed on the SHORT (football-data.co.uk-style)
# name — that's what build.py's loader expects, converting to the long/
# historic key itself via N.historic_key(). openfootball gives long names
# directly, so those need translating back before writing.
LONG_TO_SHORT = {}
for _short, _long in N.ALIAS.items():
    LONG_TO_SHORT.setdefault(_long, _short)

# bump these each August when the new European season opens
OF_SEASON = "2026-27"     # openfootball's folder/file naming
FD_SEASON = "2627"        # football-data.co.uk's mmz4281 folder naming

UA = {"User-Agent": "Mozilla/5.0 (scorecast data refresh)"}
RETRYABLE_HTTP = {429, 500, 502, 503, 504}
RUN_AT = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _previous_status():
    try:
        with open(STATUS_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


PREVIOUS_STATUS = _previous_status()
REFRESH_STATUS = {"runAt": RUN_AT, "fixtures": {}}


def mark_fixtures(state, detail=None):
    previous = PREVIOUS_STATUS.get("fixtures", {})
    last_success = RUN_AT if state == "fresh" else previous.get("lastSuccessfulFetch")
    REFRESH_STATUS["fixtures"] = {
        "state": state,
        "lastSuccessfulFetch": last_success,
        "detail": detail,
    }


def write_refresh_status():
    os.makedirs(DATA, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as fh:
        json.dump(REFRESH_STATUS, fh, separators=(",", ":"))


def fetch(url, timeout=20, attempts=4):
    """Fetch text with bounded exponential backoff for transient failures.

    Provider-side 429/5xx responses and temporary network errors are retried.
    Non-transient HTTP failures (for example a genuine 404) are surfaced
    immediately so callers can keep their existing source-specific handling.
    """
    last_error = None
    for attempt in range(1, attempts + 1):
        req = urllib.request.Request(url, headers=UA)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8-sig")
        except urllib.error.HTTPError as e:
            last_error = e
            if e.code not in RETRYABLE_HTTP or attempt == attempts:
                raise
            retry_after = e.headers.get("Retry-After") if e.headers else None
            try:
                delay = max(1, min(30, int(retry_after))) if retry_after else min(8, 2 ** (attempt - 1))
            except ValueError:
                delay = min(8, 2 ** (attempt - 1))
            print(f"  transient HTTP {e.code}; retrying in {delay}s ({attempt}/{attempts})")
            time.sleep(delay)
        except (urllib.error.URLError, TimeoutError) as e:
            last_error = e
            if attempt == attempts:
                raise
            delay = min(8, 2 ** (attempt - 1))
            print(f"  transient network error; retrying in {delay}s ({attempt}/{attempts})")
            time.sleep(delay)
    raise last_error or RuntimeError(f"fetch failed: {url}")


def write_rows(path, rows):
    """rows: iterable of (date_str, home, away, hg, ag), date_str already dd/mm/yyyy."""
    with open(path, "w", newline="\n", encoding="utf-8") as fh:
        fh.write("\n".join(",".join([d, h, a, str(hg), str(ag)]) for d, h, a, hg, ag in rows))
        if rows:
            fh.write("\n")


def check_mapped(rows, label):
    """Every (date, home, away, hg, ag) row's team names — already SHORT
    names — must resolve via N.historic_key(). Returns (ok_rows, missing)
    rather than raising, so one bad name doesn't take down the whole
    refresh — it's reported and that division's file is left untouched."""
    ok, missing = [], set()
    for d, h, a, hg, ag in rows:
        hk, ak = N.historic_key(h), N.historic_key(a)
        if not hk:
            missing.add(h)
        if not ak:
            missing.add(a)
        if hk and ak:
            ok.append((d, h, a, hg, ag))
    if missing:
        print(f"  ! {label}: unmapped team name(s), not written — add to names.py ALIAS: {sorted(missing)}")
    return ok, missing


# ---------------------------------------------------------------- fixtures + odds

PRICED_DIVS = tuple(d for d in N.DIV_PRIMARY if d not in
                     ("USA", "MEX", "BRA", "ARG", "JPN"))


def refresh_fixtures_odds():
    print("fixtures + odds")
    path = os.path.join(DATA, "fixtures_odds.csv")
    cached = os.path.exists(path) and os.path.getsize(path) > 0
    try:
        text = fetch("https://www.football-data.co.uk/fixtures.csv")
    except Exception as e:
        if cached:
            detail = f"provider unavailable ({type(e).__name__}: {e}); using last-known-good fixtures"
            mark_fixtures("cached", detail)
            print(f"  ! {detail}")
            return False
        mark_fixtures("failed", f"provider unavailable and no cached fixture file ({e})")
        raise

    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if r.get("Div") not in PRICED_DIVS:
            continue
        if not r.get("AvgH") or not r.get("AvgD") or not r.get("AvgA"):
            continue
        rows.append({
            "Div": r["Div"], "Date": r["Date"], "Time": r["Time"],
            "HomeTeam": r["HomeTeam"], "AwayTeam": r["AwayTeam"],
            "AvgH": r["AvgH"], "AvgD": r["AvgD"], "AvgA": r["AvgA"],
            "AvgO25": r.get("Avg>2.5", ""), "AvgU25": r.get("Avg<2.5", ""),
        })
    if not rows:
        if cached:
            detail = "fixture pull was empty; using last-known-good fixtures"
            mark_fixtures("cached", detail)
            print(f"  ! {detail}")
            return False
        mark_fixtures("failed", "fixture pull was empty and no cached fixture file exists")
        raise RuntimeError("empty fixture pull and no cached fixtures_odds.csv")

    rows.sort(key=lambda r: (*reversed(r["Date"].split("/")), r["Time"]))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["Div", "Date", "Time", "HomeTeam", "AwayTeam",
                                            "AvgH", "AvgD", "AvgA", "AvgO25", "AvgU25"])
        w.writeheader()
        w.writerows(rows)
    by_div = {}
    for r in rows:
        by_div[r["Div"]] = by_div.get(r["Div"], 0) + 1
    mark_fixtures("fresh")
    print(f"  {len(rows)} fixtures  " + "  ".join(f"{d}:{n}" for d, n in sorted(by_div.items())))
    return True


# ---------------------------------------------------------------- openfootball (E0, E1, F1, N1)

OF_REPO = {
    "E0": "https://raw.githubusercontent.com/openfootball/england/master/{s}/1-premierleague.txt",
    "E1": "https://raw.githubusercontent.com/openfootball/england/master/{s}/2-championship.txt",
    "D1": "https://raw.githubusercontent.com/openfootball/deutschland/master/{s}/1-bundesliga.txt",
    "F1": "https://raw.githubusercontent.com/openfootball/europe/master/france/{s}_fr1.txt",
    "N1": "https://raw.githubusercontent.com/openfootball/europe/master/netherlands/{s}_nl1.txt",
}

MONTHS = {m: i + 1 for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}
DATE_RE = re.compile(r"^\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+(\w{3})\w*\s+(\d{1,2})(?:\s+(\d{4}))?\s*$")
MATCH_RE = re.compile(r"^\s*(?:(\d{1,2}:\d{2})\s+)?(.+?)\s+v\s+(.+?)(?:\s+(\d+)-(\d+)(?:\s*\([^)]*\))?)?\s*$")


def parse_openfootball(text):
    year, cur_date, out = None, None, []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#") or s.startswith("=") or s.startswith("▪"):
            m = re.search(r"(\d{4})", s)
            if m and year is None:
                year = int(m.group(1))
            continue
        dm = DATE_RE.match(line)
        if dm:
            _, mon, day, yr = dm.groups()
            if yr:
                year = int(yr)
            cur_date = (year, MONTHS[mon], int(day))
            continue
        mm = MATCH_RE.match(line)
        if mm and cur_date:
            _, home, away, hg, ag = mm.groups()
            if hg is not None and ag is not None:
                out.append((cur_date, home.strip(), away.strip(), int(hg), int(ag)))
    return out


def refresh_openfootball(div):
    url = OF_REPO[div].format(s=OF_SEASON)
    try:
        text = fetch(url)
    except Exception as e:
        print(f"  {div}: fetch failed ({e}) — leaving current file untouched")
        return
    matches = parse_openfootball(text)
    rows, unresolved = [], set()
    for (y, m, d), h, a, hg, ag in matches:
        hs, as_ = LONG_TO_SHORT.get(h), LONG_TO_SHORT.get(a)
        if not hs:
            unresolved.add(h)
        if not as_:
            unresolved.add(a)
        if hs and as_:
            rows.append((f"{d:02d}/{m:02d}/{y}", hs, as_, hg, ag))
    if unresolved:
        print(f"  ! {div}: openfootball name(s) with no short-name mapping, dropped: {sorted(unresolved)}")
    ok, _ = check_mapped(rows, div)
    if not ok:
        print(f"  {div}: 0 played matches yet")
        return
    write_rows(os.path.join(DATA, N.CURRENT_STEM[div]), ok)
    print(f"  {div}: {len(ok)} matches  {ok[0][0]} .. {ok[-1][0]}")


# ---------------------------------------------------------------- football-data.co.uk mmz4281 (D1, I1, SP1)

def refresh_fd_mmz(div):
    url = f"https://www.football-data.co.uk/mmz4281/{FD_SEASON}/{div}.csv"
    try:
        text = fetch(url)
    except urllib.error.HTTPError as e:
        if e.code in (300, 404):
            print(f"  {div}: season file not published yet")
        else:
            print(f"  {div}: fetch failed ({e})")
        return
    except Exception as e:
        print(f"  {div}: fetch failed ({e})")
        return
    if not text.lstrip().startswith("Div,") and "Div,Date" not in text[:200]:
        print(f"  {div}: season file not published yet")
        return
    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if not r.get("Date") or r.get("FTHG") in (None, ""):
            continue
        rows.append((r["Date"], r["HomeTeam"].strip(), r["AwayTeam"].strip(),
                      int(r["FTHG"]), int(r["FTAG"])))
    if not rows:
        print(f"  {div}: 0 played matches yet")
        return
    rows.sort(key=lambda r: tuple(reversed(r[0].split("/"))))
    ok, _ = check_mapped(rows, div)
    if not ok:
        return
    write_rows(os.path.join(DATA, N.CURRENT_STEM[div]), ok)
    print(f"  {div}: {len(ok)} matches  {ok[0][0]} .. {ok[-1][0]}")


# ---------------------------------------------------------------- football-data.co.uk "new" archive (global)

FD_NEW_CODE = {"USA": "USA", "MEX": "MEX", "BRA": "BRA", "ARG": "ARG", "JPN": "JPN"}
# div -> the fixed "last full season" file whose final date marks where
# "current season" begins. Season *labels* in this source aren't reliable
# for that cut — Japan's autumn-to-spring switch used two different labels
# ("2026" and "2026/2027") for the same run of matches — so the cutoff is
# a date, not a label, and current-season rows are just "everything after."
LAST_SEASON_FILE = {"USA": "2025_usa1.csv", "MEX": "2025-26_mex1.csv",
                     "BRA": "2025_bra1.csv", "ARG": "2025_arg1.csv", "JPN": "2025_jpn1.csv"}


def refresh_fd_new(div):
    code = FD_NEW_CODE[div]
    try:
        text = fetch(f"https://www.football-data.co.uk/new/{code}.csv")
    except Exception as e:
        print(f"  {div}: fetch failed ({e}) — leaving current file untouched")
        return

    with open(os.path.join(DATA, LAST_SEASON_FILE[div]), encoding="utf-8") as fh:
        cutoff = max(line.split(",", 1)[0] for line in fh if line.strip())  # "YYYY-MM-DD", sorts fine as text

    rows = []
    for r in csv.DictReader(io.StringIO(text)):
        if not r.get("Date") or r.get("HG") in (None, ""):
            continue
        dd, mm, yy = r["Date"].split("/")
        iso = f"{yy}-{mm}-{dd}"
        if iso > cutoff:
            rows.append((r["Date"], r["Home"].strip(), r["Away"].strip(), int(r["HG"]), int(r["AG"])))
    if not rows:
        print(f"  {div}: 0 played matches yet this season")
        return
    rows.sort(key=lambda r: tuple(reversed(r[0].split("/"))))
    ok, _ = check_mapped(rows, div)
    if not ok:
        return
    write_rows(os.path.join(DATA, N.CURRENT_STEM[div]), ok)
    print(f"  {div}: {len(ok)} matches  {ok[0][0]} .. {ok[-1][0]}")


# ---------------------------------------------------------------- main

def main():
    print(f"refresh run: {RUN_AT}\n")

    try:
        refresh_fixtures_odds()

        print("\ncurrent season — openfootball")
        for div in ("E0", "E1", "D1", "F1", "N1"):
            refresh_openfootball(div)

        print("\ncurrent season — football-data.co.uk mmz4281")
        for div in ("I1", "SP1"):
            refresh_fd_mmz(div)

        print("\ncurrent season — football-data.co.uk global archive")
        for div in ("USA", "MEX", "BRA", "ARG", "JPN"):
            refresh_fd_new(div)
    finally:
        write_refresh_status()

    print("\nrunning build.py + finalize_payload.py + site.py")
    subprocess.run([sys.executable, os.path.join(ROOT, "model", "build.py")], check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "model", "finalize_payload.py")], check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "model", "site.py")], check=True)


if __name__ == "__main__":
    main()

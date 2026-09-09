"""Production-safe dual-source daily refresh.

The bookmaker feed remains the preferred fixture source because it includes
odds. Before the normal refresh runs, this module seeds a fresh schedule from
OpenFootball. If football-data.co.uk is unavailable (or returns no genuinely
upcoming fixtures), the normal refresh keeps that OpenFootball schedule rather
than an old bookmaker cache. Bookmaker fields stay blank in fallback mode.

Run: python3 model/production_refresh.py
"""

import csv
import os
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import refresh as R

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_PATH = os.path.join(R.DATA, "fixtures_odds.csv")
FALLBACK_HORIZON_DAYS = 21
FIXTURE_FIELDS = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "AvgH", "AvgD", "AvgA", "AvgO25", "AvgU25",
]

# OpenFootball publishes the season schedule in the same Football.TXT files
# Scorecast already trusts for current results. These seven competitions are
# exactly the leagues that can appear on the daily prediction card.
OPENFOOTBALL_FIXTURES = {
    "E0": "https://raw.githubusercontent.com/openfootball/england/master/{s}/1-premierleague.txt",
    "E1": "https://raw.githubusercontent.com/openfootball/england/master/{s}/2-championship.txt",
    "D1": "https://raw.githubusercontent.com/openfootball/deutschland/master/{s}/1-bundesliga.txt",
    "F1": "https://raw.githubusercontent.com/openfootball/europe/master/france/{s}_fr1.txt",
    "N1": "https://raw.githubusercontent.com/openfootball/europe/master/netherlands/{s}_nl1.txt",
    "SP1": "https://raw.githubusercontent.com/openfootball/espana/master/{s}/1-liga.txt",
    "I1": "https://raw.githubusercontent.com/openfootball/italy/master/{s}/1-seriea.txt",
}

# The legacy refresh already uses OpenFootball for five leagues. Add Spain and
# Italy at runtime so their played results can stay current even while the
# football-data.co.uk season CSVs are unavailable.
OPENFOOTBALL_RESULTS_EXTRA = {
    "SP1": OPENFOOTBALL_FIXTURES["SP1"],
    "I1": OPENFOOTBALL_FIXTURES["I1"],
}


def parse_openfootball_schedule(text):
    """Return every scheduled match, including unplayed fixtures.

    Output tuples are (date, time, long_home, long_away, played). Football.TXT
    commonly prints one kick-off time followed by several matches at that same
    time, so the most recent time on a date is carried forward.
    """
    year = None
    cur_date = None
    carried_time = None
    out = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith(("#", "=", "▪")):
            if year is None:
                import re
                match = re.search(r"(\d{4})", stripped)
                if match:
                    year = int(match.group(1))
            continue

        dm = R.DATE_RE.match(line)
        if dm:
            _, mon, day, explicit_year = dm.groups()
            if explicit_year:
                year = int(explicit_year)
            if year is None:
                continue
            cur_date = date(year, R.MONTHS[mon], int(day))
            carried_time = None
            continue

        mm = R.MATCH_RE.match(line)
        if not mm or cur_date is None:
            continue

        kickoff, home, away, hg, ag = mm.groups()
        if kickoff:
            carried_time = kickoff
        kickoff = kickoff or carried_time or "TBC"
        out.append((cur_date, kickoff, home.strip(), away.strip(), hg is not None and ag is not None))

    return out


def build_openfootball_fixture_rows(fetcher=None, today=None, horizon_days=FALLBACK_HORIZON_DAYS,
                                     repos=None):
    """Build upcoming fixture rows with intentionally blank bookmaker odds."""
    fetcher = fetcher or R.fetch
    today = today or datetime.now(timezone.utc).date()
    repos = repos or OPENFOOTBALL_FIXTURES
    end = today + timedelta(days=horizon_days)
    collected = []
    unresolved = {}

    for div, template in repos.items():
        url = template.format(s=R.OF_SEASON)
        text = fetcher(url, attempts=2)
        missing = set()

        for match_date, kickoff, home, away, played in parse_openfootball_schedule(text):
            if played or match_date < today or match_date > end:
                continue
            home_short = R.LONG_TO_SHORT.get(home)
            away_short = R.LONG_TO_SHORT.get(away)
            if not home_short:
                missing.add(home)
            if not away_short:
                missing.add(away)
            if not home_short or not away_short:
                continue

            collected.append((match_date, {
                "Div": div,
                "Date": match_date.strftime("%d/%m/%Y"),
                "Time": kickoff,
                "HomeTeam": home_short,
                "AwayTeam": away_short,
                "AvgH": "",
                "AvgD": "",
                "AvgA": "",
                "AvgO25": "",
                "AvgU25": "",
            }))

        if missing:
            unresolved[div] = sorted(missing)

    collected.sort(key=lambda item: (item[0], item[1]["Time"], item[1]["Div"], item[1]["HomeTeam"]))
    rows = [row for _, row in collected]
    return rows, unresolved


def write_fixture_rows(rows, path=FIXTURE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIXTURE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def fixture_file_has_upcoming(path=FIXTURE_PATH, today=None):
    today = today or datetime.now(timezone.utc).date()
    try:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    if datetime.strptime(row.get("Date", ""), "%d/%m/%Y").date() >= today:
                        return True
                except ValueError:
                    continue
    except OSError:
        return False
    return False


def make_provider_aware_fetch(base_fetch):
    """Fail fast on the known flaky provider; keep normal retries elsewhere."""
    def wrapped(url, timeout=20, attempts=4):
        if "football-data.co.uk" in url:
            return base_fetch(url, timeout=min(timeout, 15), attempts=1)
        return base_fetch(url, timeout=timeout, attempts=attempts)
    return wrapped


def _rebuild_payload_metadata():
    # build.py has already used the fixture file. Only freshness metadata and
    # the generated HTML need another pass after the status is corrected.
    subprocess.run([sys.executable, os.path.join(ROOT, "model", "finalize_payload.py")], check=True)
    subprocess.run([sys.executable, os.path.join(ROOT, "model", "site.py")], check=True)


def main():
    print("preparing OpenFootball fixture fallback")
    fallback_rows = []
    fallback_error = None
    try:
        fallback_rows, unresolved = build_openfootball_fixture_rows()
        if not fallback_rows:
            raise RuntimeError("OpenFootball returned no mapped upcoming fixtures in the fallback window")
        write_fixture_rows(fallback_rows)
        by_div = {}
        for row in fallback_rows:
            by_div[row["Div"]] = by_div.get(row["Div"], 0) + 1
        print("  seeded %d upcoming fixtures  %s" % (
            len(fallback_rows),
            "  ".join(f"{div}:{count}" for div, count in sorted(by_div.items())),
        ))
        for div, names in sorted(unresolved.items()):
            print(f"  ! {div}: fallback skipped unmapped OpenFootball team name(s): {names}")
    except Exception as exc:
        fallback_error = exc
        print(f"  ! OpenFootball fallback unavailable: {type(exc).__name__}: {exc}")

    # Add OpenFootball result coverage for Spain and Italy before the normal
    # refresh. If the bookmaker-provider result CSVs work, they still win later.
    R.OF_REPO.update(OPENFOOTBALL_RESULTS_EXTRA)

    base_fetch = R.fetch
    R.fetch = make_provider_aware_fetch(base_fetch)

    print("\ncurrent season — OpenFootball extra coverage")
    for div in ("SP1", "I1"):
        R.refresh_openfootball(div)

    # Guard against a 200 response whose fixture body is itself stale. The
    # original refresh is allowed to try the odds feed; if it writes no future
    # fixtures, restore the fresh fallback before build.py sees the file.
    original_fixture_refresh = R.refresh_fixtures_odds

    def refresh_fixtures_with_freshness_guard():
        result = original_fixture_refresh()
        if result and not fixture_file_has_upcoming():
            if fallback_rows:
                write_fixture_rows(fallback_rows)
                R.mark_fixtures("cached", "bookmaker fixture feed contained no upcoming matches; using OpenFootball fallback")
                print("  ! bookmaker fixture feed was stale; restored OpenFootball fallback")
                return False
            raise RuntimeError("bookmaker fixture feed contained no upcoming matches and fallback was unavailable")
        return result

    R.refresh_fixtures_odds = refresh_fixtures_with_freshness_guard
    R.main()

    state = R.REFRESH_STATUS.get("fixtures", {}).get("state")
    if state == "cached":
        if not fallback_rows:
            raise RuntimeError(
                "football-data.co.uk did not provide current fixtures and OpenFootball fallback failed: "
                f"{fallback_error}"
            )
        R.REFRESH_STATUS["fixtures"] = {
            "state": "fallback",
            "source": "openfootball",
            "lastSuccessfulFetch": R.RUN_AT,
            "detail": (
                f"football-data.co.uk unavailable or stale; using {len(fallback_rows)} fresh "
                "OpenFootball fixtures without bookmaker odds"
            ),
        }
        R.write_refresh_status()
        _rebuild_payload_metadata()
        print("  fixture status: fallback/openfootball (fresh schedule, no bookmaker odds)")


if __name__ == "__main__":
    main()

import os
import sys
import tempfile
import unittest
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

import production_refresh as P


SAMPLE = """= English Premier League 2026/27
# Date Fri Aug 21 2026 - Sun May 30 2027
▪ Matchday 3
  Sun Sep 6
    16:30  Arsenal FC              v Chelsea FC               2-1 (1-1)
▪ Matchday 4
  Sat Sep 12
    15:00  Arsenal FC              v Chelsea FC
           Liverpool FC            v Fulham FC
    17:30  Everton FC              v Tottenham Hotspur FC
  Sun Sep 13
    16:30  Manchester United FC    v Manchester City FC
▪ Matchday 5
  Fri Oct 9
    20:00  Brentford FC            v Chelsea FC
"""


class OpenFootballScheduleTests(unittest.TestCase):
    def test_parser_keeps_unplayed_matches_and_carries_kickoff_time(self):
        matches = P.parse_openfootball_schedule(SAMPLE)
        upcoming = [m for m in matches if not m[4]]
        self.assertEqual(upcoming[0][0], date(2026, 9, 12))
        self.assertEqual(upcoming[0][1], "15:00")
        self.assertEqual(upcoming[1][1], "15:00")
        self.assertEqual(upcoming[2][1], "17:30")
        self.assertTrue(matches[0][4])

    def test_fallback_filters_window_and_leaves_bookmaker_odds_blank(self):
        def fake_fetch(url, timeout=20, attempts=4):
            return SAMPLE

        rows, unresolved = P.build_openfootball_fixture_rows(
            fetcher=fake_fetch,
            today=date(2026, 9, 9),
            horizon_days=7,
            repos={"E0": "https://example.test/{s}.txt"},
        )
        self.assertEqual(len(rows), 4)
        self.assertEqual(unresolved, {})
        self.assertEqual(rows[0]["Date"], "12/09/2026")
        self.assertEqual(rows[0]["AvgH"], "")
        self.assertEqual(rows[0]["AvgD"], "")
        self.assertEqual(rows[0]["AvgA"], "")

    def test_upcoming_guard_rejects_old_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "fixtures_odds.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,AvgO25,AvgU25\n")
                fh.write("E0,05/09/2026,15:00,Arsenal,Chelsea,2,3,4,,\n")
            self.assertFalse(P.fixture_file_has_upcoming(path, today=date(2026, 9, 9)))

    def test_football_data_fetch_fails_fast_but_other_sources_keep_retries(self):
        calls = []

        def base_fetch(url, timeout=20, attempts=4):
            calls.append((url, timeout, attempts))
            return "ok"

        fetch = P.make_provider_aware_fetch(base_fetch)
        fetch("https://www.football-data.co.uk/fixtures.csv", timeout=20, attempts=4)
        fetch("https://raw.githubusercontent.com/openfootball/england/master/x", timeout=20, attempts=4)
        self.assertEqual(calls[0][1:], (15, 1))
        self.assertEqual(calls[1][1:], (20, 4))


if __name__ == "__main__":
    unittest.main()

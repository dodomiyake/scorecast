import os
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "model"))

import refresh
import finalize_payload
import production_refresh


class FakeResponse:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self):
        return self.text.encode("utf-8")


class RefreshRetryTests(unittest.TestCase):
    def test_fetch_retries_transient_503_then_succeeds(self):
        err = urllib.error.HTTPError("https://example.test", 503, "busy", {}, None)
        with mock.patch.object(refresh.urllib.request, "urlopen", side_effect=[err, FakeResponse("ok")]) as urlopen, \
             mock.patch.object(refresh.time, "sleep") as sleep:
            self.assertEqual(refresh.fetch("https://example.test", attempts=2), "ok")
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once()

    def test_fetch_does_not_retry_non_transient_404(self):
        err = urllib.error.HTTPError("https://example.test", 404, "missing", {}, None)
        with mock.patch.object(refresh.urllib.request, "urlopen", side_effect=err) as urlopen, \
             mock.patch.object(refresh.time, "sleep") as sleep:
            with self.assertRaises(urllib.error.HTTPError):
                refresh.fetch("https://example.test", attempts=4)
            self.assertEqual(urlopen.call_count, 1)
            sleep.assert_not_called()

    def test_fixture_refresh_uses_cached_file_when_provider_is_down(self):
        with tempfile.TemporaryDirectory() as tmp:
            cached = os.path.join(tmp, "fixtures_odds.csv")
            original = "Div,Date,Time,HomeTeam,AwayTeam,AvgH,AvgD,AvgA,AvgO25,AvgU25\n"
            with open(cached, "w", encoding="utf-8") as fh:
                fh.write(original)
            with mock.patch.object(refresh, "DATA", tmp), \
                 mock.patch.object(refresh, "fetch", side_effect=urllib.error.HTTPError(
                     "https://www.football-data.co.uk/fixtures.csv", 503, "busy", {}, None
                 )):
                self.assertFalse(refresh.refresh_fixtures_odds())
            with open(cached, encoding="utf-8") as fh:
                self.assertEqual(fh.read(), original)


class CurrentTableTests(unittest.TestCase):
    def test_table_is_computed_only_from_supplied_current_matches(self):
        from datetime import date
        matches = [
            (date(2026, 8, 15), "Alpha", "Beta", 2, 0),
            (date(2026, 8, 22), "Beta", "Alpha", 1, 1),
        ]
        table = finalize_payload.build_table(matches)
        by_team = {row["team"]: row for row in table}
        self.assertEqual(by_team["Alpha"]["p"], 2)
        self.assertEqual(by_team["Alpha"]["pts"], 4)
        self.assertEqual(by_team["Beta"]["pts"], 1)
        self.assertEqual(by_team["Alpha"]["form"], ["W", "D"])


class LeagueExpansionTests(unittest.TestCase):
    def test_new_divisions_have_complete_registry_entries(self):
        expected = {
            "P1": ("2025-26_pt.1.csv", "cur_2627_P1.csv"),
            "SC0": ("2025-26_sco.1.csv", "cur_2627_SC0.csv"),
            "B1": ("2025-26_be.1.csv", "cur_2627_B1.csv"),
            "T1": ("2025-26_tr.1.csv", "cur_2627_T1.csv"),
        }
        for div, (history, current) in expected.items():
            self.assertEqual(refresh.N.DIV_PRIMARY[div], history)
            self.assertEqual(refresh.N.CURRENT_STEM[div], current)
            self.assertIn(div, refresh.N.DIV_META)

    def test_scotland_score_middle_layout_is_parsed(self):
        self.assertEqual(
            refresh.match_openfootball_line("  20:00   Dundee United  1-1 (1-0)  Rangers"),
            ("20:00", "Dundee United", "Rangers", "1", "1"),
        )
        self.assertEqual(
            refresh.match_openfootball_line("  15:00   Falkirk   v   Dundee"),
            ("15:00", "Falkirk", "Dundee", None, None),
        )

    def test_new_historical_training_files_are_complete(self):
        minimum_rows = {"P1": 300, "SC0": 220, "B1": 300, "T1": 300}
        for div, minimum in minimum_rows.items():
            path = os.path.join(refresh.DATA, refresh.N.DIV_PRIMARY[div])
            with open(path, encoding="utf-8") as fh:
                rows = [line for line in fh if line.strip()]
            self.assertGreaterEqual(len(rows), minimum, div)

    def test_promoted_team_aliases_exist(self):
        for short_name in (
            "Academico Viseu", "Maritimo", "St Johnstone",
            "Beveren", "Kortrijk", "Lommel SK",
            "Amedspor", "Corum", "Erzurumspor",
        ):
            self.assertIsNotNone(refresh.N.historic_key(short_name), short_name)


class FixtureFallbackTests(unittest.TestCase):
    def test_footballwebpages_parser_builds_belgium_and_turkey_rows(self):
        from datetime import date
        belgium = """
        <h3>Friday 9th October 2026</h3>
        <span>7.45pm</span><span>KSV Beveren</span><span>v</span><span>Lommel</span>
        """
        rows, missing = production_refresh.parse_footballwebpages_schedule(
            belgium, "B1",
            production_refresh.FOOTBALLWEBPAGES_FIXTURES["B1"]["aliases"],
            today=date(2026, 10, 4),
        )
        self.assertEqual(missing, [])
        self.assertEqual(rows[0]["Date"], "09/10/2026")
        self.assertEqual(rows[0]["Time"], "19:45")
        self.assertEqual(rows[0]["HomeTeam"], "Beveren")
        self.assertEqual(rows[0]["AwayTeam"], "Lommel SK")

        turkey = """
        <h3>Saturday 10th October 2026</h3>
        <span>5pm</span><span>Çaykur Rizespor</span><span>v</span><span>Fenerbahçe</span>
        """
        rows, missing = production_refresh.parse_footballwebpages_schedule(
            turkey, "T1",
            production_refresh.FOOTBALLWEBPAGES_FIXTURES["T1"]["aliases"],
            today=date(2026, 10, 4),
        )
        self.assertEqual(missing, [])
        self.assertEqual(rows[0]["Time"], "17:00")
        self.assertEqual(rows[0]["HomeTeam"], "Rizespor")
        self.assertEqual(rows[0]["AwayTeam"], "Fenerbahce")

    def test_fallback_merge_prefers_confirmed_time_over_tbc(self):
        base = {
            "Div": "B1", "Date": "09/10/2026", "HomeTeam": "Beveren",
            "AwayTeam": "Lommel SK", "AvgH": "", "AvgD": "", "AvgA": "",
            "AvgO25": "", "AvgU25": "",
        }
        rows = production_refresh.merge_fixture_rows(
            [dict(base, Time="TBC")],
            [dict(base, Time="19:45")],
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["Time"], "19:45")


if __name__ == "__main__":
    unittest.main()

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


if __name__ == "__main__":
    unittest.main()

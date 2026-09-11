"""
Assemble index.html from data/payload.json and the parts in src/.

Run:  python3 model/site.py       (after model/build.py)
Out:  index.html
"""

import json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
DATA = os.path.join(ROOT, "data")

# Division code -> how the interface labels it, and which result file backs it.
LEAGUES = {
    "E0": {"name": "Premier League", "country": "England",     "file": "2025-26_en.1.csv"},
    "E1": {"name": "Championship",   "country": "England",     "file": "2025-26_en.2.csv"},
    "D1": {"name": "Bundesliga",     "country": "Germany",     "file": "2025-26_de.1.csv"},
    "F1": {"name": "Ligue 1",        "country": "France",      "file": "2025-26_fr.1.csv"},
    "N1": {"name": "Eredivisie",     "country": "Netherlands", "file": "2025-26_nl.1.csv"},
    "SP1": {"name": "La Liga",       "country": "Spain",       "file": "2025-26_sp.1.csv"},
    "I1": {"name": "Serie A",        "country": "Italy",       "file": "2025-26_it.1.csv"},
    "USA": {"name": "MLS",              "country": "USA",       "file": "2025_usa1.csv"},
    "MEX": {"name": "Liga MX",          "country": "Mexico",    "file": "2025-26_mex1.csv"},
    "BRA": {"name": "Série A",          "country": "Brazil",    "file": "2025_bra1.csv"},
    "ARG": {"name": "Liga Profesional", "country": "Argentina", "file": "2025_arg1.csv"},
    "JPN": {"name": "J1 League",        "country": "Japan",     "file": "2025_jpn1.csv"},
}

PARTS = ["01-head.html", "01b-mobile.css.html", "02-data.js.html", "03-ui.js.html",
         "03b-date-fix.js.html", "04-views.js.html", "04b-empty-today.js.html",
         "05-app.js.html"]

# Written last: fills the left rail's footer from the payload rather than
# hardcoding counts that go stale the moment the model is refitted.
RAIL = """
<script>
(function(){
  var P = window.__P;
  var n = Object.keys(P.meta.train).reduce(function(a,k){ return a + P.meta.train[k].n; }, 0) +
          Object.keys(P.season.played).reduce(function(a,k){ return a + P.season.played[k]; }, 0);
  var a = document.getElementById("rail-a"), b = document.getElementById("rail-b");
  if (a) a.textContent = n.toLocaleString() + " matches fitted";
  if (b) b.textContent = Object.keys(window.__LEAGUES).length + " leagues modelled \\u00b7 " +
                         P.card.length + " priced";
})();
</script>
"""


def main():
    payload_path = os.path.join(DATA, "payload.json")
    if not os.path.exists(payload_path):
        sys.exit("data/payload.json is missing — run model/build.py first")

    with open(payload_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    # the data module is generated, never hand-edited
    with open(os.path.join(SRC, "02-data.js.html"), "w", encoding="utf-8") as fh:
        fh.write("\n<script>\nwindow.__P = " + json.dumps(payload, separators=(",", ":")) + ";\n")
        fh.write("window.__LEAGUES = " + json.dumps(LEAGUES, separators=(",", ":")) + ";\n</script>\n")

    out = []
    for part in PARTS:
        path = os.path.join(SRC, part)
        if not os.path.exists(path):
            sys.exit(f"missing source part: src/{part}")
        with open(path, encoding="utf-8") as fh:
            out.append(fh.read())
    out.append(RAIL)

    html = "".join(out)
    with open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)

    print(f"wrote index.html  ({len(html)/1024:.0f} KB)")
    print(f"  {len(payload['card'])} fixtures priced")
    print(f"  {payload['backtest']['n']} out-of-sample predictions in the record")
    print(f"  {len(payload['graded'])} current-season matches graded")


if __name__ == "__main__":
    main()

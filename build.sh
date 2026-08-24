#!/usr/bin/env bash
# Fit the model and assemble the site. Both steps are deterministic.
set -euo pipefail
cd "$(dirname "$0")"
python3 model/build.py
python3 model/site.py
echo
echo "open index.html in a browser, or serve it:"
echo "  python3 -m http.server 8000   ->  http://localhost:8000/index.html"

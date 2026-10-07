# Community submissions

One JSON file per measured model: `submissions/<github-handle>/<model-slug>.json`.
Files here are append-only and are shown on the dashboard in a separate
"community" table. They never enter the reference ranking, which only holds
runs measured on the reference host.

Create a file with `python3 scripts/reporting/submission.py make …` and check
it with `python3 scripts/reporting/submission.py validate`. The rules are in
[CONTRIBUTING.md](../CONTRIBUTING.md).

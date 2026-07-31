---
schema_version: 1
open_count: 1
waived_count: 0
fixed_count: 0
total_count: 1
last_updated: 2026-07-31T03:52:51.171Z
---

# Broken Windows Ledger

> Cross-phase defect register. `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 01 | deviation | notebooks/01_manifold_and_gate.ipynb |  | STAGE2_K=[5,10,15,30] is unevenly spaced (gaps 5,5,15); k=8/k=20 dropped by STAGE2_MAX_FITS=4, so the plateau run [10,15,30] is maximal in index space not k space (disclosed, not acted on, per 01-03-SUMMARY.md Known Limitations) | open |  | 2026-07-31T03:52:51.171Z |  |

````json
[
  {
    "id": 1,
    "kind": "deviation",
    "phase": "01",
    "file": "notebooks/01_manifold_and_gate.ipynb",
    "line": null,
    "description": "STAGE2_K=[5,10,15,30] is unevenly spaced (gaps 5,5,15); k=8/k=20 dropped by STAGE2_MAX_FITS=4, so the plateau run [10,15,30] is maximal in index space not k space (disclosed, not acted on, per 01-03-SUMMARY.md Known Limitations)",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-07-31T03:52:51.171Z",
    "resolved_at": null
  }
]
````

---
schema_version: 1
open_count: 2
waived_count: 0
fixed_count: 0
total_count: 2
last_updated: 2026-08-07T16:58:35.354Z
---

# Broken Windows Ledger

> Cross-phase defect register. `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 01 | deviation | notebooks/01_manifold_and_gate.ipynb |  | STAGE2_K=[5,10,15,30] is unevenly spaced (gaps 5,5,15); k=8/k=20 dropped by STAGE2_MAX_FITS=4, so the plateau run [10,15,30] is maximal in index space not k space (disclosed, not acted on, per 01-03-SUMMARY.md Known Limitations) | open |  | 2026-07-31T03:52:51.171Z |  |
| 2 | 02.4 | deviation | .planning/phases/02.4-topological-auto-encoder-validity-test-inserted/02.4-PREREGISTRATION.md |  | Known Limitation 2's 'paper's own minimum searched lambda' justification for LAMBDA_TOPO=0.1 is withdrawn (fifth fidelity gap: reconstruction term sums over features vs reference's mean, so EffDim's lambda is ~D times smaller in paper convention than stated). LAMBDA_TOPO unchanged, no re-fit; pre-registration doc itself must be corrected by plan 02.4-08. | open |  | 2026-08-07T16:58:35.354Z |  |

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
  },
  {
    "id": 2,
    "kind": "deviation",
    "phase": "02.4",
    "file": ".planning/phases/02.4-topological-auto-encoder-validity-test-inserted/02.4-PREREGISTRATION.md",
    "line": null,
    "description": "Known Limitation 2's 'paper's own minimum searched lambda' justification for LAMBDA_TOPO=0.1 is withdrawn (fifth fidelity gap: reconstruction term sums over features vs reference's mean, so EffDim's lambda is ~D times smaller in paper convention than stated). LAMBDA_TOPO unchanged, no re-fit; pre-registration doc itself must be corrected by plan 02.4-08.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-07T16:58:35.354Z",
    "resolved_at": null
  }
]
````

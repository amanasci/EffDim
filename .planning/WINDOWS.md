---
schema_version: 1
open_count: 3
waived_count: 0
fixed_count: 1
total_count: 4
last_updated: 2026-08-10T14:20:45.037Z
---

# Broken Windows Ledger

> Cross-phase defect register. `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 01 | deviation | notebooks/01_manifold_and_gate.ipynb |  | STAGE2_K=[5,10,15,30] is unevenly spaced (gaps 5,5,15); k=8/k=20 dropped by STAGE2_MAX_FITS=4, so the plateau run [10,15,30] is maximal in index space not k space (disclosed, not acted on, per 01-03-SUMMARY.md Known Limitations) | open |  | 2026-07-31T03:52:51.171Z |  |
| 2 | 02.4 | deviation | .planning/phases/02.4-topological-auto-encoder-validity-test-inserted/02.4-PREREGISTRATION.md |  | Known Limitation 2's 'paper's own minimum searched lambda' justification for LAMBDA_TOPO=0.1 is withdrawn (fifth fidelity gap: reconstruction term sums over features vs reference's mean, so EffDim's lambda is ~D times smaller in paper convention than stated). LAMBDA_TOPO unchanged, no re-fit; pre-registration doc itself must be corrected by plan 02.4-08. | fixed |  | 2026-08-07T16:58:35.354Z | 2026-08-07T17:14:42.235Z |
| 3 | 02.4 | deviation | notebooks/pu_manifold/topoae.py |  | Gap #5 (fidelity gap, found on re-audit during 02.4-07, corrected in prose only by 02.4-PREREGISTRATION-AMENDMENT-02.md): train_topoae's reconstruction term sums over ambient features (.sum(-1).mean()) where the reference implementation's nn.MSELoss() means over them, reparameterizing LAMBDA_TOPO by a factor of D. Recorded, not fixed -- closing it would change every sealed fit's training objective and requires a fresh pre-registration plus a full sixteen-fit re-run, not authorised by Amendment 2. | open |  | 2026-08-07T17:14:49.608Z |  |
| 4 | 02.6 | deviation | notebooks/pu_manifold/tests/test_decoder_curvature.py |  | plan must-have claimed plain_decoder_curvature batch-split results are exact torch.equal bit-identical; measured false at real (hidden=64x3) architecture scale (~7e-14, amplified by pullback-metric condition number ~470), also confirmed in sealed chart_curvature.chart_mean_curvature itself via a duck-typed decoder of matching width -- test corrected to atol=1e-9, determinism (same z twice) kept as exact torch.equal | open |  | 2026-08-10T14:20:45.037Z |  |

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
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-08-07T16:58:35.354Z",
    "resolved_at": "2026-08-07T17:14:42.235Z"
  },
  {
    "id": 3,
    "kind": "deviation",
    "phase": "02.4",
    "file": "notebooks/pu_manifold/topoae.py",
    "line": null,
    "description": "Gap #5 (fidelity gap, found on re-audit during 02.4-07, corrected in prose only by 02.4-PREREGISTRATION-AMENDMENT-02.md): train_topoae's reconstruction term sums over ambient features (.sum(-1).mean()) where the reference implementation's nn.MSELoss() means over them, reparameterizing LAMBDA_TOPO by a factor of D. Recorded, not fixed -- closing it would change every sealed fit's training objective and requires a fresh pre-registration plus a full sixteen-fit re-run, not authorised by Amendment 2.",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-07T17:14:49.608Z",
    "resolved_at": null
  },
  {
    "id": 4,
    "kind": "deviation",
    "phase": "02.6",
    "file": "notebooks/pu_manifold/tests/test_decoder_curvature.py",
    "line": null,
    "description": "plan must-have claimed plain_decoder_curvature batch-split results are exact torch.equal bit-identical; measured false at real (hidden=64x3) architecture scale (~7e-14, amplified by pullback-metric condition number ~470), also confirmed in sealed chart_curvature.chart_mean_curvature itself via a duck-typed decoder of matching width -- test corrected to atol=1e-9, determinism (same z twice) kept as exact torch.equal",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-08-10T14:20:45.037Z",
    "resolved_at": null
  }
]
````

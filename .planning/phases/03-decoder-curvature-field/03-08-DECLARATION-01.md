# 03-08 Declaration 1 — which grid is the phase's result

**Declared:** 2026-08-14T18:29:00Z
**Status:** one-way, declared before any PU grid number existed
**Scope:** applies to plan 03-08's nine-cell PU grid and everything downstream of it

---

## 1. The decision

The **CPU grid is the phase's result.** A GPU grid, if one is run, is an **independent replication**
— reported alongside, never substituted for the CPU grid, and never permitted to override or replace
the CPU selection of `n_charts`.

## 2. Why this had to be declared in advance

The Phase 3 device-support supplement (`03-07-SUPPLEMENT-01.md`) makes it possible to run the same
nine-cell grid on CPU or on CUDA. Because CUDA RNG differs from CPU RNG, the two are **different
draws, not a reproduction of one another** — every record carries its own `device` field for exactly
this reason.

That creates a hazard: two complete grids, and a free choice of which to report. Selecting the more
favourable of two grids after seeing both is the same forking-paths move that D-02's pre-registered
`0.65` floor exists to prevent, and this milestone already has a recorded instance of a criterion
shifting after an unfavourable result (`02.6-FINDINGS.md` §4). Computing both is legitimate;
**choosing between them after seeing them is not.**

## 3. Verification that this declaration precedes the data

At the moment of declaration:

```
$ wc -l < notebooks/.cache/03_curvature_field_pu.jsonl
/bin/bash: notebooks/.cache/03_curvature_field_pu.jsonl: No such file or directory
$ date -u
2026-08-14T18:29:00Z
```

The record file did not exist. **Zero** grid cells had completed. The CPU grid was launched minutes
earlier (PID 230191, `--device cpu --resume`, reverse mode) and had not yet appended its first cell.

Unlike `03-02-AMENDMENT-01.md` — which honestly disclosed in its §6 that it was written with partial
knowledge of the outcome — this declaration **is** blind. No PU grid value of any kind existed when
it was made.

## 4. Rationale for CPU as primary

1. **CPU is the reference device for this milestone.** Every prior measurement — the sealed 02.2
   architecture, the 02.5 curvature work, the Phase 3 Swiss roll gate at both n=3000 and n=12000,
   the reproduction anchor `rho_chart = -0.06041003026778113` — was made on CPU. Making the phase's
   headline PU result a different RNG draw on different hardware would break that continuity for no
   scientific gain.
2. **The GPU was only ever a budget optimisation.** Plan 03-07's timing probe projected ~5.6-5.7h
   against D-13's 5-hour envelope. That envelope is a soft planning estimate, not a correctness
   bound. A 5.6h CPU run resolves it at zero cost to the measurement.
3. **Reverse mode is the declared reference path.** The CPU grid runs `mode="reverse"`, unchanged,
   requiring no code modification and no justification. (`mode="forward"` is proved equal at
   `rtol=1e-9, atol=1e-12` and would have saved ~1.07h, but was not needed once the envelope stopped
   binding.)

## 5. What the replication is for, and what it is not for

**It is for:** an independent robustness check on different hardware and a different RNG draw. This
phase has already produced one result that reversed under changed conditions — `02.5-09`'s
monotone-in-charts-used relationship did not survive at adequate sample size (`03-02-SUMMARY.md`).
A second draw of the PU grid is genuinely informative about how stable the selection is.

**It is not for:** replacing the CPU selection, breaking a tie, or being reported as the result if it
happens to look better. If the GPU replication selects a different `n_charts` than the CPU grid, that
disagreement is **a finding to report**, not a conflict to resolve in favour of the nicer number.

## 6. Standing constraints, unchanged by this declaration

- Devices must not be mixed **within** a single grid — all nine cells plus the three D-12 controls
  share one device, or the three-seed spread blends two different draws. The runner's `--resume`
  guard enforces this.
- `PU_N_CHARTS_SWEEP = (4, 8, 16)` and the three `PU_SEEDS` remain as declared in plan 03-07, before
  any PU number existed.
- The selection rule declared in 03-07 is the rule applied, unchanged.
- Nothing measured on the Swiss roll selects or constrains any PU hyperparameter (D-06, D-11).

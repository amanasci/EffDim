# ML4PS 2026 compliance note

Target: 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026), NeurIPS 2026 workshop, initial submission
Official sources:
- https://ml4physicalsciences.github.io/2026/ (call for papers)
- https://ml4physicalsciences.github.io/2026/guidelines.html (submission guidelines)
- https://media.neurips.cc/Conferences/NeurIPS2026/Formatting_Instructions_For_NeurIPS_2026.zip (official style package)
Checked: 2026-09-10

Rules recorded from the guidelines page:
- Page limit: 4 pages, excluding references. Appendices discouraged; reviewers need not read them.
- Template: NeurIPS 2026 template mandatory. "Papers submitted without using this template will be desk rejected." No other modifications to the template.
- Footer must read exactly: "Submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026). Do not distribute."
- Review: double-blind (single-blind optional for the Evaluations & Benchmarks track). Outside that track, fully anonymized: names, code links, text, figures.
- Checklist: not required for workshop submissions.
- Deadline: Saturday 12 September 2026, 23:59 AoE. Notification 10 October 2026. Workshop 11 December 2026.
- Submission via OpenReview: https://openreview.net/group?id=ML4PS/2026/Workshop
- Tracks named on the call: Research, Evaluations & Datasets, Perspectives.
- Reviewing criteria: novelty, correctness, relevance, potential impact.

How this project meets them:
- `neurips_2026.sty` is the file from the official zip, byte-unchanged. `checklist.tex` and `neurips_2026.tex` from the zip are kept alongside for reference and are not included in the build.
- The footer string is set by redefining `\@noticestring` in the preamble of `main.tex`, not by editing the style file.
- Submission mode (no `final` option): author block anonymized, line numbers on.
- No author names, affiliations, repository links, or identifying paths appear in `main.tex`.
- Main text ends before the References heading on or before page 4 (verify with the page-count check below after every edit).
- Track: Research (assumed; confirm at submission).

Page-count check after compiling:
    pdftotext main.pdf - | grep -n "^References"   # and confirm which page it lands on
    pdfinfo main.pdf | grep Pages

Page-count status (2026-09-12, after the probe-facing split and the Eq. (2) correction):
- `check.sh` now builds with XeLaTeX + Liberation Serif (metric-compatible with Times New Roman; Adobe
  Times in the official build differs by under 1 %) and gates on that: main text must end on page 4.
  Current: PASS, main text ends on page 4 with about 4 lines to spare (`preview_times_metric.pdf`);
  References start on page 5. The Computer Modern fallback (`preview_cm_fallback.pdf`) is informational
  only: it overstates length by roughly 20 lines over four pages, not the 45 assumed earlier.
  MUST still be confirmed on Overleaf with the official style (Adobe Times) before submission.
  First cuts if it does not fit: Figure 1 to 0.48 textwidth; Table 1's CKA column; the Validation
  paragraph's cubic/ridge clause.
- Table 2 and Figure 1(b) are decoder-only and read from `notebooks/.cache/09_physics_probe_facing_split.jsonl`
  (sha256 09e869ff…7be65, pod run 2026-09-12, runner `09_physics_probe_facing_split_run.py`, labels from the
  hf_hub_download'ed shards, sha256 60f2f82e…); Figure 1(a) reads `09_fixture_probe_facing.jsonl` and
  `09_fixture_probe_decodability*.jsonl`; the sphere-term identification of panel (a) is from
  `09_fixture_probe_facing_split.jsonl`. The quadratic-chart estimator is one clause in Section 5, from
  `09_physics_probe_facing.jsonl` (sha256 aefe4474…1a4b31).
- Eq. (2) was corrected 2026-09-12 (missing `c s^2 tr Δ` cross term; now `c̄² + s²‖δ‖² + ½s⁴‖Δ‖²_F`),
  verified by Monte Carlo; no table value depended on the old form.

Open items before submission:
- Author affiliation and names are only needed for the camera-ready `final` build.
- Seven uncited bibliography entries still carry a `note` field asking for author lists (harmless with unsrtnat, which prints cited entries only); `lee2023curvature` and `acosta2022templates` were verified against arXiv 2026-09-12.
- Confirm the track choice on OpenReview.

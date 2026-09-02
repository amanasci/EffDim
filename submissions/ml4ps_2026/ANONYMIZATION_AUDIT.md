# ANONYMIZATION_AUDIT

Double-blind check of `submissions/ml4ps_2026/` sources and compiled `main.pdf`.

## Required footer

Exact string on page 1:

`Submitted to the 9th Workshop on Machine Learning and the Physical Sciences (ML4PS 2026). Do not distribute.`

Not present: any claim that ML4PS is an official NeurIPS workshop.

## Author / institution

| Check | Result |
|---|---|
| Author block | `Anonymous Author(s)` only |
| Acknowledgements | none |
| Grant / programme names | none |
| Employer, SOAR, EleutherAI | not present |
| First-person “our previous work” | not used |

## PDF metadata (`pdfinfo`)

- Title: paper title (content, not an identity)
- Author: empty
- Creator / Producer: empty (`\hypersetup` clears them)
- Subject / Keywords: empty

## Self-citation

`duraphe2025platonicuniverse` is cited in the third person as background on astronomical vision embeddings. The bibliography lists UniverseTBD and named coauthors because that is the public author list of the cited preprint. This submission does not claim authorship of that paper.

## Strings searched (sources + `pdftotext`)

`Angus`, `github.com`, `PlatonicUniverse`, `Eleuther`, `SOAR`, `/home/`, acknowledgements, usernames, local paths: **no hits in the compiled PDF** except the required workshop footer.

`sample_id` appears in the method as an object-alignment field name, not a filesystem path or account.

## Template internals

`neurips_2026.sty` contains historical maintainer emails (e.g. Garnett). That file is the unmodified official style and is not author-identifying for this paper. It was not edited.

## Code availability

No anonymous repository URL is fabricated. The manuscript states that code and frozen artifacts will be released after review.

## Residual risk

Reviewers who already know the Platonic Universe / UniverseTBD literature might guess a research lineage from the citation. That is ordinary bibliographic inference, not a deanonymizing acknowledgement or affiliation line.

# CITATION_AUDIT

All five bibliography entries were copied from `submissions/neurreps_2026/references.bib` and re-verified against primary records. No citation was added from memory. Unused keys: none (`main.bbl` contains exactly these five).

## Sentence-level support

| Key | Surrounding sentence | Does the cited work support it? |
|---|---|---|
| `alain2017probes` | Linear probes as a standard readout of whether a representation contains a scientific observable. | Yes. Alain & Bengio introduce linear classifier probes on frozen activations. |
| `duraphe2025platonicuniverse` | In astronomy this question is now asked of large vision embeddings of galaxies. | Yes. UniverseTBD et al. evaluate foundation-model embeddings of galaxies (third-person citation of a public preprint; not written as “our work”). |
| `huh2024platonic` | Background hypothesis that foundation models converge toward shared geometric structure. | Yes. Huh et al. state the Platonic representation hypothesis. |
| `dosovitskiy2021vit` | Frozen ViT-B galaxy embedding. | Yes. Dosovitskiy et al. introduce Vision Transformers, including ViT-B. |
| `donoho2003hessian` | Local tangent coordinates / quadratic charting tradition. | Yes. Donoho & Grimes develop Hessian eigenmaps from local quadratic fits. |

No first-person self-citation.

## Entry-by-entry verification

### `huh2024platonic`

- **Title:** Position: The Platonic Representation Hypothesis — matches ICML 2024 proceedings, [PMLR 235:20617–20642](https://proceedings.mlr.press/v235/huh24a.html). (The arXiv posting is titled without the “Position:” prefix; the bibliography correctly uses the proceedings title.)
- **Authors:** Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola — match.
- **Year:** 2024 — match.
- **Venue:** Proceedings of the 41st ICML, PMLR volume 235 — match.
- **arXiv:** 2405.07987 — match.

### `duraphe2025platonicuniverse`

- **Title:** The Platonic Universe: Do Foundation Models See the Same Sky? — matches [arXiv:2509.19453](https://arxiv.org/abs/2509.19453).
- **Authors:** UniverseTBD; Kshitij Duraphe; Michael J. Smith; Shashwat Sourav; John F. Wu — match the public author list / recommended bibtex.
- **Year:** 2025 — match.
- **Venue / ID:** arXiv:2509.19453, doi 10.48550/arXiv.2509.19453 — match.
- Not treated as an official NeurIPS ML4PS 2025 archival citation; the arXiv record is the verifiable primary.

### `dosovitskiy2021vit`

- **Title:** An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale — matches [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) and ICLR 2021.
- **Authors:** Dosovitskiy, Beyer, Kolesnikov, Weissenborn, Zhai, Unterthiner, Dehghani, Minderer, Heigold, Gelly, Uszkoreit, Houlsby — match (12 authors).
- **Year:** 2021 (conference); preprint 2020 — bibliography uses conference year, as in the NeurReps file.
- **Venue:** ICLR 2021 — match.

### `donoho2003hessian`

- **Title:** Hessian eigenmaps: Locally linear embedding techniques for high-dimensional data — matches PNAS 100(10):5591–5596 (2003), doi [10.1073/pnas.1031596100](https://doi.org/10.1073/pnas.1031596100).
- **Authors:** David L. Donoho, Carrie Grimes — match.
- **Year / venue:** 2003, PNAS — match.

### `alain2017probes`

- **Title:** Understanding Intermediate Layers Using Linear Classifier Probes — matches [arXiv:1610.01644](https://arxiv.org/abs/1610.01644) (title capitalization in the .bib is title case; arXiv uses sentence case).
- **Authors:** Guillaume Alain, Yoshua Bengio — match.
- **Year:** bibliography 2017 with note “ICLR 2017 Workshop”. The arXiv datestamp is 2016; the ICLR workshop presentation is 2017. This is the same choice as the verified NeurReps bibliography and is retained.
- **Identifier:** 1610.01644 — match.

## Removed / not added

- No fabricated entries.
- No extra NeurReps-only or internal technical reports.
- No GitHub URLs in the bibliography.

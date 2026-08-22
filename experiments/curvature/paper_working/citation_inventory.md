# Citation inventory

External claims only. Repository numbers live in `claim_provenance.md`.
Metadata checked against arXiv / publisher / DOI pages (2026-08-15), not memory.

---

## Representational convergence

### Platonic Representation Hypothesis

- **Claim / topic:** Independently trained models become more aligned with scale; PRH proposes convergence toward a shared statistical model of reality.
- **Paper:** Position: The Platonic Representation Hypothesis
- **Authors:** Minyoung Huh, Brian Cheung, Tongzhou Wang, Phillip Isola
- **Year:** 2024
- **Venue:** ICML 2024 (PMLR 235:20617–20642); arXiv:2405.07987
- **DOI / ID:** PMLR v235/huh24a; arXiv:2405.07987
- **BibTeX key:** `huh2024platonic`
- **Where cited:** Abstract framing; Introduction; Related Work §convergence; Discussion (original hypothesis, not our refinement)
- **Verified source:** https://proceedings.mlr.press/v235/huh24a.html ; https://arxiv.org/abs/2405.07987 ; project page https://phillipi.github.io/prh/

### Astronomy PRH / mKNN size scaling

- **Claim / topic:** Astronomy PRH setup; HSC / Legacy / DESI / JWST comparisons; mKNN methodology in this lineage; within-family size ladders; 28/33 positive crossmodal steps (\(p\approx 3\times 10^{-5}\)).
- **Paper:** The Platonic Universe: Do Foundation Models See the Same Sky?
- **Authors:** UniverseTBD / Kshitij Duraphe, Michael J. Smith, Shashwat Sourav, John F. Wu
- **Year:** 2025
- **Venue:** NeurIPS 2025 Workshop on Machine Learning and the Physical Sciences; arXiv:2509.19453
- **DOI / ID:** 10.48550/arXiv.2509.19453
- **BibTeX key:** `duraphe2025platonicuniverse`
- **Where cited:** Introduction; Related Work; Experimental setup (data, mKNN protocol, paper-style binomial test); Results (comparison of power / 28/33)
- **Verified source:** https://arxiv.org/abs/2509.19453 ; ADS 2025arXiv250919453U ; official GitHub bibtex on UniverseTBD/platonic-universe

### Null-calibrated / critical PRH

- **Claim / topic:** Representational similarity can be confounded by width/depth; permutation null-calibration of similarity metrics; after calibration, global spectral “convergence” largely disappears while local neighbourhood overlap (mKNN) remains; analytic chance baseline \(\mathbb{E}[\mathrm{mKNN}]=k/(n-1)\).
- **Paper:** Revisiting the Platonic Representation Hypothesis: An Aristotelian View
- **Authors:** Fabian Gröger, Shuo Wen, Maria Brbić
- **Year:** 2026
- **Venue:** arXiv:2602.14486
- **Editorial note (not in `references.bib`):** authors also list ICML 2026; we cite the arXiv record because PMLR pages were not independently confirmed here.
- **DOI / ID:** arXiv:2602.14486
- **BibTeX key:** `groger2026aristotelian`
- **Where cited:** Related Work §convergence; Discussion (chance-adjusted mKNN / nulls). Must acknowledge; do not present permutation/null calibration as our invention.
- **Verified source:** https://arxiv.org/abs/2602.14486 ; https://arxiv.org/html/2602.14486v1 ; project page https://brbiclab.epfl.ch/projects/aristotelian/

---

## Structured representations

### Sparse autoencoders / TopK SAE

- **Claim / topic:** What an SAE is; TopK / \(k\)-sparse autoencoders as the implementation family used in-repo.
- **Paper:** Scaling and evaluating sparse autoencoders
- **Authors:** Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, Jeffrey Wu
- **Year:** 2025 (ICLR); preprint 2024
- **Venue:** ICLR 2025; arXiv:2406.04093
- **DOI / ID:** arXiv:2406.04093
- **BibTeX key:** `gao2025sae`
- **Where cited:** Introduction; Related Work §structured; Methods (shared-basis SAE)
- **Verified source:** https://arxiv.org/abs/2406.04093 ; https://proceedings.iclr.cc/paper_files/paper/2025/file/42ef3308c230942d223c411adf182c88-Paper-Conference.pdf
- **Repo note:** In-repo `TopKSAE` follows this TopK SAE lineage (Gao et al. / OpenAI SAE), not a different published SAE variant.

### Cross-model SAE feature spaces

- **Claim / topic:** Prior evidence that independently trained SAE feature spaces can correspond / share geometry (in LLMs).
- **Paper:** Quantifying Feature Space Universality Across Large Language Models via Sparse Autoencoders
- **Authors:** Michael Lan, Philip Torr, Austin Meek, Ashkan Khakzar, David Krueger, Fazl Barez
- **Year:** 2024– (arXiv)
- **Editorial note (not in `references.bib`):** earlier arXiv versions used the title “Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models.”
- **Venue:** arXiv:2410.06981
- **DOI / ID:** arXiv:2410.06981
- **BibTeX key:** `lan2024saeuniversality`
- **Where cited:** Introduction (contrast: LLM feature-space correspondence, not astronomy or shared-basis Ridge maps); Methods is our protocol. Do not imply they studied astronomy or the Ridge shared-basis construction.
- **Verified source:** https://arxiv.org/html/2410.06981 (current title); v2 used the older title.

### Block-sparse featurizers

- **Claim / topic:** BSF method; block sparsity; concepts as low-dimensional subspaces/manifolds rather than single directions.
- **Paper:** Structuring Sparsity: Block-Sparse Featurizers Capture Visual Concept Manifolds
- **Authors:** Thomas Fel, Matthew Kowal, Mozes Jacobs, and collaborators (Goodfire et al.)
- **Year:** 2026
- **Venue:** arXiv:2606.25234
- **DOI / ID:** arXiv:2606.25234
- **BibTeX key:** `fel2026bsf`
- **Where cited:** Introduction; Related Work §structured; Methods (shared BSF). Distinguish Fel et al.’s method from our shared-basis / cross-modal application.
- **Verified source:** https://arxiv.org/abs/2606.25234 ; https://arxiv.org/html/2606.25234v1

---

## Geometry and alignment metrics

### Mutual \(k\)-NN as used in PRH

- **Claim / topic:** Definition and use of mutual nearest-neighbour overlap as a representational alignment metric.
- **Paper:** Huh et al. 2024 (primary PRH operationalisation); Duraphe et al. 2025 (astronomy application and size-ladder protocol)
- **BibTeX keys:** `huh2024platonic`, `duraphe2025platonicuniverse`
- **Where cited:** Background/Related Work; Experimental setup (equation); Methods
- **Verified source:** Huh Appendix A (mNN); Duraphe §2. Huh describes mNN as a variant of Park et al. (2024), Klabunde et al. (2023), Oron et al. (2017). We cite Huh/Duraphe as the sources actually used; we do not add those earlier variants unless we discuss them.
- **Not cited:** Chechik et al. 2010 (Duraphe cite it for MKNN; we could not verify that paper as the definition of this overlap statistic). Prefer Huh.

### CKA

- **Claim / topic:** Centered kernel alignment as a representational similarity metric (used in unpaired smoke).
- **Paper:** Similarity of Neural Network Representations Revisited
- **Authors:** Simon Kornblith, Mohammad Norouzi, Honglak Lee, Geoffrey Hinton
- **Year:** 2019
- **Venue:** ICML 2019, PMLR 97:3519–3529
- **DOI / ID:** arXiv:1905.00414
- **BibTeX key:** `kornblith2019cka`
- **Where cited:** unused in this draft (CKA appendix removed)
- **Verified source:** https://proceedings.mlr.press/v97/kornblith19a.html

### Unpaired universal embedding geometry

- **Claim / topic:** Unsupervised translation between embedding spaces without paired examples; universal latent geometry (vec2vec).
- **Paper:** Harnessing the Universal Geometry of Embeddings
- **Authors:** Rishi Jha, Collin Zhang, Vitaly Shmatikov, John X. Morris
- **Year:** 2025
- **Venue:** NeurIPS 2025; arXiv:2505.12540
- **DOI / ID:** arXiv:2505.12540
- **BibTeX key:** `jha2025vec2vec`
- **Where cited:** Introduction (probe framing); Related Work; Methods 4.4; Results 5.4. Experiment is inspired by this work, not a reproduction.
- **Verified source:** https://arxiv.org/abs/2505.12540 ; https://proceedings.neurips.cc/paper_files/paper/2025/hash/4175dee33d6145cb8f0323703d138a53-Abstract-Conference.html

### Chance-corrected agreement (analogy only)

- **Claim / topic:** Precedent for subtracting chance agreement and normalising by attainable excess agreement. Not “Cohen’s \(\kappa\) for mKNN.”
- **Paper:** A Coefficient of Agreement for Nominal Scales
- **Authors:** Jacob Cohen
- **Year:** 1960
- **Venue:** Educational and Psychological Measurement 20(1):37–46
- **DOI / ID:** 10.1177/001316446002000104
- **BibTeX key:** `cohen1960kappa`
- **Where cited:** unused in this draft (\(M_{\mathrm{adj}}\) / \(\kappa\) subsection removed)
- **Verified source:** https://journals.sagepub.com/doi/10.1177/001316446002000104

### Epiplexity (verified; not cited in this draft)

- **Claim / topic:** Computationally bounded / observer-relative accessible structure. Does **not** predict an efficiency-relative PRH.
- **Paper:** From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence
- **Authors:** Marc Finzi, Shikai Qiu, Yiding Jiang, Pavel Izmailov, J. Zico Kolter, Andrew Gordon Wilson
- **Year:** 2026
- **Venue:** arXiv:2601.03220
- **DOI / ID:** arXiv:2601.03220
- **BibTeX key:** `finzi2026epiplexity`
- **Where cited:** One Discussion sentence only (“reminiscent of”). Does not predict our result.
- **Verified source:** https://arxiv.org/abs/2601.03220

---

## Models (main Legacy↔HSC ladders)

| Family | Paper | Authors | Year | Venue / ID | Key | Verified |
|---|---|---|---|---|---|---|
| ViT | An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale | Dosovitskiy et al. | 2021 | ICLR 2021; arXiv:2010.11929 | `dosovitskiy2021vit` | arXiv abs; ICLR 2021 |
| ConvNeXtv2 | ConvNeXt V2: Co-Designing and Scaling ConvNets With Masked Autoencoders | Woo, Debnath, Hu, Chen, Liu, Kweon, Xie | 2023 | CVPR 2023, pp. 16133–16142 | `woo2023convnextv2` | openaccess.thecvf.com |
| DINOv2 | DINOv2: Learning Robust Visual Features without Supervision | Oquab et al. | 2024 | TMLR 2024; arXiv:2304.07193 | `oquab2024dinov2` | OpenReview TMLR; HAL |
| I-JEPA | Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture | Assran, Duval, Misra, Bojanowski, Vincent, Rabbat, LeCun, Ballas | 2023 | CVPR 2023 | `assran2023ijepa` | openaccess.thecvf.com (CVPR, not ICCV) |
| AstroPTv2 | AstroPT: Scaling Large Observation Models for Astronomy | Smith, Roberts, Angeloudi, Huertas-Company | 2024 | ICML 2024 AI4Science; arXiv:2405.14930 | `smith2024astropt` | arXiv:2405.14930 ; OpenReview aOLuuLxqav |
| CLIP | Learning Transferable Visual Models From Natural Language Supervision | Radford et al. | 2021 | ICML 2021, PMLR 139:8748–8763; arXiv:2103.00020 | `radford2021clip` | PMLR radford21a |
| DINOv3 | DINOv3 | Siméoni et al. | 2025 | arXiv:2508.10104 | `simeoni2025dinov3` | arXiv abs; Meta GitHub bibtex (author list shortened with “and others” in `references.bib`) |

Where cited: Experimental setup (family list). CLIP and DINOv3 are cited in Appendix A only.

---

## Astronomy surveys / datasets

| Dataset | Paper | Authors | Year | Venue / ID | Key | Verified |
|---|---|---|---|---|---|---|
| HSC | Hyper Suprime-Cam: System design and verification of image quality | Miyazaki et al. | 2018 | PASJ 70, S1 | `miyazaki2018hsc` | NAOJ instrument citation page; DOI 10.1093/pasj/psx063 ; ADS 2018PASJ...70S...1M |
| DESI Legacy Imaging Surveys | Overview of the DESI Legacy Imaging Surveys | Dey et al. | 2019 | AJ 157, 168 | `dey2019legacy` | DOI 10.3847/1538-3881/ab089d ; ADS 2019AJ....157..168D |

These are the survey citations used by Duraphe et al. for the official Legacy↔HSC embeddings we reuse. We do **not** invent a data-release number beyond what that paper and the official pair files specify.

JWST / DESI spectra / Specformer are in Duraphe et al. but **not** in our main Legacy-only ladders — do not cite as if we analysed them.

No separate “Smith42 dataset paper” beyond AstroPT / Platonic Universe.

---

## Intentionally not cited

- Repository artifacts, SAE/BSF training scripts — our methods/results, not literature (`claim_provenance.md`).
- Decoder-metric / tangent / curvature papers — out of scope for this draft.
- Extra SAE papers beyond Gao (definition) and Lan (universality).
- Chechik et al. 2010 — not verified as the source of this mKNN overlap statistic.
- Huh’s earlier mNN variants (Park, Klabunde, Oron) — not needed unless we discuss metric history.
- Finzi as predicting efficiency-relative PRH — it does not.

---

## Phrase audit (manuscript)

Phrases that required a citation and now have one:

| Phrase / claim | Citation |
|---|---|
| representations become increasingly aligned | Huh |
| shared statistical model of reality | Huh |
| previous astronomy work reports mKNN increasing with capacity | Duraphe |
| SAEs decompose activations into sparse features | Gao |
| BSF models concepts as low-dimensional blocks/manifolds | Fel |
| mKNN definition / paper-style adjacent-size test | Huh, Duraphe |
| CKA | Kornblith |
| unpaired embedding translation | Jha (appendix / Related Work) |
| chance-corrected agreement analogy | Cohen |
| permutation / null-calibrated similarity; scale confounding | Gröger |
| model families / surveys | model and survey keys above |

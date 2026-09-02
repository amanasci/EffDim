# Citation-sensitive novelty audit

## Strongly defensible

* Learned shared basis for cross-modal transfer between independently trained SAE (and BSF) dictionaries: Ridge map of one model's sparse codes into the other's basis, then held-out mKNN. Distinct from Gao (SAE as a tool), Lan (same-modality LLM feature-space correspondence), and Duraphe (dense mKNN). Do not claim we invented SAEs, BSFs, or SAE universality.
* Probe × scale interaction on official Legacy↔HSC ladders: shared-basis probes raise the level; \(D_R\) is small and mixed.
* Matched held-out dense / SAE / BSF comparison (same queries, gallery, \(k\)).
* Unpaired relational size-shape triangulation (ConvNeXt, DINOv2).
* Appendix: Physics same-object cross-architecture pairs also show a shared-basis **level** lift (10/10). Do not treat this as a scale result.

## Do not overclaim

* SAE / BSF / unpaired methods themselves.
* Probe-invariant scaling, a universal size law, or “all probes have the same scaling.”
* That BSF “does not scale” (raw 9/11 vs interaction near zero are different statements).
* Efficiency-relative PRH as a result.
* Chance-adjusted mKNN / Cohen \(\kappa\) as a contribution of this draft.
* CKA as an independent experiment (appendix removed).
* That Lan et al. studied astronomy or shared-basis Ridge maps.
* That the Physics appendix tests probe × scale, uses side1, or is the official Legacy ladder.

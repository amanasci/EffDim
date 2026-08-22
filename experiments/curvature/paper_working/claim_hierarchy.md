# Claim hierarchy

Every Abstract / Conclusion sentence must sit on one of these levels.
Numbers: `claim_provenance.md`.

---

## Demonstrated

1. SAE and BSF substantially increase matched held-out mKNN@10 over dense on all 16 rungs (mean lifts \(+0.0056\) / \(+0.0076\)).
2. Those lifts do not correlate strongly with \(\log_{10}P\) (Spearman \(+0.11\) / \(+0.13\), both n.s.).
3. Adjacent probe\(\times\)scale interactions \(D_R=\Delta M_R-\Delta M_{\mathrm{dense}}\) (11 within-family steps, matched holdout, \(k{=}10\)):
   - SAE: mean \(-5.2\times 10^{-4}\), median \(+4.9\times 10^{-4}\), \(6\) positive / \(5\) negative;
   - BSF: mean \(\approx 0\), median \(+1.2\times 10^{-4}\), \(6\) positive / \(5\) negative.
   Sign test vs zero is uninformative (\(p{=}1\)). We find **no systematic positive probe\(\times\)scale interaction**.
4. Unpaired relational recovery is non-monotonic on ConvNeXt and flat/slightly declining on DINOv2 (adjacent signs \(2/6\)).
5. Paper-style full-catalog dense is a mild \(8/11\) trend (\(p_1{=}0.113\)); continuity only.
6. Physics same-object cross-architecture pairs (appendix only): dense mean 0.153; SAE lift +0.043; BSF lift +0.068; 10/10 positive. **Level only.** Not a \(D_R\) / scale result. Protocol is `shared_best_cosine`, not side1.

---

## Supported interpretation

1. Changing probe changes alignment **level** substantially more reliably than it changes the **size response**.
2. **Scale does not necessarily imply shared representation:** small models already show transferable sparse structure in a shared basis; larger models do not systematically share more of it.
3. Weak/heterogeneous model-size dependence is not obviously an artifact of dense coordinates alone (structured paired probes and an unpaired relational probe agree on the lack of a clean size law).
4. BSF’s raw \(9/11\) adjacent signs do not contradict the paper: BSF can increase with size while \(D_{\mathrm{BSF}}\) stays mixed around zero. Do not say “nothing scales.”
5. The extra correspondence is specifically that of a **learned shared basis** between independently trained SAE/BSF dictionaries (modal transfer in a common sparse coordinate system), not of SAE/BSF methods in isolation. Distinguish from Lan (LLM feature-space universality) and Gao (SAE definition).

---

## Speculation

1. Raw parameter count is not the fundamental convergence variable.
2. Efficiency / compression pressure may be more relevant.

Do not claim a probe-invariant scaling law.
Do not rank unpaired mKNN against SAE/BSF.

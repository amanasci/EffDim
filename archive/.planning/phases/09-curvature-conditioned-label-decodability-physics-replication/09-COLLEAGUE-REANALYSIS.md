# 09-COLLEAGUE-REANALYSIS — numbers reproduced from `origin/curvature-experiments` on 2026-09-02

Source tables (both on that branch, 512 anchors, ViT-B Physics, k=2048, chart rank d=16):
`paper/curvature_neurreps/audit_outputs/multilabel_chart_screen/mag_r_desi/global_anchor_metrics.csv`,
`paper/curvature_neurreps/audit_outputs/adaptive_dataset_curvature_probe_audit/per_anchor_curvature_parity.csv`.

| statistic | value |
|---|---|
| raw rho(K_H_cross, r2_G) | -0.4124 (matches frozen) |
| controlled, 3 controls | -0.2405 (matches frozen) |
| controlled, log_knn_radius only | -0.2463 |
| rho(K_H_cross, log_knn_radius) | +0.765 |
| rho(r2_G, log_knn_radius) | -0.332 |
| radius as 10 / 20 bin dummies + label variance | -0.232 / -0.222 |
| mean within-radius-stratum rho, 10 / 20 strata | -0.216 / -0.194 |
| 07.1-style within-stratum permutation null, S=10 | band -0.103..+0.068, p = 0.0002 (5000 draws) |
| same, S=20 | band -0.103..+0.064, p = 0.0002 |
| split-half reliability R_H at d=16 | median 0.514, 42% of anchors < 0.5, none > 0.7 |
| rho(R_H, K_H_cross) | +0.779 |
| anchors with R_H > 0.5 (n=296) | raw -0.372, controlled -0.219 |
| d=12 (parity table) | raw -0.038, controlled +0.143, rho(K_H, log r) +0.495 |
| d=20 (parity table) | raw -0.392, controlled -0.233, rho(K_H, log r) +0.711 |

Scale rows from his `submission_validation/scale_sensitivity.csv` (n=128 anchors, not power-matched):
k=1024 controlled -0.027 at d=16; k=1536 controlled -0.080 (p 0.37); k=512 curvature reliability
negative. The association is present only at his largest k. Dataset is 16,384 rows on his host
(his hash-selected subset of the 86,471-row Physics test set), so k=2048 is one eighth of it.

Precautions he took that Phase 9 must match: OOF probe; MSE and SST checks beside R^2;
label-identity audit (`probe_label_alignment_failure`); Freedman–Lane rank permutation with FWER across d;
paired anchor bootstrap; sphere-radial term removed before the quadratic fit.

Precaution he did not take: no known-answer validation of the curvature estimator at D=768, k=2048,
d=16. Reliability gate is split-half R_H only; `06-FINDINGS.md` measured R_H = 0.990 beside rho = 0.469
against truth on the Swiss roll, so split halves cannot see a shared bias.

Script that produced the table:

```python
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
S="/tmp/claude-1000/-home-akagi-Documents-Projects-EffDim/c52068e2-2c20-41c1-b6fd-ff3e2883cabe/scratchpad"
a=pd.read_csv(f"{S}/anchors_d16.csv"); p=pd.read_csv(f"{S}/kh_parity.csv")
print("rows",len(a),"fields",a.field.unique())
x=a.K_H_cross.values; y=a.r2_G.values; r=a.log_knn_radius.values; v=a.local_label_variance.values; c=a.local_evaluation_count.values
def partial(x,y,Z):
    xr,yr=rankdata(x),rankdata(y); A=np.column_stack([np.ones(len(x))]+[rankdata(z) for z in Z])
    rx=xr-A@np.linalg.lstsq(A,xr,rcond=None)[0]; ry=yr-A@np.linalg.lstsq(A,yr,rcond=None)[0]
    return spearmanr(rx,ry).statistic
print("raw rho(KH,r2)",spearmanr(x,y).statistic)
print("ctl (3 controls)",partial(x,y,[r,v,c]))
print("ctl radius only",partial(x,y,[r]))
print("rho(KH,log_r)",spearmanr(x,r).statistic," rho(r2,log_r)",spearmanr(y,r).statistic," rho(KH,labelvar)",spearmanr(x,v).statistic)
# nonlinear density control: bin dummies on radius
for S_ in (10,20):
    q=pd.qcut(rankdata(r),S_,labels=False)
    D=[(q==j).astype(float) for j in range(S_)]
    print(f"ctl radius as {S_} bin dummies (+labelvar):",partial(x,y,D+[v]))
    # within-stratum spearman, weighted mean
    rs=[spearmanr(x[q==j],y[q==j]).statistic for j in range(S_)]
    print(f"  mean within-stratum rho ({S_} strata):",np.mean(rs)," per-stratum:",np.round(rs,2))
# density-stratified permutation null for the controlled partial (07.1 scheme: permute x and y independently within strata)
rng=np.random.default_rng(0)
obs=partial(x,y,[r,v,c])
for S_ in (10,20):
    q=pd.qcut(rankdata(r),S_,labels=False); null=[]
    for b in range(5000):
        xp=x.copy(); yp=y.copy()
        for j in range(S_):
            idx=np.where(q==j)[0]; xp[idx]=x[rng.permutation(idx)]; yp[idx]=y[rng.permutation(idx)]
        null.append(partial(xp,yp,[r,v,c]))
    null=np.array(null)
    print(f"stratified null S={S_}: obs {obs:.4f}, null 2.5/97.5% {np.quantile(null,0.025):.4f}/{np.quantile(null,0.975):.4f}, p={(1+np.sum(np.abs(null)>=abs(obs)))/(len(null)+1):.5f}")
# reliability at d=16
p16=p[p.d==16].set_index("sample_id").loc[a.sample_id]
print("R_H d16: median",np.median(p16.R_H_new)," frac<0.5",np.mean(p16.R_H_new<0.5)," frac<0",np.mean(p16.R_H_new<0))
print("KH parity check d16 match:",np.allclose(p16.K_H_new.values,x))
for thr in (0.5,0.7):
    m=(p16.R_H_new.values>thr)
    print(f"anchors R_H>{thr}: n={m.sum()} raw {spearmanr(x[m],y[m]).statistic:.3f} ctl {partial(x[m],y[m],[r[m],v[m],c[m]]):.3f}")
print("rho(R_H, log_r)",spearmanr(p16.R_H_new.values,r).statistic, " rho(R_H,KH)",spearmanr(p16.R_H_new.values,x).statistic)
# d=12 and d=20 from parity table
for d in (12,20):
    pd_=p[p.d==d].set_index("sample_id").loc[a.sample_id]; xd=pd_.K_H_new.values
    print(f"d={d}: raw {spearmanr(xd,y).statistic:.3f} ctl {partial(xd,y,[r,v,c]):.3f} rho(KH,log_r) {spearmanr(xd,r).statistic:.3f}")
```

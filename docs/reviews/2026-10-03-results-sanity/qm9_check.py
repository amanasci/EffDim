import numpy as np, json, glob
from scipy.stats import binomtest
root='/home/akagi/Documents/Projects/EffDim/notebooks/.cache/qm9/'
def indep(ov,thr):
    A=ov>thr; np.fill_diagonal(A,False); alive=np.ones(ov.shape[0],bool); keep=np.zeros(ov.shape[0],bool)
    while alive.any():
        deg=(A&alive[None,:]).sum(1); deg[~alive]=10**9
        i=int(np.argmin(deg)); keep[i]=True; alive[i]=False; alive[A[i]]=False
    return keep
encs=['chemberta_10m_mlm','chemberta_77m_mlm','chemberta_5m_mtr','chemberta_10m_mtr','chemberta_77m_mtr','molformer_xl','chemfm_1b','chemfm_3b']
for e in encs:
    z=np.load(root+f'arrays/scaling__{e}__cf.npz'); th=np.load(root+f'arrays/scaling__{e}__thin.npz')
    keep=indep(th['overlap'].astype(float),0.05)
    rows=[json.loads(l) for l in open(root+f'records/scaling__{e}__main_xfit.jsonl')]
    res={r['label']:r for r in rows if r.get('row')=='result'}
    for lab in ('gap','mu','alpha','cv'):
        c=z[f'{lab}:S_model:r2_curve']; m=np.isfinite(c[:,0])
        cr=z[f'{lab}:random_qmatched:r2_curve']
        cs=z[f'{lab}:S:r2_curve']
        help_=np.mean(c[m,4]>c[m,2]); hurt=np.mean(c[m,0]<c[m,2]); rh=np.mean(cr[m,4]>cr[m,2]); rhu=np.mean(cr[m,0]<cr[m,2])
        sh=np.mean(cs[m,4]>cs[m,2]); ts=np.median(z[f'{lab}:S_model:t_star'][m]); tS=np.median(z[f'{lab}:S:t_star'][m])
        mm=keep&m; kh=int((c[mm,4]>c[mm,2]).sum()); n=int(mm.sum())
        p=binomtest(kh,n,0.5,alternative='greater').pvalue
        col=res[lab]['columns']['hess_mismatch_emp']
        seal=col.get('sealed',col)
        print(f"{e:18s} {lab:5s} mm={seal.get('partial'):+.3f} p={seal.get('p'):.4f} help={help_:.2f} rand={rh:.2f} hurt={hurt:.2f} rhurt={rhu:.2f} S_help={sh:.2f} t*M={ts:.2f} t*S={tS:.2f} thin n={n} k={kh} p={p:.2g}")

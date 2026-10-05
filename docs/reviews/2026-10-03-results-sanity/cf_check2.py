import numpy as np, glob, json
root='/home/akagi/Documents/Projects/EffDim/notebooks/.cache/'
for f in sorted(glob.glob(root+'09_physics_normal_scaling_*_d16.npz')):
    z=np.load(f); enc=f.split('scaling_')[1].replace('_d16.npz','')
    for lab in ('mag_r','photo_z','smooth_fraction','stellar_mass'):
        dS=z[f'{lab}:S:dR2']; dM=z[f'{lab}:S_model:dR2']; m=np.isfinite(dS)&np.isfinite(dM)
        print(f"{enc:14s} {lab:16s} median dR2(t=1) S {np.median(dS[m]):+.4f}  S_model {np.median(dM[m]):+.4f}  med|diff| {np.median(np.abs(dS[m]-dM[m])):.4f}  frac S_model>S {np.mean(dM[m]>dS[m]):.2f}")
# held-out p05 at alpha=100 exact values
for f in sorted(glob.glob('/home/akagi/Documents/Projects/EffDim/curvature-experiment/results/review-robustness/records/*.jsonl')):
    for l in open(f):
        r=json.loads(l)
        if r.get('row')=='result':
            h=r['heldout']; print(r['encoder'],r['label'],r['alpha_mode'],'p05=%.5f med=%.5f'%(h['p05'],h['median']), 'ext mm=%.4f'%r['partials']['extended_controls']['hess_mismatch_emp']['partial'])

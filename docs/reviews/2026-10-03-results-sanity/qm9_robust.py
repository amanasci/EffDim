import json, glob, collections
root='/home/akagi/Documents/Projects/EffDim/notebooks/.cache/qm9/records/'
c=collections.defaultdict(lambda:[0,0])
for f in sorted(glob.glob(root+'*__robust.jsonl')):
    enc=f.split('scaling__')[1].split('__')[0]
    for l in open(f):
        r=json.loads(l)
        if r.get('row')=='guard': print(enc,'guard',r['mode'],r['passed'],r['max_abs_diff_split'],r['max_abs_diff_cf'])
        if r.get('row')!='result': continue
        k=(r['label'],r['alpha_mode']); b=r['bootstrap']['32']['hess_mismatch_emp']; ba=r['bootstrap']['32']['align_cos_tan']
        c[('mm',)+k][0]+= b['hi']<0; c[('mm',)+k][1]+=1
        c[('al',)+k][0]+= ba['lo']>0; c[('al',)+k][1]+=1
        s=r['surrogate']; cf=r['cf']
        print(f"{enc:18s} {r['label']:5s} {r['alpha_mode']:9s} a={r['alpha']:<7.3g} oof={r['global_oof_r2']:.3f} mm={r['partials']['published_controls']['hess_mismatch_emp']['partial']:+.2f} ext={r['partials']['extended_controls']['hess_mismatch_emp']['partial']:+.2f} CI=[{b['lo']:+.2f},{b['hi']:+.2f}] help={cf['S_model']['help']:.2f} rh={cf['random_qmatched']['help']:.2f} t*={cf['S_model']['t_star']:.2f} surr rho={s['spearman']:+.2f} gap={s['median_abs_diff']:.3f} held={r['heldout']['median']:+.3f}[{r['heldout']['p05']:+.3f}]")
for k in sorted(c): print(k, f'{c[k][0]} of {c[k][1]}')

import json, glob, collections
root='/home/akagi/Documents/Projects/EffDim/notebooks/.cache/scaling/records/'
cnt=collections.defaultdict(lambda:[0,0]); cnt_al=collections.defaultdict(lambda:[0,0]); thin=collections.defaultdict(lambda:[0,0])
perm=collections.defaultdict(lambda:[0,0]); hold=collections.defaultdict(lambda:[0,0])
for f in sorted(glob.glob(root+'*__robust.jsonl')):
    enc=f.split('scaling__')[1].split('__')[0]
    for l in open(f):
        r=json.loads(l)
        if r.get('row')!='result': continue
        key=(r['label'],r['alpha_mode'])
        b=r['bootstrap']['32']['hess_mismatch_emp']; ba=r['bootstrap']['32']['align_cos_tan']
        cnt[key][0]+= b['excludes_zero'] and b['hi']<0; cnt[key][1]+=1
        cnt_al[key][0]+= ba['excludes_zero'] and ba['lo']>0; cnt_al[key][1]+=1
        t=r['thinned']['hess_mismatch_emp']; thin[key][0]+= (t['partial']<0 and t['p']<0.05); thin[key][1]+=1
        pp=r['partials']['published_controls']['hess_mismatch_emp']; perm[key][0]+= (pp['partial']<0 and pp['p']<0.05); perm[key][1]+=1
        h=r['heldout']; hold[key][0]+= h['p05']>0; hold[key][1]+=1
        if r['alpha_mode']=='tuned': print(enc, r['label'], 'alpha*', r['alpha'], 'oof', round(r['global_oof_r2'],3), 'cf t*', round(r['cf']['S_model']['t_star'],2), 'help',round(r['cf']['S_model']['help'],2),'rhelp',round(r['cf']['random_qmatched']['help'],2),'rhurt',round(r['cf']['random_qmatched']['hurt'],2))
for name,c in (('mismatch boot32 excl0 (neg)',cnt),('align boot32 excl0 (pos)',cnt_al),('mismatch thinned neg p<.05',thin),('mismatch perm neg p<.05',perm),('heldout p05>0',hold)):
    print(name)
    for k in sorted(c): print('  ',k,f'{c[k][0]} of {c[k][1]}')

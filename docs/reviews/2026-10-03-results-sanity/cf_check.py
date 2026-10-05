import numpy as np, glob
root='/home/akagi/Documents/Projects/EffDim/notebooks/.cache/'
for f in sorted(glob.glob(root+'09_physics_normal_scaling_*_d16.npz')):
    z=np.load(f); enc=f.split('scaling_')[1].replace('_d16.npz','')
    for lab in ('mag_r','photo_z','smooth_fraction','stellar_mass'):
        out=[]
        for v in ('S_model','S','random_qmatched'):
            c=z[f'{lab}:{v}:r2_curve']; ts=z[f'{lab}:{v}:t_star']
            m=np.isfinite(ts)
            help_=np.mean(c[m,4]>c[m,2]); hurt=np.mean(c[m,0]<c[m,2])
            # identity checks
            id_help=np.mean((ts[m]>0.5)==(c[m,4]>c[m,2])); id_hurt=np.mean((ts[m]>-0.5)==(c[m,0]<c[m,2]))
            out.append(f"{v}: help {help_:.3f} hurt {hurt:.3f} t*med {np.median(ts[m]):.2f} [idH {id_help:.3f} idU {id_hurt:.3f}]")
        print(f"{enc:14s} {lab:16s} "+' | '.join(out))

import json, glob, numpy as np
C = "/home/akagi/Documents/Projects/EffDim/notebooks/.cache/"
def rows(f): return [json.loads(l) for l in open(f) if l.strip()]
LAB = ("mag_r", "photo_z", "smooth_fraction", "stellar_mass")
sp = rows(C + "09_physics_probe_facing_split.jsonl")
print("== published split record (ViT-B): in-sample multiscale partials, emp vs dec vs |Hess y| vs align_tan vs align_full")
for r in sp:
    if r.get("row") != "result": continue
    c = r["columns"]; f = lambda k: c[k]["multiscale"]["partial"]
    print(r["d"], f"{r['label']:16s} emp {f('hess_mismatch_emp'):+.3f} dec {f('hess_mismatch_dec'):+.3f} |Hy| {f('hess_label'):+.3f} "
          f"al_tan {f('align_cos_tan'):+.3f} al_full {f('align_cos_full'):+.3f} pf_full {f('pf_full'):+.3f} pf_tan {f('pf_tan'):+.3f} pf_rad {f('pf_rad'):+.3f} "
          f"| checks II_rad+g rel {r['checks']['II_rad_vs_minus_g_max_rel']:.1e} JTx {r['checks']['JT_xhat_max']:.1e} cos(dec,emp) {r['checks']['cos_dec_vs_emp_probe_median']:+.2f} cos_tan(dec,emp) {r['checks']['cos_dec_tan_vs_emp_probe_median']:+.2f} rank(pf_full,pf_tan) {r['checks']['rank_pf_full_vs_pf_tan']:+.2f}")
print("== xfit record: which columns were cross-fitted, values")
for r in rows(C + "09_physics_probe_facing_split_xfit.jsonl"):
    if r.get("row") != "xfit": continue
    c = r["columns"]
    print(r["d"], f"{r['label']:16s}", sorted(c), " dec A->B %+.3f B->A %+.3f | al_tan A->B %+.3f | split cos p50 %+.2f" % (
        c["hess_mismatch_dec"]["fitA_scoreB"]["partial"], c["hess_mismatch_dec"]["fitB_scoreA"]["partial"],
        c["align_cos_tan"]["fitA_scoreB"]["partial"], r["hessian_split_half_cos_p25_p50_p75"][1]))
print("== |emp - |Hy|| gap across 10-encoder sweep main_xfit (multiscale)")
gaps = []
for f in sorted(glob.glob(C + "scaling/records/scaling__*__main_xfit.jsonl")):
    for r in rows(f):
        if r.get("row") != "result" or r.get("label") not in LAB: continue
        c = r["columns"]; e = c["hess_mismatch_emp"]["multiscale"]["partial"]; h = c["hess_label"]["multiscale"]["partial"]
        gaps.append((abs(e - h), f.split("__")[1], r["label"], e, h))
g = np.array([x[0] for x in gaps]); print(f"n={len(g)} median {np.median(g):.3f} <=0.05: {(g<=0.05).sum()} <=0.03: {(g<=0.03).sum()} max {g.max():.3f}", sorted(gaps)[-3:])
print("== counterfactual npz (published, alpha=100)")
for f in sorted(glob.glob(C + "09_physics_normal_scaling_*_d*.npz")):
    if f.endswith("_thin.npz"): continue
    z = np.load(f); tag = f.split("scaling_")[1][:-4]
    for lab in LAB:
        out = []
        for v in ("S_model", "S_proj", "S", "random_qmatched"):
            eq, qq, cv = z[f"{lab}:{v}:eq"], z[f"{lab}:{v}:qq"], z[f"{lab}:{v}:r2_curve"]
            m = np.isfinite(eq); t = eq[m] / qq[m]
            help_ = np.mean(cv[m, 4] > cv[m, 2]); hurt = np.mean(cv[m, 0] < cv[m, 2])
            ident = np.all((cv[m, 4] > cv[m, 2]) == (t > 0.5)) and np.all((cv[m, 0] < cv[m, 2]) == (t > -0.5))
            out.append(f"{v} help {help_:.2f} hurt {hurt:.2f} t* {np.median(t):+.2f}{'' if ident else ' IDENT-FAIL'}")
        ratio = np.sqrt(z[f"{lab}:S_model:qq"] / z[f"{lab}:S:qq"])
        print(f"{tag:20s} {lab:16s} | " + " | ".join(out) + f" | med ||q||/||p|| {np.nanmedian(ratio):.2f}")

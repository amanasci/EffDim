"""Print Table 2 rows for the ML4PS manuscript from the split record. No values typed by hand."""
import json, sys
rec = sys.argv[1] if len(sys.argv) > 1 else "notebooks/.cache/09_physics_probe_facing_split.jsonl"
rows = [json.loads(l) for l in open(rec) if l.strip()]
res = {(r["label"], r["d"]): r for r in rows if r.get("row") == "result"}
labels = ["mag_r", "photo_z", "smooth_fraction", "stellar_mass"]
cols = ["H_tan_norm", "pf_tan", "pf_rad", "hess_mismatch_emp", "align_cos_tan"]
ctrl = sys.argv[2] if len(sys.argv) > 2 else "multiscale"
def cell(r, c):
    v = r["columns"][c][ctrl]
    star = "^{*}" if v["p"] > 0.05 else ""
    return f"${v['partial']:+.2f}{star}$"
for lab in labels:
    out = [lab.replace("_", "\\_")]
    for c in cols:
        for d in (16, 20):
            r = res.get((lab, d))
            out.append(cell(r, c) if r else "--")
    print(" & ".join(out) + " \\\\")
print("% checks:")
for (lab, d), r in sorted(res.items(), key=lambda kv: (kv[0][1], labels.index(kv[0][0]))):
    ch = r["checks"]
    print(f"% {lab} d={d}: rank(pf_full,pf_tan)={ch['rank_pf_full_vs_pf_tan']:+.3f} rank(pf_full,pf_rad)={ch['rank_pf_full_vs_pf_rad']:+.3f} "
          f"cos(dec,emp)={ch['cos_dec_tan_vs_emp_probe_median']:+.2f} rank(dec,emp)={ch['rank_dec_tan_vs_emp_probe']:+.2f} "
          f"wN={ch['w_N_fraction_median']:.2f} med pf_tan/pf_rad={r['columns']['pf_tan']['median']:.3g}/{r['columns']['pf_rad']['median']:.3g} "
          f"quadR2 probe {r['probe_quad_r2_p50']:.3f} vs lin {r['probe_lin_r2_p50']:.3f}; label gain {r['label_quad_r2_gain_p50']:.3f}; "
          f"align p50 {r['align_cos_tan_p25_p50_p75'][1]:+.2f}")
    for c in ("pf_full", "pf_tan", "pf_rad", "hess_label", "hess_mismatch_dec", "hess_mismatch_emp", "align_cos_tan", "cross_tan"):
        v = r["columns"][c]
        print(f"%    {c:18s} sealed {v['sealed']['partial']:+.3f} (p={v['sealed']['p']:.3f})  multi {v['multiscale']['partial']:+.3f} (p={v['multiscale']['p']:.3f})  vs logr {v['rho_vs_log_r']:+.2f}")

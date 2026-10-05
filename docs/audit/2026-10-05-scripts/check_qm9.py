import json, glob, numpy as np
C = "/home/akagi/Documents/Projects/EffDim/notebooks/.cache/qm9/records/"
sur = {"published": [], "tuned": []}; help_t = []; tstar_t = []; alpha_edge = 0; n = 0; mm_t = []
for f in sorted(glob.glob(C + "*__robust.jsonl")):
    enc = f.split("scaling__")[1].split("__")[0]
    for l in open(f):
        r = json.loads(l)
        if r.get("row") == "guard": g = r
        if r.get("row") != "result": continue
        sur[r["alpha_mode"]].append(r["surrogate"]["spearman"])
        if r["alpha_mode"] == "tuned":
            n += 1; alpha_edge += r["alpha"] <= 1e-3 + 1e-12
            help_t.append(r["cf"]["S_model"]["help"]); tstar_t.append(r["cf"]["S_model"]["t_star"])
            mm_t.append((enc, r["label"], round(r["partials"]["published_controls"]["hess_mismatch_emp"]["partial"], 3),
                         r["bootstrap"]["32"]["hess_mismatch_emp"]["excludes_zero"]))
for k, v in sur.items(): print(k, "surrogate spearman range %+.2f..%+.2f" % (min(v), max(v)))
print("tuned help range %.2f..%.2f, t* range %.2f..%.2f, alpha* at grid floor %d/%d" % (min(help_t), max(help_t), min(tstar_t), max(tstar_t), alpha_edge, n))
print([x for x in mm_t if x[0].startswith("chemfm")])

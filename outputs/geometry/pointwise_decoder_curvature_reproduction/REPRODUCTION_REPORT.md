# REPRODUCTION_REPORT

Bounded reproduction of Austin Lutterbach's pointwise decoder-curvature pipeline.
No manuscript edits. No local-quadratic Q estimator. No n=86471 sphere training.

- curvature-experiments SHA: `97efb2eb6cd7dec7f2c568f53c534752ff3c32c8`
- fixture-validity-audit SHA (tables + fixtures): `dcd2208803f27224ee182cce47fdee21c1bc6ba5`
- primary label: `colleague_decoder_results_reproduced`
- estimand label: `reported_quantity_is_full_euclidean_curvature`
- radial label: `radial_baseline_not_applicable`
- runtime_s: 834.2675635814667
- skipped: ['R5:hard_cap_max_4_new_decoders', 'cached_sphere_evaluation_skipped_missing_weights']
- unit tests: 11/11

R1 misses the frozen ρ tolerance (0.654 vs 0.553, Δ=0.101 > 0.05). That cell’s
historical numbers come from the Supp. 02 orchestrator brief, not from a JSONL
dump of the swiss-roll runner. Cosine and ratio on R1 still pass. R2–R4 (cubic
and both ridge cells) pass every available metric, which is what the primary
label requires.

## Recovered formula

Reported H is the unnormalized metric trace `H = g^{ab} II_ab` of the full Euclidean
second fundamental form `II = (I - P_T) D²F`, differentiated through raw `model.decode`.
The factor `1/d` is **not** used. Sphere-radial projection is **not** applied on R1–R4.

## Cell parity

- R1 swiss_roll d=2 D=3: ΔVE=nan Δρ=0.10075085831998154 Δcos=0.00030104664898433775 Δratio=0.03365439373474599 pass=False
- R2 cubic d=16 D=28: ΔVE=0.00043859237461829625 Δρ=0.0030230961849238014 Δcos=0.00027202820605154443 Δratio=0.011742447523654831 pass=True
- R3 ridge d=16 D=28: ΔVE=0.00010692221211727748 Δρ=0.0025529993341198987 Δcos=0.00017225887826533004 Δratio=0.001637050915719085 pass=True
- R4 ridge d=16 D=768: ΔVE=0.0005444944903043591 Δρ=0.0013198865807955151 Δcos=5.7438719567648455e-05 Δratio=0.003399002872776391 pass=True

## Radial baseline

Not applicable: none of R1–R4 constrain the decoder image to the unit sphere.
Cached n=86471 D=768 sphere weights were not found.

## What this does not claim

Numerical parity does not mean the learned decoder recovered the data Hessian.
High reconstruction R² does not validate second derivatives.
This run does not adjudicate the finite-patch quadratic estimator.


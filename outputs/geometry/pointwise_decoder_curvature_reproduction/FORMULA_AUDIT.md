# FORMULA_AUDIT

Source of record: `notebooks/pu_manifold/decoder_curvature.py::plain_decoder_curvature`
on `97efb2eb6cd7dec7f2c568f53c534752ff3c32c8`, with `CURVATURE_CONVENTION = "trace"`.

## Reported H

```
H = g^{ab} II_ab
```

Not `(1/d) g^{ab} II_ab`. `H_IS_AVERAGED = False`.

The implementation traces first (`raw = g^{jk} Hess_jk`) then removes the
tangent component of that ambient vector. That equals the g-trace of
`II = (I-P_T) D²F`.

## Reported II

```
II = (I - P_T) D²F
P_T = J g^{-1} J^T
g = J^T J
```

Not `(I - xx^T - P_T) D²F`. `II_REMOVES_SPHERE_RADIAL = False`.

## Differentiated map

Raw `model.decode`. `F_DIFFERENTIATED_AFTER_NORMALIZE = False`.

On a unit-sphere immersion the Euclidean identity `II^E = -g ⊗ x + B^S` still
holds mathematically, but the historical estimator reports `H^E` (full
Euclidean), not `H^S`. R1–R4 are not sphere-constrained, so the radial
decomposition is not applied to those cells.

## Secondary estimand

`reported_quantity_is_full_euclidean_curvature`

## Propositions kept separate

1. Full `II^E` is the Euclidean curvature of the decoder image.
2. `B^S` is the second fundamental form of that image inside the unit sphere.
3. A large full-curvature score may be dominated by a normalization sphere.
4. A small residual-energy fraction does not make residual curvature unreal.
5. Numerical reproduction does not establish recovery of the data Hessian.
6. High reconstruction R² does not validate second derivatives.
7. This experiment does not adjudicate the finite-patch quadratic estimator.

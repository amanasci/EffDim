# Does the Curvature of a Representation Manifold Predict Where Two Models Agree?

## A white paper on curvature-conditioned alignment in foundation-model embeddings of galaxies

## Abstract

Foundation models trained on images of the same galaxies from different sky surveys produce
embeddings that agree weakly about which galaxies are neighbours. We asked whether that agreement
varies with the local curvature of the embedding manifold, on the hypothesis that regions where
the manifold bends sharply are where two models disagree most.

We built a pipeline to measure per-point mean curvature of a 768-dimensional embedding manifold by
differentiating through a trained decoder, validated that instrument on synthetic surfaces with
known curvature, and correlated the resulting field against three alignment probes and one
label-decodability probe. Every experiment fixed its decision rule before seeing data.

Curvature magnitude and cross-survey neighbour agreement correlate negatively, in the direction
the hypothesis predicts, at every latent dimension tried. Local point density explains between
half and four fifths of that correlation. After controlling for density, one latent dimension
retains a small significant association, and that association does not survive correcting for the
curvature the unit-sphere normalization adds by construction. A parallel experiment on label
decodability returned a sign opposite to a colleague's, and a known-answer test then showed the
statistic both of us used reads density coupling rather than curvature. We conclude that the
record contains no curvature-alignment finding free of a named confound, and we describe the
instrument, the validation protocol, and the confounds in enough detail to be reused.

---

## 1. The question

### 1.1 Background

The Platonic Representation Hypothesis proposes that large models trained on different data with
different objectives converge toward a shared internal representation of the world. Duraphe,
Smith, Sourav and Wu tested a version of it on astronomy (arXiv:2509.19453). They embedded images
of the same galaxies from several sky surveys with several vision foundation models and measured
how well the resulting embeddings agree about neighbourhood structure. Two model sizes on the same
survey agreed on 28 to 56 percent of nearest neighbours. Two surveys through the same model agreed
on 0.4 to 2 percent for the Legacy Survey against Hyper Suprime-Cam (HSC), still far above chance.

That result is one number per pair of embeddings. It says nothing about whether the agreement is
spread uniformly across the space the embeddings occupy or concentrated in some regions.

### 1.2 Hypothesis

Neural embeddings of a coherent dataset tend to lie near a low-dimensional curved surface inside
the high-dimensional output space. We hypothesized that where that surface bends sharply, small
differences in input pixels move a point far along the surface, so two models trained on
different pixels of the same galaxy will disagree more about its neighbours. The prediction is a
negative association between local curvature magnitude and local cross-survey agreement.

### 1.3 What would count as an answer

A per-point curvature field on the embedding manifold, a per-point alignment score, and a
correlation between them that survives the obvious alternative explanations. The obvious
alternative turned out to be local point density, which affects both quantities.

---

## 2. Data

The primary data are image embeddings from a self-supervised vision transformer (DINOv3 ViT-B/16)
applied to galaxy cutouts from two surveys. Each galaxy appears once per survey, so each row of
the dataset pairs an HSC embedding with a Legacy Survey embedding of the same object. Each
embedding has 768 dimensions.

| Item | Value |
|---|---|
| Galaxies available | 101,725 |
| Galaxies used | 10,000, drawn once at random with a recorded seed |
| Embedding width | 768 |
| Normalization | each embedding divided by its own Euclidean norm |

The dataset ships no labels and no object identifiers. Row position is the only link between the
two surveys. We checked that link statistically: the mean cosine similarity between paired HSC and
Legacy Survey embeddings was 204 standard deviations above its value under random row shuffles.

Normalizing every embedding to unit length places all 10,000 points on the unit sphere in 768
dimensions. That choice is standard practice for cosine-similarity comparisons. It has a
consequence for curvature that we return to in Section 4.4.

A second experiment (Section 7) used 86,471 embeddings of galaxies from a single survey, joined by
row position to catalogue labels: r-band magnitude, photometric redshift, a morphology fraction,
and stellar mass. That join was also verified statistically before use.

---

## 3. Concepts

This section defines the tools the rest of the paper uses. Readers familiar with manifold
learning, differential geometry and permutation statistics can skip ahead and return as needed.

### 3.1 Manifolds and intrinsic dimension

Points in a 768-dimensional space can still lie on a much thinner surface, the way a crumpled
sheet of paper occupies a three-dimensional room while being two-dimensional itself. The room is
the **ambient space** with dimension `D`. The sheet is the **manifold** with **intrinsic
dimension** `d`. Here `D = 768` throughout and `d` was estimated at 18 to 25.

Estimating `d` from samples is a research field of its own. We used a panel of eight estimators
(maximum likelihood, TwoNN, DANCo, MiND, ESS, TLE, GMST and a variant) and took their median, which
gave 18. TwoNN, which reads `d` from the ratio of each point's second-nearest to nearest-neighbour
distance, gave 19.5. Local principal component analysis, which counts how many directions carry
90 percent of the variance in a small neighbourhood, gave a median of 25.

The difference `D − d` is the **codimension**: the number of independent directions in which the
surface can bend at each point. For a sheet in a room it is one. For our data it is about 748.
Much geometric intuition comes from the codimension-one case, and several of our early mistakes
came from applying it here.

### 3.2 Isomap and classical multidimensional scaling

**Isomap** unrolls a manifold in three steps. Connect each point to its `k` nearest neighbours.
Compute shortest paths between all pairs along that graph; these approximate **geodesic
distances**, the distance you would walk along the surface. Then find coordinates in `d`
dimensions whose straight-line distances match the geodesic distances.

The last step is **classical multidimensional scaling** (MDS). Square the distance matrix,
double-centre it, and take its eigenvalues and eigenvectors. If the distances came from points
that sit flat in some Euclidean space, every eigenvalue is non-negative and the top `d`
eigenvectors are the coordinates. Negative eigenvalues mean no flat space can reproduce the
distances. A small negative tail is noise. A large one means the surface is intrinsically curved
and Isomap's coordinates are distorted.

### 3.3 Auto-encoders as parameterizations of a manifold

An **auto-encoder** is a neural network trained to reproduce its input through a narrow
bottleneck. An encoder maps a 768-dimensional embedding to a `d`-dimensional latent vector `z`; a
decoder `F` maps `z` back to 768 dimensions; training minimizes the reconstruction error. If `F`
is smooth and `d` matches the intrinsic dimension, `F` is a **parameterization** of the manifold:
a differentiable map from a flat `d`-dimensional chart onto the curved surface. Curvature is a
property of that map.

We tested three decoder architectures.

**Plain auto-encoder.** One encoder, one decoder, three hidden layers of width 250.

**Chart auto-encoder** (Schonsheck, Chen, Lai, arXiv:1912.10094). Some manifolds cannot be
covered by one flat chart without tearing; a sphere is the standard example. A chart
auto-encoder learns several charts, each covering part of the manifold, plus a **partition of
unity** that softly assigns each point to charts. The loss rewards whichever chart reconstructs a
point best.

**Topological auto-encoder** (Moor, Horn, Rieck, Borgwardt, ICML 2020, arXiv:1906.00722). A plain
auto-encoder whose loss adds a term that preserves the persistent homology (Section 3.5) of each
training batch between input and latent space. In practice this preserves the edge lengths of the
batch's minimum spanning tree.

Every decoder used the SiLU activation `x · sigmoid(x)` rather than ReLU. Curvature is a second
derivative of the decoder. ReLU is piecewise linear, so its second derivative is zero almost
everywhere. SiLU is infinitely differentiable.

### 3.4 Curvature of a parameterized surface

Let `F: R^d → R^D` be a smooth decoder and `z` a latent point. The Jacobian `J = DF(z)` is a
`D × d` matrix whose columns span the tangent space at `F(z)`.

The **first fundamental form**, or metric, is

```
g = Jᵀ J          (d × d)
```

It records how lengths in latent space stretch on the surface. Two summaries recur below. The
condition number `cond(g) = λ_max / λ_min` compares the most and least stretched directions. The
log-determinant `log det g` measures overall scale; zero means the geometric-mean stretch is one.
The condition number is scale-invariant: a decoder can shrink every direction by a million and
`cond(g)` will not notice.

The **second fundamental form** measures how the surface bends away from its tangent plane. Take
the Hessian of `F`, a `D × d × d` tensor of second derivatives, and project it onto the normal
space:

```
P_N = I_D − J g⁻¹ Jᵀ          (normal projector)
II  = P_N · D²F(z)            (D × d × d)
```

The **mean curvature vector** is the trace of `II` with respect to the metric:

```
H = tr_g(II) = Σ_{j,k} g^{jk} II_{jk}          (a vector in R^D)
```

`H` points in the direction the surface bends on average, and `‖H‖` says how hard. On a sphere of
radius `R` this convention gives `‖H‖ = d / R`, so a unit `d`-sphere has `‖H‖ = d`. Some texts divide
by `d`; we do not, and we pin the convention in code because the two conventions differ by a
factor that has caused at least one bug in this project and one factor-of-two discrepancy with a
collaborator.

All derivatives are computed by automatic differentiation. We verified the implementation against
central finite differences to a relative error of 5e-8 on the raw Hessian and, on a later
fixture, to a cosine of 1.0000000000 between the two mean-curvature vectors.

We report `‖H‖`, never Gaussian curvature, which is not canonically defined above codimension one
(arXiv:1312.2554).

### 3.5 Persistent homology and Betti numbers

**Homology** counts holes. The **Betti numbers** of a shape are `β_0` (connected pieces), `β_1`
(independent loops), and `β_2` (enclosed voids). A circle is `(1, 1, 0)`; a sphere `(1, 0, 1)`; a
torus `(1, 2, 1)`; a solid ball `(1, 0, 0)`.

A finite point cloud has no holes. **Persistent homology** grows a ball of radius `ε` around every
point, joins overlapping balls, and records the `ε` at which each hole appears and the `ε` at which
it fills in. Each hole is a bar `(birth, death)` in a **persistence diagram**. Long bars are
structure; short bars are noise. To decide which bars are real, we used the bootstrap band of Fasy
and colleagues (arXiv:1303.7117): resample the cloud, recompute the diagram, and treat bars longer
than twice the 95th-percentile bottleneck distance to the original as significant.

### 3.6 A training-free curvature estimator

Mean curvature can also be estimated directly from a point cloud. Take a point, find its `k`
nearest neighbours, fit a tangent plane by local singular value decomposition, and observe where
the neighbours' centroid sits relative to that plane. On a flat surface the centroid lies in the
plane. On a curved surface it is displaced along the normal by an amount proportional to the
trace of `II`. Inverting that gives `H`. We call this the **centroid estimator**. It estimates one
unknown from `k` vectors, which keeps it feasible at `d = 20` where a full quadratic fit would need
210 coefficients per normal direction. Its bias grows with the neighbourhood radius, which at
`d = 20` is a large fraction of the whole manifold.

### 3.7 Validating a curvature instrument

A curvature field on real data has no ground truth. Two checks appear in this work.

**Split-half reliability** divides the cloud in two, estimates curvature on each half, and
correlates the estimates at shared anchor points. It measures reproducibility. It cannot detect a
bias both halves share: on a synthetic test surface the centroid estimator scored split-half
reliability 0.990 while its rank correlation with true curvature was 0.469.

**Known-answer validation** runs the estimator on a synthetic surface whose curvature is known in
closed form and scores it on four separate axes: **rank** (Spearman correlation between estimated
and true `‖H‖`), **direction** (median cosine between estimated and true `H` vectors),
**magnitude** (median ratio of estimated to true norm), and **calibration** (slope of estimate
against truth). Keeping the axes separate matters. A combined score would have reported `d = 20` as
hopeless when direction was in fact recovered at cosine 1.000.

The cheapest known-answer test is the **Swiss roll**, a flat two-dimensional sheet rolled into
three dimensions with curvature known analytically. Every model in this project was required to
pass it before being trusted on real data. A model that fails the roll is broken; a model that
passes the roll and fails on real data has found something about the data.

### 3.8 Alignment probes

**Mutual k-nearest-neighbour score (MKNN).** For two embeddings of the same objects, let
`N_k(z1)_i` be the `k` nearest neighbours of object `i` in the first embedding and `N_k(z2)_i` the
same in the second. The per-object score is `|N_k(z1)_i ∩ N_k(z2)_i| / k`, the fraction of
neighbours both embeddings agree on. Under independent embeddings its expected value is `k / n`,
the **chance floor**: 0.002 at `n = 10,000`, `k = 20`.

MKNN is label-free and training-free. It is also a nearest-neighbour statistic, so it is sensitive
to local point density by construction. That sensitivity is the central confound of this paper.

Because the per-object score is `j / k` for an integer `j`, it takes at most `k + 1` distinct values
(we observed 15 at `k = 20` across 10,000 points). Any test that assumes continuous values, such as
the textbook p-value for Spearman correlation, does not apply, so we used permutation nulls.

**Centered kernel alignment (CKA)** (Kornblith et al. 2019). Compute a Gram matrix for each
embedding, centre both, and take their normalized Hilbert-Schmidt inner product. CKA measures
agreement about global similarity structure rather than nearest neighbours, and is invariant to
rotation and isotropic scaling. We used the unbiased estimator, which zeroes the Gram diagonal.

### 3.9 Decodability probes

A **linear probe** asks how much of a target a linear map can read from a representation.
**Ridge regression** adds a penalty `α‖W‖²` to least squares so the weights stay small. We used two
forms. First, a ridge map from one survey's embedding to the other's, scored by per-point squared
residual on held-out rows. Second, a ridge map from an embedding to a catalogue label, fit
**out-of-fold** (each row predicted by a model that never saw it) and scored by **local R²**, one
minus the ratio of residual to total sum of squares inside each anchor point's 2,048-nearest-
neighbour patch. Low local R² means the probe works badly in that region.

### 3.10 Correlation, nulls, and partial correlation

**Spearman's ρ** is the correlation between ranks. It is scale-free: it asks whether an ordering
is resolvable, not how wide the spread is.

A **permutation null** shuffles one variable relative to the other many times and recomputes the
statistic, giving the distribution of values expected under no relationship. We report either a
threshold (the 97.5th percentile of each tail, a two-sided test at 0.05) or an exact p-value
`(1 + draws at least as extreme) / (N + 1)`. When no draw reaches the observed value we report
`p < 1/(N+1)`, never `p = 0`.

A **bootstrap confidence interval** resamples the data with replacement and reports the 2.5th and
97.5th percentiles of the recomputed statistic.

A **partial correlation** asks how two variables relate after removing what a third explains.
Our version: rank each variable, regress each rank vector on the rank of the control, and
correlate the residuals. A partial correlation needs its own null. Permuting the raw variables
destroys the structure the partial exists to control for, so we built a **stratified null**: cut
points into quantile strata of the control variable, permute the two variables of interest
independently within each stratum, and recompute the partial. This preserves the control's
structure and breaks only the link under test.

The **Freedman-Lane** procedure is the regression analogue: fit the outcome on the controls,
permute the residuals, add them back, recompute. Taking the maximum absolute statistic per
permutation across several latent dimensions gives a **family-wise** null that accounts for testing
several `d` at once.

### 3.11 Controls and pre-registration

A **negative control** feeds the pipeline data with no real structure and checks it returns a null
at the nominal rate. A **positive control** plants a known effect and checks the pipeline detects
it; without one, a null result cannot be distinguished from a test too weak to see anything.

**Pre-registration** fixes the decision rule before the data are seen. Every threshold, seed, grid
and rule in this project was committed to version control before the corresponding number existed,
and each analysis script refused to run unless that commit was a strict ancestor of the running
code. This does not make a result true. It makes it impossible to have tuned the rule to the
result.

### 3.12 Local density

We measured density at a point as the inverse volume of the ball reaching its 30th nearest
neighbour in ambient space. With `d` fixed this is a strictly decreasing function of the radius, so
any rank statistic on density is a rank statistic on neighbourhood radius. Density in our data
spans about eight orders of magnitude between the 5th and 95th percentiles.

Density enters both sides of the hypothesis. Nearest-neighbour alignment reads differently in
dense regions. Curvature estimators, both decoder-based and point-cloud, read differently in dense
regions. If both correlate with density, they correlate with each other whether or not they are
related. Most of the work below is about separating that from a real effect.

---

## 4. Building and validating the curvature instrument

### 4.1 The manifold is intrinsically curved and has no loops

We fit Isomap at `k = 15` neighbours (chosen by a stability rule fixed in advance) and computed
the full eigenspectrum of the double-centred geodesic distance matrix. Of 10,000 eigenvalues,
5,029 were negative, carrying 41 percent of the total absolute eigenvalue mass. The largest
negative eigenvalue was 5 percent of the largest positive one. Against acceptance criteria fixed in
advance (negative mass below 5 percent to pass, below 15 percent for marginal), the embedding
failed. The result was stable across neighbour counts from 5 to 30, across the two surveys, and
across a disjoint subsample. Isomap coordinates are not a trustworthy flat embedding of this
manifold.

A second, independent measurement agrees: the median ratio of geodesic distance to straight-line
distance between points is 1.54.

Persistent homology of the cloud gave `β_1 = 0` on every bootstrap draw, with a positive control
(a circle times an 19-dimensional flat factor) returning `β_1 = 1` on every draw at the same sample
size and a negative control (a 20-ball) returning 0. The manifold has no loops at this resolution.
That rules out the need for a multi-chart decoder and supports a single Euclidean latent space.

### 4.2 Which decoder to differentiate through

We tested the chart auto-encoder, the topological auto-encoder, and the plain auto-encoder against
acceptance criteria written before training, on the 10,000-row sample at `d = 20`.

The **chart auto-encoder** with 16 charts reconstructed held-out data 3.6 times worse than a plain
auto-encoder at the same bottleneck and distorted geodesic distances by a median 30 percent against
a 15 percent bar. A later audit found the fit had collapsed to a 16-point quantizer: chart
coordinates had a standard deviation of 5e-7, and a nearest-centroid classifier using the model's
own 16 constants outperformed the model. On the Swiss roll the same architecture worked well,
recovering the sheet at 4.8 percent relative error. On the real data at `d = 20` it does not
underfit; it diverges. Six-hundred-epoch runs on a synthetic `d = 20` surface went from 21 percent
variance explained to −7 percent while a plain auto-encoder climbed monotonically to 98.6 percent
on identical data.

The **topological auto-encoder** preserved persistence-pair edge lengths better than the plain
auto-encoder on the Swiss roll (correlation 0.680 versus 0.471) at a 38 percent cost in
reconstruction error, which is the trade the method is designed to make. On the real data it lost
on both topological fidelity and reconstruction against its matched baseline. Reproducing it
required an audit against the authors' code that found four normalizations the paper does not
mention, two of which compounded to a 32-fold error in the effective loss weight. The published
method is sound; our first translation of it was not.

An attempt to rank the three decoders by persistent-homology agreement with the Swiss roll's
true topology separated nothing. Every gap between architectures sat inside every architecture's
own seed-to-seed spread.

The **plain auto-encoder** was adopted, at first by elimination and later on evidence. Fit on a
synthetic `d = 20` surface it reached 99.8 percent variance explained in 25 seconds with
`cond(g) = 3.7`, and its decoder curvature scored rank 0.73, direction 0.91 and magnitude ratio 0.94
against the truth. The chart auto-encoder on the same surface had never reconstructed above 69
percent and gave `cond(g)` of 1e8.

### 4.3 Why early curvature attempts returned zero

Before the plain auto-encoder result, every attempt to recover curvature at `d = 20` had
returned a rank correlation near zero, on decoders and on point clouds alike. Three separate causes
were eventually identified.

**Metric collapse.** Differentiating through the chart auto-encoder gave three seeds whose median
`‖H‖` differed by a factor of 52, with two of three fields piecewise constant. Their metrics had
collapsed by seven orders of magnitude in absolute scale while `cond(g)` stayed near 1e7 for all
three. The training objective constrains no decoder derivative at any order, and the condition
number cannot see uniform collapse. Adding a scale prior `(log det g / d)²` to the loss drove
`log₁₀ det g` from −83.9 to +0.04 and `cond(g)` from 1.7e8 to 573, at a 2 percent reconstruction
improvement, and moved the curvature rank correlation only from −0.12 to +0.12. Repairing the
metric was necessary and not sufficient.

**The test surface had nothing to rank.** The synthetic control used for `d = 20` was a quadratic
saddle `f(x) = ½ xᵀ diag(±1) x` lifted into the ambient space. Every quadratic graph has a constant
Hessian, so its second fundamental form is constant at every point, and all variation in `‖H‖`
comes from the metric tilt `1 / (1 + |∇f|²)` rather than from the geometry an estimator is asked to
measure. At identical `d = 20`, `n = 5,000`, `k = 231` and estimator, the saddle gave rank +0.02 and a
cubic surface with varying Hessian gave +0.61. Sweeping `k` from 60 to 800 never moved the saddle
off zero. Making the surface non-minimal makes it worse, since raising the trace adds a constant
to every point. For separable surfaces the Hessian norm concentrates like `1/√d`, so at high `d`
they are nearly constant-curvature however they are built; a ridge surface `A sin(w · x)` with a
rank-one Hessian escapes that and keeps its variation flat in `d`.

**Neighbourhood scale.** For the centroid estimator at `k = 30`, the ratio of neighbourhood radius
to manifold radius is 0.12 at `d = 2` and 0.91 at `d = 20`, following `(k/n)^{1/d}`. Halving it at
`d = 20` would need about a million times more data. The estimator's per-point magnitude error has a
coefficient of variation of 2.25 at `d = 20`: it behaves like noise while being bias.

### 4.4 What survives at high dimension, and the unit-sphere term

With a rankable surface and a well-fit decoder, the four fidelity axes separate cleanly at
`d = 20`. Direction is recovered at cosine up to 1.000. Rank saturates between 0.4 and 0.65 for
the centroid estimator and reaches 0.73 to 0.98 for the decoder. Magnitude is attenuated
fifty-fold for the centroid estimator (median ratio 0.018) and recovered to within 6 percent by
the decoder. This matches the published result that bias in curvature estimation grows sharply with
dimension (arXiv:2511.02873), whose own validation stops at dimension twelve on spheres, which are
constant-curvature and cannot test ranking at all.

Dynamic range of `‖H‖` does not determine rankability. A ridge surface at 1.1-fold spread ranked
at +0.48; at 36-fold spread, +0.36. Spearman is scale-free. The real axis is constant versus
varying second fundamental form.

The decoder instrument was then scored on two rankable analytic surfaces at production ambient
width:

| Surface | `D` | Rank ρ at `d = 20` | Rank ρ at `d = 25` | Rank ρ at `d = 16` |
|---|---|---|---|---|
| cubic | 28 | 0.87 | 0.78 | |
| cubic | 768 | 0.53 | 0.17 | |
| ridge | 28 | 0.98 | 0.96 | |
| ridge | 768 | 0.97 | 0.97 | |
| combined range | | 0.53 to 0.99 | 0.17 to 0.97 | 0.84 to 0.99 |

Reconstruction quality does not predict curvature fidelity: the two `D = 768` rows at `d = 20`
reconstruct at 99.70 and 99.88 percent and score 0.53 and 0.97. Nothing on the record says which
surface the real data resembles. At `d = 32` no fidelity measurement exists, for a fixture-design
reason: the small-ambient arm's width was a literal 28 and a `d = 32` graph surface needs 33.

**The radial term.** Because every embedding was normalized to unit length, the manifold sits
inside the unit sphere `S^767`, and any `d`-dimensional submanifold of a unit sphere carries a
mean-curvature component pointing at the centre of magnitude exactly `d`. That component is a
property of the sphere, not of the data's shape. The tangential part, `‖H_tan‖ = sqrt(‖H‖² − d²)`,
carries the shape. The decomposition is exact only when the decoder's image lies on the sphere,
which an unconstrained decoder does not guarantee. Differentiating the renormalized decoder
`F / ‖F‖` instead of `F` fixes this and gives `H_rad = −d` to fourteen decimal places. We adopted
that correction late, and its effect on results appears in Sections 5.4 and 7.

---

## 5. Curvature and cross-survey alignment

### 5.1 Two designs that were confounded, and why

Our first design partitioned the 10,000 points into two regions by curvature direction (the sign
of each unit mean-curvature vector's projection onto the leading principal direction of all such
vectors) and compared MKNN between regions. Region 1 scored 0.174 against region 0's 0.081 at
`k = 20`, with disjoint bootstrap intervals and each region clearing its own permutation null. That
looks like a result. It is not one. The split direction correlated with density at Spearman +0.82,
the two regions' median densities differed by a factor of about 5,700, and because MKNN's chance
floor is `k / n_region`, the smaller region has a higher floor: 90 percent of the raw gap
disappears when scores are expressed as multiples of chance. The direction split also rested on
codimension-one intuition applied to a codimension-748 problem.

Our second design bucketed held-out residuals of a ridge map from HSC to Legacy Survey embeddings
into tertiles of `‖H‖`. With the chart-auto-encoder field, two of three seeds showed higher
residual at higher curvature and the third did not, on fields that correlated with each other at
−0.14, +0.20 and −0.27 and had 4 and 3 distinct values across 10,000 points in two of three seeds.
With the centroid field on the same 3,000 residuals, the top and bottom tertiles' intervals
overlapped and the pattern was not monotone. The centroid field correlated with the three decoder
fields at −0.09, +0.05 and −0.12. Two estimators, near-orthogonal fields, different answers on
byte-identical residuals. Neither design measured MKNN, the probe the origin paper used.

### 5.2 Per-point correlation with a validated instrument

The design that finally addressed the question used the plain auto-encoder decoder at three
latent dimensions, per-point MKNN at `k = 20`, and Spearman correlation over all 10,000 points.
Per-point scores give 10,000 paired observations instead of two or three buckets, and Spearman's
scale-freeness sidesteps the fact that the real curvature field's spread is only about 1.5-fold.

| `d` | Variance explained | Median `‖H‖` | ρ(`‖H‖`, MKNN) | Two-sided 0.05 threshold | Exact p |
|---|---|---|---|---|---|
| 20 | 0.982 | 37.2 | −0.112 | 0.0206 | < 1e-5 |
| 25 | 0.984 | 41.4 | −0.128 | 0.0197 | < 1e-5 |
| 32 | 0.986 | 47.0 | −0.024 | 0.0187 | 0.009 |

All three are negative and clear their thresholds. The sign matches the hypothesis. A planted
positive control at the real field's spread was detectable down to a target correlation of 0.02.
The sign and rough magnitude hold across `k` in `{5, 10, 20, 50}`.

Two caveats on the fit numbers. The variance-explained figure divides by mean squared norm rather
than by variance about the mean; on unit-norm data a model emitting one constant vector scores 81
percent by that formula, and the centred figure at `d = 20` is 90.5 percent. And reconstruction
never plateaus: a latent sweep gave 97.3, 97.9, 98.2, 98.4, 98.6 and 98.9 percent at `d` in `{10, 15,
20, 25, 32, 48}`, so no single bottleneck width is defensible on its own, which is why we swept.

### 5.3 Density explains most of it

| `d` | ρ(density, `‖H‖`) | ρ(density, MKNN) | Partial ρ(`‖H‖`, MKNN | density) | Fraction of raw association removed |
|---|---|---|---|---|
| 20 | +0.428 | −0.212 | −0.024 | 78 % |
| 25 | +0.315 | −0.212 | −0.066 | 49 % |
| 32 | +0.012 | −0.212 | −0.022 | 8 % |

Curvature magnitude and density correlate at `d = 20` and `d = 25`; density and MKNN correlate at
every `d`. Residualizing both sides on density removes most of the raw association at the two `d`
where the coupling exists. At `d = 32` there is nothing to control for and also nothing left.

We tested whether ambient density is itself contaminated by curvature, since a curved surface's
chord is shorter than its geodesic. It is not: the Spearman between the geodesic-to-ambient radius
ratio and `‖H‖` is +0.02 at every `d`. Geodesic density gives the same confound (+0.41 against +0.43
at `d = 20`), and controlling on it flatters the result slightly, so we kept ambient density as the
conservative control.

Under a stratified null built for the partial itself (permute curvature and MKNN independently
within density quantile strata), only `d = 25` survives: exact p below 5e-5 at every stratum count
tried, against 0.06 to 0.07 at `d = 20` and 0.10 to 0.17 at `d = 32`. Three independently
initialized decoders at `d = 25` gave partials of −0.066, −0.131 and −0.138, all clearing; their
curvature fields correlate pairwise at 0.72 to 0.85, so they are real replicates and not one fit
counted thrice.

A second alignment probe agrees. Splitting points into curvature tertiles within each density
stratum and computing the CKA difference between top and bottom tertiles gave −0.04 to −0.05 at
`d = 20` and −0.08 at `d = 25` against a null band of about ±0.017, clearing at every stratum count
in `{10, 20, 50}` and in all three seeds at `d = 25`; `d = 32` did not clear. A negative control on
structureless labels gave 1 false clearance in 60 cells. The three `d = 25` seed fields rank in the
same order under MKNN and CKA. The CKA experiment's positive control was invalid (its zero-magnitude
anchor re-measured the live signal), so it carries no power estimate and its `d = 32` null cannot be
distinguished from an underpowered test.

### 5.4 The survivor does not survive the radial correction

The `d = 25` partial is the strongest surviving number. Replacing `‖H‖` with the sphere-tangential
`‖H_tan‖` (Section 4.4) in the density partial does something different at each `d`:

| `d` | Partial with `‖H‖` | Partial with `‖H_tan‖` | Effect |
|---|---|---|---|
| 20 | −0.023 | −0.025 | strengthens 1.1× |
| 25 | −0.066 | −0.023 | collapses 2.8× |
| 32 | −0.027 | +0.056 | sign flips |

The raw correlation at `d = 25` does not move (−0.127 to −0.128). The density-controlled partial
drops into the range of the two `d` that fail. The two fields agree on ranking at Spearman 0.89 to
0.96, which had been our pre-specified test for whether the radial term was a harmless offset; it
passed, and the partial says otherwise. A rank correlation is the wrong sufficient statistic for a
partial correlation, because a partial of magnitude 0.02 to 0.07 lives in the residual the two
fields do not share.

Two readings are open. Either the `d = 25` signal is materially an artifact of the normalization,
or the deviation of `H_rad` from exactly `−d` (it is not constant; the 5th to 95th percentile at
`d = 20` spans −23.7 to −16.6 around a median of −19.8) encodes something real about local
intrinsic dimension or decoder fit that the projection discards. The permutation p-value of the
tangential partial at `d = 25`, which would separate "collapsed to noise" from "collapsed but still
significant," has not been measured.

---

## 6. Where the alignment question stands

Curvature magnitude and cross-survey neighbour agreement correlate negatively on this manifold.
Most of that correlation is density. What remains after density control is small, survives its own
calibrated null at one latent dimension out of three, is replicated across seeds and across two
alignment probes at that dimension, and does not survive removing the curvature the unit-sphere
normalization contributes. The instrument is validated on synthetic surfaces at rank 0.53 to 0.99
at `d = 20` and 0.17 to 0.97 at `d = 25`, with no evidence about which surface the real data
resembles, and unvalidated at `d = 32`.

We do not claim an effect. We also do not claim its absence: at `d = 32`, a dying instrument and a
vanishing effect are indistinguishable, and the CKA experiment has no power estimate.

---

## 7. Curvature and label decodability: a replication that measured density

### 7.1 The experiment

A colleague ran a related experiment on 86,471 single-survey embeddings with catalogue labels. He
fit a ridge probe from embedding to r-band magnitude, scored local out-of-fold R² inside each of
512 anchor points' 2,048-nearest-neighbour patches, estimated curvature at each anchor by a
quadratic fit to the same 2,048 neighbours in a nested-PCA chart of rank `d` after removing the
sphere-radial component, and computed the rank-partial Spearman between curvature and local R²
controlling for log neighbourhood radius, local label variance and evaluation count. His result:
−0.240 at rank 16 (raw −0.412), +0.143 at rank 12, −0.233 at rank 20. Negative means the probe
works worse where curvature is higher.

We reran his design with our curvature instrument in place of his, everything else matched: same
rows, same `k`, same anchors count, same probe with `α = 100`, same controls, Freedman-Lane
permutation with family-wise error across `d` in `{16, 20, 25, 32}`, paired-anchor bootstrap, and
the stratified null beside it. Row alignment between embeddings and labels was proved first (R² at
the assumed alignment 0.516 against −0.0001 for the best misaligned pairing). The decision rule,
fixed in advance, asked whether the controlled partial was negative and cleared its family-wise
null at any `d`.

### 7.2 Opposite sign

| `d` | Variance explained | Raw ρ | Controlled partial | Family-wise p |
|---|---|---|---|---|
| 16 | 0.952 | +0.43 | +0.35 | < 1e-4 |
| 20 | 0.957 | +0.23 | +0.03 | 0.50 |
| 25 | 0.961 | +0.25 | +0.04 | 0.35 |
| 32 | 0.965 | +0.21 | −0.003 | 0.94 |

At `d = 16`, his magnitude and the opposite sign. Shuffled-label calibration returned 5 false
positives in 80 at nominal 0.05. The planted positive control cleared no target at any `d`, for a
structural reason: the plant was capped by the real curvature-outcome relation and aimed at a
negative target while the real relation is positive, so no target in its grid was reachable on this
data regardless of instrument sensitivity.

The radial backstop failed on the first run: median `H_rad` read −20.4 at `d = 16` against the
expected −16, because the decoder image was not constrained to the sphere. Projecting the decoder
onto the sphere before differentiating (Section 4.4) made `H_rad = −d` exactly, moved the field by
rank correlation 0.98 to 0.997, and moved the `d = 16` partial from +0.35 to +0.33. The off-sphere
image was not the cause of the sign.

### 7.3 His estimator inside our pipeline

We imported his estimator unchanged and ran it on the same 512 anchors with the same probe,
controls and nulls. It returned his sign: −0.10 at rank 12, −0.15 at rank 16 (p = 0.0005), −0.235 at
rank 20 against his −0.233. On shared anchors the two curvature fields anticorrelate at −0.46
(`d = 16`) and −0.41 (`d = 20`). His field rises with neighbourhood radius (Spearman +0.70 to +0.77;
his own reanalysis found +0.765); ours falls with it (−0.56). Partial out radius and the two fields
are nearly unrelated (−0.07). Within radius deciles, the mean Spearman with local R² is +0.36 for
our field and −0.19 for his, every decile carrying the sign of its instrument. Stratifying on
density does not reconcile them.

### 7.4 A known-answer test of both estimators

We built a synthetic surface in the production regime: an inverse-stereographic map from `R^16`
into `S^16`, perturbed by four Gaussian bumps, rotated into `R^768`, sampled at 86,471 points from
a mixture of latent scales so that neighbourhood radius varies about twofold across anchors. Its
image lies on the unit sphere, its curvature varies across points, and the truth is exact autodiff
of the explicit generator (`max |H_rad + 16| = 2e-14`). Both estimators saw the same points, anchors
and 2,048-patches, with and without isotropic noise at 25 percent of the median patch radius.
Validation bar, fixed first: rank ≥ 0.7, and for ours direction cosine ≥ 0.8.

| Noise | Instrument | Rank | Direction | Magnitude ratio | Reliability |
|---|---|---|---|---|---|
| none | decoder `H_tan` | 0.94 | 0.999 | 0.999 | var. expl. 0.9999 |
| none | quadratic `K_H` | 0.62 | | | split-half 0.80 |
| 25 % | decoder `H_tan` | 0.75 | 0.95 | 1.09 | var. expl. 0.971 |
| 25 % | quadratic `K_H` | −0.30 | | | split-half −0.61 |

On a clean known answer the decoder field tracks true curvature in rank, direction and
magnitude, and the quadratic estimator tracks it weakly. Under noise the decoder keeps rank 0.75
and direction 0.95; the quadratic estimator becomes anticorrelated with the truth and its own
split-half reliability turns negative. On the real data that reliability read 0.45, a value this
test shows says nothing about accuracy.

Two qualifications from external review. The fixture spans only 18 of the 768 dimensions: it is a
hypersurface of a great 17-sphere, so its in-sphere curvature lives in one normal direction and the
other 750 are flat. And the quadratic estimator's `H` is half the averaged mean curvature (its
features carry no factor of one half), which affects calibration and not rank. On the Swiss roll
the decoder passes (rank 0.55, direction 0.9998); the quadratic estimator is degenerate there by
construction, since sphere projection in `R³` leaves no normal direction for the quadratic term.

### 7.5 The statistic reads density, not curvature

None of the above says what the partial correlation itself measures. No experiment had run the
probe pipeline on a surface where curvature, tangent planes, density and label are all known. We
did that on the same synthetic surface. A latent pool of 259,413 points was subsampled to 86,471
with weight `‖H_tan‖^γ` for `γ` in `{−1, 0, +1}`, so the identical surface was sampled with
curvature concentrated where points are sparse, uniformly, or where points are dense. Four
curvature columns were scored on the same 512 anchors: exact pointwise autodiff, exact mean over
each 2,048-patch, the decoder field, and the quadratic estimator. The label was linear in the
surface's intrinsic coordinates, `y = a · z`.

| `γ` | ρ(exact `‖H_tan‖`, log radius) | Partial, exact | Partial, patch mean | Partial, decoder | Partial, quadratic |
|---|---|---|---|---|---|
| −1 | +0.80 | −0.29 | −0.49 | −0.29 | −0.26 |
| 0 | +0.51 | −0.00 | +0.04 | +0.01 | +0.19 |
| +1 | −0.15 | +0.21 | +0.22 | +0.22 | +0.21 |

Same surface, same label, exact curvature, and the sign of the statistic flips with the sampling.
Every non-zero cell sits at the permutation floor. The three controls, including log radius, do not
remove the dependence. Pointwise and patch-mean columns agree in sign, so neighbourhood scale is
not the explanation. The decoder column matches the exact column in sign and magnitude, so the
instrument is not the explanation either.

The rule the surface exhibits: the partial takes the opposite sign to the curvature field's
coupling with neighbourhood radius. On the real data our field couples at −0.60 and read +0.33; his
couples at +0.70 and read −0.24. Both match. The two instruments disagree on the real data because
they couple to density in opposite directions, and the statistic follows the coupling.

The Swiss roll behaved differently: with the same manipulation the partial stayed negative
(−0.49, −0.76, −0.52). On the roll the global ridge probe fails almost everywhere (global R² 0.07,
local R² down to −60), so curvature's effect on the probe dominates. On the synthetic sphere
surface and, by inference, on the real data, the probe works (R² 0.72 to 0.82), the surface is
mildly curved, and whatever curvature does to the probe is smaller than what sampling does to the
statistic.

### 7.6 Where the decodability question stands

Neither +0.33 nor −0.24 is a curvature finding. The decoder instrument tracks true curvature in
the production regime and the quadratic estimator does so weakly and unreliably under noise, but
the pre-specified statistic cannot turn a good curvature field into a curvature claim without a
density-robust design. Three candidates exist and none has been run: compare curvature to local R²
at matched neighbourhood radius; residualize on a richer density model than one log radius; or
re-weight the sample until the chosen instrument's coupling with radius is zero and read the
partial there.

---

## 8. Discussion

### 8.1 What we can say

The embedding manifold of these galaxies is intrinsically curved, has trivial first homology, and
has intrinsic dimension somewhere between 18 and 25 with no single value defensible. A plain
auto-encoder with a smooth activation, differentiated through its decoder, is a usable mean-
curvature instrument at those dimensions on synthetic surfaces, provided the decoder image is
projected onto the sphere the data live on. Its curvature field correlates negatively with
cross-survey neighbour agreement, and the correlation is mostly local density.

### 8.2 What we cannot say

Whether any curvature-alignment relationship exists beyond density. The one density-controlled
association that survived its null does not survive the radial correction, and the two
experiments that reached opposite signs on label decodability were both reading density coupling.
The record contains no positive finding free of a named confound and no null result with a valid
power estimate at the dimension where the signal vanishes.

### 8.3 Limitations we know about

One model architecture. One survey pair for alignment, one single-survey set for decodability. A
10,000-point subsample chosen so the dense geodesic matrix fits in memory. Instrument fidelity
measured on two synthetic surface families that may not resemble the data, unmeasured at `d = 32`.
An adjudication fixture spanning 18 of 768 dimensions with a 46-fold curvature spread where the real
field spans about 1.8-fold. Density measured as one scalar per point. The unit-sphere normalization,
which contributes a radial curvature of exactly `d` and was corrected only in the last experiment.
Some intermediate analysis decisions in the later experiments were taken by the automated pipeline
under a standing instruction rather than reviewed one by one by a person; the underlying records
mark which.

### 8.4 What a density-robust design would look like

The cleanest test is a within-anchor comparison: at each anchor, compare curvature to local
alignment or local R² among neighbourhoods of matched radius, so density is held fixed by
construction rather than removed by regression. A second option is to re-weight the sample until
the chosen curvature field is uncorrelated with radius, then read the association there, on the
synthetic surface first to confirm the partial goes to zero when curvature is exactly known and
the label is independent of it. Either would settle whether the sign we measured on real data is
curvature or sampling.

---

## 9. Methodological lessons

These are the practices we would keep in any rebuild. Each was learned by paying for its absence.

**Test every manifold model on a surface with a known answer, using the exact code that will run on
real data.** A failure on real data has two causes, no structure or a broken implementation, and
only a known answer separates them. This check caught an unfaithful translation of a published
method, an acceptance check with no baseline, a score that confounded model with estimator, and a
training-budget asymmetry that produced plausible wrong numbers.

**Audit against the authors' reference code, not only the paper.** Normalizations and learnable
scale parameters are what papers omit and implementations depend on.

**Give every acceptance check a baseline, especially on the method's home ground.** An absolute
bound on the one check that measures what a method optimizes will fail it for making the trade it
exists to make.

**Anchor at low dimension before interpreting a failure at high dimension.** Every zero we
measured at `d = 20` was a fixture defect or an unconverged fit, not a dimension limit.

**Keep rank, direction, magnitude and calibration separate.** They come apart at high dimension.
A rank gain without a direction gain is the neighbourhood ball engulfing the manifold.

**Check that a synthetic control has a varying second fundamental form.** Quadratic surfaces are
unrankable by construction. Spread of `‖H‖` is a red herring; Spearman is scale-free.

**Write acceptance criteria at production scale.** Three defects passed every test on a
3,000-point Swiss roll and failed at 768 dimensions. A synthetic tensor at real size costs nothing.

**Match the null to the statistic.** A plain permutation null for a density-controlled partial
ignores the structure the partial exists to control for.

**Calibrate a decision rule on the quantity it decides.** A rank correlation of 0.9 between two
curvature fields passed a rule that the partial correlation those fields produce contradicted.

**Run the full statistic on a surface where everything is known before believing its sign.** The
decodability partial passed instrument validation, control adjustment and permutation testing and
still read density.

**Fix the rule before computing the number, and prove the ordering.** Version control makes this
cheap. Check that the rule's commit is a strict ancestor, since a commit is its own ancestor.

**Do not pool seeds whose fields disagree.** Pooling asserts an agreement the measurement did not
find.

**Report `p < 1/(N+1)`, never `p = 0`, and report margins to full precision whichever way they
land.**

**On unit-normalized data, project the decoder image to the sphere before differentiating, and
print the radial curvature as a hard check on every run.**

**Check convergence before trusting a comparison.** A model stopped at its epoch ceiling while
still improving is a truncation, not a result.

**A decision rule's output is not a finding.** Several of our rules returned affirmative tokens on
confounded inputs, and every one then needed prose to walk it back. When a rule fires on a
confounded input, say so in the same sentence.

---

## 10. References

**Used directly.**

- Duraphe, Smith, Sourav, Wu. *The Platonic Universe: Do Foundation Models See the Same Sky?*
  NeurIPS 2025 ML4PS workshop. arXiv:2509.19453.
- Schonsheck, Chen, Lai. *Chart Auto-Encoders for Manifold Structured Data.* arXiv:1912.10094.
- Moor, Horn, Rieck, Borgwardt. *Topological Autoencoders.* ICML 2020. arXiv:1906.00722.
- Kornblith, Norouzi, Lee, Hinton. *Similarity of Neural Network Representations Revisited.* ICML
  2019.
- Tenenbaum, de Silva, Langford. *A Global Geometric Framework for Nonlinear Dimensionality
  Reduction.* Science 2000.
- Fasy, Lecci, Rinaldo, Wasserman, Balakrishnan, Singh. *Confidence sets for persistence
  diagrams.* Annals of Statistics 42(6). arXiv:1303.7117.
- Radovanović, Nanopoulos, Ivanović. *Hubs in Space.* JMLR 2010.

**Cited for grounding.**

- Aamari, Levrard. *Non-Asymptotic Rates for Manifold, Tangent Space, and Curvature Estimation.*
  Annals of Statistics 47(1), 2019. arXiv:1705.00989.
- Chen, Latifi Jebelli, et al. *Curvature of high-dimensional data.* arXiv:2511.02873.
- Cao, Li. *Efficient Weingarten map and curvature estimation on manifolds.* Machine Learning 110,
  2021.
- *Gaussian curvature in codimension > 1.* arXiv:1312.2554.
- Acosta et al. Manifold templates and analytic curvature. arXiv:2212.10414.

**Considered and not adopted.**

- Complexity Decoupled Chart Autoencoders, arXiv:2208.10570.
- RTD-AE, ICLR 2023, arXiv:2302.00136; Topological Autoencoders++, arXiv:2502.20215.
- GRAE, arXiv:2007.07142.
- Neuc-MDS, NeurIPS 2024, arXiv:2411.10889.
- Conformal decoder regularization for scalar curvature, arXiv:2508.20413.
- Ollivier-Ricci graph curvature (continuum limit at arXiv:2307.02378), recommended at one point as
  an embedding-free alternative and not pursued because it measures intrinsic rather than
  extrinsic curvature.

**Tools used in drafting.**

- Kassis, Agarwal, He, Patel, Brueckner. *Scientific Agent Skills: A Library of Procedural
  Knowledge for Research Agents.* 2026. arXiv:2609.00065.

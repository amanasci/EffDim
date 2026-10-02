8 of 8 encoders have records
8 of 8 encoders complete their battery (main_xfit, seed1, seed2, cf, thin, robust; main_d16 where d_run != 16)

## d per encoder

d_ID = median of mle, two_nn, tle, mind_mlk, rounded half to even (Python round), on a 10,000-row subsample of the row-normalised embeddings (one shared k-NN, k = 10); d_run = min(d_ID, 20).

| encoder | D | params | d_ID | d_run | mle | two_nn | tle | mind_mlk |
|---|---|---|---|---|---|---|---|---|
| chemberta_10m_mlm | 384 | 3,428,808 | 10 | 10 | 10.59 | 0.94 | 10.59 | 9.35 |
| chemberta_77m_mlm | 384 | 3,428,808 | 10 | 10 | 10.95 | 0.93 | 10.95 | 9.71 |
| chemberta_5m_mtr | 384 | 3,504,453 | 8 | 8 | 8.05 | 0.88 | 8.05 | 6.97 |
| chemberta_10m_mtr | 384 | 3,504,453 | 8 | 8 | 8.60 | 0.85 | 8.60 | 7.43 |
| chemberta_77m_mtr | 384 | 3,504,453 | 8 | 8 | 9.09 | 1.04 | 9.09 | 7.80 |
| molformer_xl | 768 | 46,805,760 | 11 | 11 | 11.71 | 6.93 | 11.71 | 10.19 |
| chemfm_1b | 2048 | 970,287,104 | 10 | 10 | 10.18 | 6.00 | 10.18 | 8.89 |
| chemfm_3b | 3072 | 3,004,357,632 | 9 | 9 | 10.09 | 5.98 | 10.09 | 8.85 |

In effdim, tle and mle are the same formula (the mean of the per-point Levina-Bickel estimate), so where tle == mle the median of the four is the mean of mle and the next value in sorted order. The pre-registered d is kept.

- tle equals mle (exact float equality) in 7 of 8 encoders
- chemberta_10m_mlm: middle pair mind_mlk, mle (d_ID = round of their mean)
- chemberta_77m_mlm: middle pair mind_mlk, mle (d_ID = round of their mean)
- chemberta_5m_mtr: middle pair mind_mlk, mle (d_ID = round of their mean)
- chemberta_10m_mtr: middle pair mind_mlk, mle (d_ID = round of their mean)
- chemberta_77m_mtr: middle pair mind_mlk, mle (d_ID = round of their mean)
- molformer_xl: middle pair mind_mlk, tle (d_ID = round of their mean)
- chemfm_1b: middle pair mind_mlk, mle (d_ID = round of their mean)
- chemfm_3b: middle pair mind_mlk, mle (d_ID = round of their mean)

## Synthetic intrinsic-dimension check (unit spheres)

Estimate minus true dimension; n = 10,000 points on a unit sphere of true dimension d, rotated into R^D.

| true d | D | mle | two_nn | tle | mind_mlk |
|---|---|---|---|---|---|
| 8 | 384 | +0.66 | -0.10 | +0.66 | -0.01 |
| 8 | 768 | +0.66 | -0.10 | +0.66 | -0.01 |
| 8 | 3072 | +0.66 | -0.10 | +0.66 | -0.01 |
| 16 | 384 | -0.30 | -1.62 | -0.30 | -1.46 |
| 16 | 768 | -0.30 | -1.62 | -0.30 | -1.46 |
| 16 | 3072 | -0.30 | -1.62 | -0.30 | -1.46 |
| 24 | 384 | -2.36 | -3.64 | -2.36 | -4.01 |
| 24 | 768 | -2.36 | -3.64 | -2.36 | -4.01 |
| 24 | 3072 | -2.36 | -3.64 | -2.36 | -4.01 |

- read low at true d = 24 in every D: mle, two_nn, tle, mind_mlk (of 4)

## Duplicate embeddings

Rows whose embedding equals another row's exactly (sweep/embedding_duplicates.py).

- chemberta_10m_mlm: 3,144 of 130,744 rows in 1,479 duplicate groups
- chemberta_77m_mlm: 3,144 of 130,744 rows in 1,479 duplicate groups
- chemberta_5m_mtr: 3,144 of 130,744 rows in 1,479 duplicate groups
- chemberta_10m_mtr: 3,144 of 130,744 rows in 1,479 duplicate groups
- chemberta_77m_mtr: 3,144 of 130,744 rows in 1,479 duplicate groups
- molformer_xl: 0 of 130,744 rows in 0 duplicate groups
- chemfm_1b: 0 of 130,744 rows in 0 duplicate groups
- chemfm_3b: 0 of 130,744 rows in 0 duplicate groups

## (a) Mismatch partial negative and significant

### at d_run (main_xfit)

- gap: 8 of 8
- mu: 6 of 8
- alpha: 8 of 8
- cv: 8 of 8

### at d = 16 (main_d16, or main_xfit where d_run = 16)

- gap: 8 of 8
- mu: 6 of 8
- alpha: 8 of 8
- cv: 8 of 8

## (c) Counterfactual: model-normal help exceeds random help, and help > 0.5 (at d_run)

- gap: help > random help in 8 of 8; help > 0.5 in 4 of 8
- mu: help > random help in 8 of 8; help > 0.5 in 4 of 8
- alpha: help > random help in 8 of 8; help > 0.5 in 3 of 8
- cv: help > random help in 8 of 8; help > 0.5 in 4 of 8

## (c') Counterfactual: sign reversal hurts (model-normal hurt > 0.5 and > random hurt, at d_run)

- gap: hurt > 0.5 and hurt > random hurt in 8 of 8
- mu: hurt > 0.5 and hurt > random hurt in 8 of 8
- alpha: hurt > 0.5 and hurt > random hurt in 8 of 8
- cv: hurt > 0.5 and hurt > random hurt in 8 of 8

## (d) Thinned-anchor sign test p_help < 0.05 (at d_run)

- gap: 2 of 8
- mu: 2 of 8
- alpha: 1 of 8
- cv: 1 of 8

## Alignment partial (descriptive; no molecular prior for its sign), at d_run

- gap: 0 negative-significant / 1 positive-significant / 7 non-significant (of 8)
- mu: 0 negative-significant / 5 positive-significant / 3 non-significant (of 8)
- alpha: 0 negative-significant / 2 positive-significant / 6 non-significant (of 8)
- cv: 1 negative-significant / 3 positive-significant / 4 non-significant (of 8)

## Per encoder and label

| encoder | d_run | label | mismatch @d_run | alignment @d_run | mismatch @16 | help | random help | hurt | p_help (thinned) |
|---|---|---|---|---|---|---|---|---|---|
| chemberta_10m_mlm | 10 | gap | -0.33 | +0.03 (ns) | -0.41 | 0.44 | 0.15 | 0.93 | 0.89 |
| chemberta_10m_mlm | 10 | mu | -0.17 | +0.13 | -0.34 | 0.46 | 0.24 | 0.86 | 0.7 |
| chemberta_10m_mlm | 10 | alpha | -0.44 | +0.18 | -0.37 | 0.48 | 0.12 | 0.99 | 0.11 |
| chemberta_10m_mlm | 10 | cv | -0.50 | +0.15 | -0.48 | 0.48 | 0.15 | 0.98 | 0.43 |
| chemberta_77m_mlm | 10 | gap | -0.36 | +0.09 | -0.44 | 0.53 | 0.20 | 0.96 | 0.43 |
| chemberta_77m_mlm | 10 | mu | -0.17 | +0.18 | -0.41 | 0.55 | 0.23 | 0.88 | 0.7 |
| chemberta_77m_mlm | 10 | alpha | -0.35 | +0.05 (ns) | -0.50 | 0.44 | 0.13 | 0.97 | 0.94 |
| chemberta_77m_mlm | 10 | cv | -0.30 | +0.10 | -0.54 | 0.52 | 0.20 | 0.94 | 0.43 |
| chemberta_5m_mtr | 8 | gap | -0.44 | -0.00 (ns) | -0.41 | 0.37 | 0.20 | 0.85 | 1 |
| chemberta_5m_mtr | 8 | mu | -0.14 | +0.03 (ns) | -0.19 | 0.37 | 0.26 | 0.82 | 0.89 |
| chemberta_5m_mtr | 8 | alpha | -0.21 | -0.03 (ns) | -0.15 | 0.29 | 0.16 | 0.95 | 1 |
| chemberta_5m_mtr | 8 | cv | -0.39 | -0.14 | -0.29 | 0.33 | 0.13 | 0.96 | 0.94 |
| chemberta_10m_mtr | 8 | gap | -0.51 | -0.03 (ns) | -0.51 | 0.28 | 0.23 | 0.84 | 1 |
| chemberta_10m_mtr | 8 | mu | -0.33 | +0.09 (ns) | -0.30 | 0.29 | 0.19 | 0.87 | 0.99 |
| chemberta_10m_mtr | 8 | alpha | -0.27 | +0.03 (ns) | -0.27 | 0.35 | 0.14 | 0.97 | 1 |
| chemberta_10m_mtr | 8 | cv | -0.40 | -0.03 (ns) | -0.45 | 0.36 | 0.12 | 0.98 | 0.97 |
| chemberta_77m_mtr | 8 | gap | -0.30 | -0.05 (ns) | -0.35 | 0.30 | 0.24 | 0.79 | 1 |
| chemberta_77m_mtr | 8 | mu | -0.09 | +0.06 (ns) | -0.28 | 0.34 | 0.30 | 0.74 | 1 |
| chemberta_77m_mtr | 8 | alpha | -0.22 | +0.02 (ns) | -0.39 | 0.45 | 0.23 | 0.93 | 0.96 |
| chemberta_77m_mtr | 8 | cv | -0.31 | -0.03 (ns) | -0.27 | 0.47 | 0.23 | 0.94 | 0.84 |
| molformer_xl | 11 | gap | -0.43 | +0.08 (ns) | -0.46 | 0.71 | 0.23 | 0.98 | 0.1 |
| molformer_xl | 11 | mu | -0.33 | +0.13 | -0.47 | 0.68 | 0.26 | 0.94 | 0.0081 |
| molformer_xl | 11 | alpha | -0.65 | +0.17 | -0.62 | 0.74 | 0.18 | 0.99 | 0.021 |
| molformer_xl | 11 | cv | -0.66 | +0.10 | -0.71 | 0.66 | 0.22 | 0.96 | 0.021 |
| chemfm_1b | 10 | gap | -0.16 | -0.04 (ns) | -0.18 | 0.64 | 0.32 | 0.92 | 0.018 |
| chemfm_1b | 10 | mu | +0.05 (ns) | +0.14 | +0.07 (ns) | 0.60 | 0.36 | 0.83 | 0.24 |
| chemfm_1b | 10 | alpha | -0.22 | +0.08 (ns) | -0.18 | 0.54 | 0.33 | 0.88 | 0.24 |
| chemfm_1b | 10 | cv | -0.34 | +0.04 (ns) | -0.35 | 0.60 | 0.34 | 0.86 | 0.15 |
| chemfm_3b | 9 | gap | -0.13 | -0.04 (ns) | -0.31 | 0.71 | 0.38 | 0.91 | 0.0081 |
| chemfm_3b | 9 | mu | +0.11 | +0.10 | +0.00 (ns) | 0.59 | 0.35 | 0.83 | 0.049 |
| chemfm_3b | 9 | alpha | -0.24 | -0.00 (ns) | -0.18 | 0.51 | 0.30 | 0.88 | 0.29 |
| chemfm_3b | 9 | cv | -0.26 | -0.07 (ns) | -0.32 | 0.57 | 0.39 | 0.82 | 0.43 |

## Encoders that break a claim

- (a) mismatch negative and significant at d_run, gap: none
- (a) mismatch negative and significant at d_run, mu: chemfm_1b, chemfm_3b
- (a) mismatch negative and significant at d_run, alpha: none
- (a) mismatch negative and significant at d_run, cv: none
- (a) mismatch negative and significant at d = 16, gap: none
- (a) mismatch negative and significant at d = 16, mu: chemfm_1b, chemfm_3b
- (a) mismatch negative and significant at d = 16, alpha: none
- (a) mismatch negative and significant at d = 16, cv: none
- (c) help > random help, gap: none
- (c) help > 0.5, gap: chemberta_10m_mlm, chemberta_5m_mtr, chemberta_10m_mtr, chemberta_77m_mtr
- (c) help > random help, mu: none
- (c) help > 0.5, mu: chemberta_10m_mlm, chemberta_5m_mtr, chemberta_10m_mtr, chemberta_77m_mtr
- (c) help > random help, alpha: none
- (c) help > 0.5, alpha: chemberta_10m_mlm, chemberta_77m_mlm, chemberta_5m_mtr, chemberta_10m_mtr, chemberta_77m_mtr
- (c) help > random help, cv: none
- (c) help > 0.5, cv: chemberta_10m_mlm, chemberta_5m_mtr, chemberta_10m_mtr, chemberta_77m_mtr
- (c') hurt > 0.5 and hurt > random hurt, gap: none
- (c') hurt > 0.5 and hurt > random hurt, mu: none
- (c') hurt > 0.5 and hurt > random hurt, alpha: none
- (c') hurt > 0.5 and hurt > random hurt, cv: none
- (d) sign test p_help < 0.05, gap: chemberta_10m_mlm, chemberta_77m_mlm, chemberta_5m_mtr, chemberta_10m_mtr, chemberta_77m_mtr, molformer_xl
- (d) sign test p_help < 0.05, mu: chemberta_10m_mlm, chemberta_77m_mlm, chemberta_5m_mtr, chemberta_10m_mtr, chemberta_77m_mtr, chemfm_1b
- (d) sign test p_help < 0.05, alpha: chemberta_10m_mlm, chemberta_77m_mlm, chemberta_5m_mtr, chemberta_10m_mtr, chemberta_77m_mtr, chemfm_1b, chemfm_3b
- (d) sign test p_help < 0.05, cv: chemberta_10m_mlm, chemberta_77m_mlm, chemberta_5m_mtr, chemberta_10m_mtr, chemberta_77m_mtr, chemfm_1b, chemfm_3b

## Reproduction guard (robust job)

- chemberta_10m_mlm: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- chemberta_77m_mlm: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- chemberta_5m_mtr: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- chemberta_10m_mtr: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- chemberta_77m_mtr: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- molformer_xl: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- chemfm_1b: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- chemfm_3b: exact PASS, 8 split cells and 36 counterfactual values, max |diff| 0 (split), 0 (counterfactual)
- not run: none

## Stale encoders

- none

## d file in the environment rows

molecules_d.json sha256 c17fb11e2e5dfd630f98b245e4331acdc36dad56c6b67c7aefd1824c73ef6e25; every split, cf and robust record should carry it.

- chemberta_10m_mlm: 6 of 6 records carry it
- chemberta_77m_mlm: 6 of 6 records carry it
- chemberta_5m_mtr: 6 of 6 records carry it
- chemberta_10m_mtr: 6 of 6 records carry it
- chemberta_77m_mtr: 6 of 6 records carry it
- molformer_xl: 6 of 6 records carry it
- chemfm_1b: 6 of 6 records carry it
- chemfm_3b: 6 of 6 records carry it

## Wall time and peak RSS per job

Exit 124: killed by the --timeout-h limit.

| job | exit | wall (h) | peak RSS (GB) |
|---|---|---|---|
| chemberta_10m_mlm__cf | 0 | 0.02 | 3.7 |
| chemberta_10m_mlm__main_d16 | 0 | 0.39 | 4.7 |
| chemberta_10m_mlm__main_xfit | 0 | 0.42 | 4.6 |
| chemberta_10m_mlm__robust | 0 | 0.18 | 4.0 |
| chemberta_10m_mlm__seed1 | 0 | 0.35 | 4.0 |
| chemberta_10m_mlm__seed2 | 0 | 0.34 | 4.0 |
| chemberta_10m_mlm__thin | 0 | 0.01 | 3.3 |
| chemberta_10m_mtr__cf | 0 | 0.02 | 3.7 |
| chemberta_10m_mtr__main_d16 | 0 | 0.37 | 4.7 |
| chemberta_10m_mtr__main_xfit | 0 | 0.41 | 4.5 |
| chemberta_10m_mtr__robust | 0 | 0.17 | 3.9 |
| chemberta_10m_mtr__seed1 | 0 | 0.34 | 4.0 |
| chemberta_10m_mtr__seed2 | 0 | 0.34 | 4.0 |
| chemberta_10m_mtr__thin | 0 | 0.01 | 3.3 |
| chemberta_5m_mtr__cf | 0 | 0.02 | 3.7 |
| chemberta_5m_mtr__main_d16 | 0 | 0.37 | 4.7 |
| chemberta_5m_mtr__main_xfit | 0 | 0.40 | 4.5 |
| chemberta_5m_mtr__robust | 0 | 0.17 | 3.9 |
| chemberta_5m_mtr__seed1 | 0 | 0.34 | 4.0 |
| chemberta_5m_mtr__seed2 | 0 | 0.34 | 4.0 |
| chemberta_5m_mtr__thin | 0 | 0.01 | 3.3 |
| chemberta_77m_mlm__cf | 0 | 0.02 | 3.7 |
| chemberta_77m_mlm__main_d16 | 0 | 0.38 | 4.7 |
| chemberta_77m_mlm__main_xfit | 0 | 0.41 | 4.6 |
| chemberta_77m_mlm__robust | 0 | 0.18 | 4.0 |
| chemberta_77m_mlm__seed1 | 0 | 0.35 | 4.0 |
| chemberta_77m_mlm__seed2 | 0 | 0.34 | 4.0 |
| chemberta_77m_mlm__thin | 0 | 0.01 | 3.3 |
| chemberta_77m_mtr__cf | 0 | 0.02 | 3.7 |
| chemberta_77m_mtr__main_d16 | 0 | 0.37 | 4.7 |
| chemberta_77m_mtr__main_xfit | 0 | 0.41 | 4.5 |
| chemberta_77m_mtr__robust | 0 | 0.17 | 3.9 |
| chemberta_77m_mtr__seed1 | 0 | 0.34 | 4.0 |
| chemberta_77m_mtr__seed2 | 0 | 0.36 | 4.0 |
| chemberta_77m_mtr__thin | 0 | 0.01 | 3.3 |
| chemfm_1b__cf | 0 | 0.07 | 15.9 |
| chemfm_1b__main_d16 | 0 | 0.62 | 18.6 |
| chemfm_1b__main_xfit | 0 | 0.66 | 18.4 |
| chemfm_1b__robust | 0 | 0.54 | 17.4 |
| chemfm_1b__seed1 | 0 | 0.55 | 16.1 |
| chemfm_1b__seed2 | 0 | 0.56 | 16.1 |
| chemfm_1b__thin | 0 | 0.03 | 15.5 |
| chemfm_3b__cf | 0 | 0.10 | 23.4 |
| chemfm_3b__main_d16 | 0 | 0.59 | 27.1 |
| chemfm_3b__main_xfit | 0 | 0.66 | 26.5 |
| chemfm_3b__robust | 0 | 0.89 | 25.1 |
| chemfm_3b__seed1 | 0 | 0.50 | 24.4 |
| chemfm_3b__seed2 | 0 | 0.50 | 24.4 |
| chemfm_3b__thin | 0 | 0.05 | 22.9 |
| molformer_xl__cf | 0 | 0.03 | 6.9 |
| molformer_xl__main_d16 | 0 | 0.40 | 8.1 |
| molformer_xl__main_xfit | 0 | 0.44 | 8.1 |
| molformer_xl__robust | 0 | 0.28 | 7.5 |
| molformer_xl__seed1 | 0 | 0.37 | 7.2 |
| molformer_xl__seed2 | 0 | 0.37 | 7.3 |
| molformer_xl__thin | 0 | 0.01 | 6.5 |

## Stated limits

- Only ChemFM 1B -> 3B is a model-size pair; the five ChemBERTa-2 models share one architecture (5M/10M/77M are pretraining-set sizes).
- The MTR models were pretrained on RDKit descriptors including molar refractivity (close to alpha), so alpha is expected near-linear for them.
- Neighbourhoods (k = 2,048 of 130,744 molecules) cover about 1/64 of the data (galaxies: 1/42).
- Special tokens are inside the mean pool.
- The ChemBERTa-2 tokenizer drops bracket-atom detail ([N+] -> N, [O-] -> O, [nH] -> n), so molecules that differ only there share an embedding (see Duplicate embeddings); zero nearest-neighbour distances pull their two_nn estimate below 1.
- ChemFM inputs carry no BOS (token id 1 is the atom 'He') and no trailing eos (its pretraining appended one).
- d = 20 is the cap and sits at an open question from the d = 20 spike findings; the d = 16 baseline covers it.

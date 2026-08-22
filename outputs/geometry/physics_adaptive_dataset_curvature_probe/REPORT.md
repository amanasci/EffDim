# Adaptive dataset curvature–physics probe

**Primary label:** `dataset_specific_curvature_probe_associations`

Discovery reference: ViT-B / `mag_r_desi` on Smith42/galaxies. It is shown and
is **not** counted as an independent confirmatory study.

Geometry freeze sha256 prefix: `d5acb1f615a73f1e`.

## 1. Which datasets possess valid aligned physics labels?

Smith42/galaxies (`physics_vit_base`) has a row-aligned `vit_base_test_labels.npz`
(`mag_r_desi`, `smooth_fraction`, `photo_z`, `stellar_mass`, `sfr`).
Smith42/DESI (`desi_vit_base_hsc`) has a cached catalog whose row count equals
the local ViT-B embedding table (n=20465): spectroscopic `Z` and `r_cmodel_mag`.

       dataset_id      raw_column canonical_label  valid_geometry_subset  include_in_association  underpowered
 physics_vit_base      mag_r_desi           mag_r                  16384                    True         False
 physics_vit_base smooth_fraction smooth_fraction                  16384                    True         False
 physics_vit_base         photo_z         photo_z                  15125                    True         False
 physics_vit_base    stellar_mass    stellar_mass                  15006                    True         False
 physics_vit_base             sfr             sfr                   1340                    True         False
desi_vit_base_hsc               Z          spec_z                  20465                    True         False
desi_vit_base_hsc           mag_r           mag_r                  20465                    True         False

## 2. Which were included or excluded, and why?

Included: ['physics_vit_base', 'desi_vit_base_hsc']

Excluded (not because associations were weak):

            dataset_id             inclusion_status                                                                                       exclusion_reason
 physics_dinov3_vitb16 inventory_only_other_encoder                                     additional encoder; primary replication is the frozen ViT-B family
     physics_clip_base inventory_only_other_encoder                                     additional encoder; primary replication is the frozen ViT-B family
 physics_convnext_base inventory_only_other_encoder                                     additional encoder; primary replication is the frozen ViT-B family
     physics_vit_large inventory_only_other_encoder                                     additional encoder; primary replication is the frozen ViT-B family
     jwst_vit_base_hsc                     excluded                         catalog n=1667 != embedding n=1496; also underpowered for k=2048 (n*0.125<256)
   legacy_vit_base_hsc                     excluded Smith42 catalog not loadable from the processed HF cache in this run; embeddings have no joined labels
              sdss_hsc                     excluded                                                                   no local ViT-B embeddings in data_hf
cosmosweb_vit_base_hsc                     excluded                                                      not a Smith42 catalog; labels not locally aligned

JWST is excluded because the Smith42 catalog has 1667 rows and the embedding
parquet has 1496 — positional join is refused. Legacy photometry exists in the
Smith42 hub cache but the processed datasets cache was not loadable. SDSS has
`Z` in the catalog and no local embeddings. CosmosWeb is not Smith42 and its
catalog columns are dropped by `prepare()`. Other encoders are inventory-only.

## 3. What dimensional interval was selected for each dataset using geometry alone?

       dataset_id  d_low  d_high  d_low_primary  d_high_primary  right_truncated  d_75  d_80  d_85  d_90  dL_plat  dQ_plat
 physics_vit_base      6      44              6              44            False     8    12    20    41      115       20
desi_vit_base_hsc      5      44              5              37            False     7    11    17    34       87       35

Labels were not loaded in this step. The range file was hashed in
`geometry_freeze.json` before associations.

## 4. Which datasets reached 80%, 85%, 90% and 95% held-out variance?

       dataset_id  d_80  d_85  d_90  d_95
 physics_vit_base    12    20    41   117
desi_vit_base_hsc    11    17    34    93

`not_reached` means the spectral pass never crossed that $\tau$. No
extrapolation was used.

## 5. Where did linear and quadratic reconstruction plateau?

       dataset_id  dL_plat  dQ_plat                 quadratic_source
 physics_vit_base      115       20 reused_physics_qpd_closest_point
desi_vit_base_hsc       87       35   coarse_fixed_then_fine_closest

Physics quadratic numbers are reused from the completed closest-point
`physics_quadratic_predictive_dimension` experiment.

## 6. Was any curvature sweep truncated by estimator identifiability?

       dataset_id  d_curv_max  curvature_range_right_truncated  right_truncated
 physics_vit_base          44                            False            False
desi_vit_base_hsc          44                            False            False

       dataset_id  d  valid_frac  median_R_H  fail_reliability  m_d
 physics_vit_base  6         1.0    0.515725             False   21
 physics_vit_base  7         1.0    0.515692             False   28
 physics_vit_base  8         1.0    0.524991             False   36
 physics_vit_base  9         1.0    0.512216             False   45
 physics_vit_base 10         1.0    0.527685             False   55
 physics_vit_base 11         1.0    0.521322             False   66
 physics_vit_base 12         1.0    0.526476             False   78
 physics_vit_base 13         1.0    0.535810             False   91
 physics_vit_base 14         1.0    0.528654             False  105
 physics_vit_base 15         1.0    0.527128             False  120
 physics_vit_base 16         1.0    0.514018             False  136
 physics_vit_base 17         1.0    0.514372             False  153
 physics_vit_base 18         1.0    0.509408             False  171
 physics_vit_base 19         1.0    0.514553             False  190
 physics_vit_base 20         1.0    0.515551             False  210
 physics_vit_base 21         1.0    0.514380             False  231
 physics_vit_base 22         1.0    0.513906             False  253
 physics_vit_base 23         1.0    0.514817             False  276
 physics_vit_base 24         1.0    0.509781             False  300
 physics_vit_base 25         1.0    0.507345             False  325
 physics_vit_base 26         1.0    0.503393             False  351
 physics_vit_base 27         1.0    0.495979             False  378
 physics_vit_base 28         1.0    0.486902             False  406
 physics_vit_base 29         1.0    0.479440             False  435
 physics_vit_base 30         1.0    0.473602             False  465
 physics_vit_base 31         1.0    0.469772             False  496
 physics_vit_base 32         1.0    0.464461             False  528
 physics_vit_base 33         1.0    0.456609             False  561
 physics_vit_base 34         1.0    0.452371             False  595
 physics_vit_base 35         1.0    0.443780             False  630
 physics_vit_base 36         1.0    0.439498             False  666
 physics_vit_base 37         1.0    0.433517             False  703
 physics_vit_base 38         1.0    0.424315             False  741
 physics_vit_base 39         1.0    0.418106             False  780
 physics_vit_base 40         1.0    0.413248             False  820
 physics_vit_base 41         1.0    0.403546             False  861
 physics_vit_base 42         1.0    0.400337             False  903
 physics_vit_base 43         1.0    0.396481             False  946
 physics_vit_base 44         1.0    0.389185             False  990
desi_vit_base_hsc  5         1.0    0.662178             False   15
desi_vit_base_hsc  6         1.0    0.690506             False   21
desi_vit_base_hsc  7         1.0    0.692064             False   28
desi_vit_base_hsc  8         1.0    0.679509             False   36
desi_vit_base_hsc  9         1.0    0.664327             False   45
desi_vit_base_hsc 10         1.0    0.660420             False   55
desi_vit_base_hsc 11         1.0    0.658871             False   66
desi_vit_base_hsc 12         1.0    0.647189             False   78
desi_vit_base_hsc 13         1.0    0.647825             False   91
desi_vit_base_hsc 14         1.0    0.642148             False  105
desi_vit_base_hsc 15         1.0    0.619873             False  120
desi_vit_base_hsc 16         1.0    0.599432             False  136
desi_vit_base_hsc 17         1.0    0.580282             False  153
desi_vit_base_hsc 18         1.0    0.565829             False  171
desi_vit_base_hsc 19         1.0    0.553273             False  190
desi_vit_base_hsc 20         1.0    0.547569             False  210
desi_vit_base_hsc 21         1.0    0.537322             False  231
desi_vit_base_hsc 22         1.0    0.529500             False  253
desi_vit_base_hsc 23         1.0    0.518422             False  276
desi_vit_base_hsc 24         1.0    0.515227             False  300
desi_vit_base_hsc 25         1.0    0.507392             False  325
desi_vit_base_hsc 26         1.0    0.498227             False  351
desi_vit_base_hsc 27         1.0    0.493954             False  378
desi_vit_base_hsc 28         1.0    0.484885             False  406
desi_vit_base_hsc 29         1.0    0.483544             False  435
desi_vit_base_hsc 30         1.0    0.472408             False  465
desi_vit_base_hsc 31         1.0    0.469203             False  496
desi_vit_base_hsc 32         1.0    0.466070             False  528
desi_vit_base_hsc 33         1.0    0.461316             False  561
desi_vit_base_hsc 34         1.0    0.461682             False  595
desi_vit_base_hsc 35         1.0    0.455297             False  630
desi_vit_base_hsc 36         1.0    0.451019             False  666
desi_vit_base_hsc 37         1.0    0.453798             False  703

## 7. Complete raw and controlled rank curves

       dataset_id           label  d       raw  controlled  p_ctl_fwer     r2_L
 physics_vit_base      mag_r_desi  6  0.069085    0.155230    0.009499 0.721537
 physics_vit_base      mag_r_desi  7  0.106099    0.221099    0.000100 0.741276
 physics_vit_base      mag_r_desi  8  0.155670    0.175882    0.001000 0.757538
 physics_vit_base      mag_r_desi  9  0.126495    0.089421    0.450355 0.771490
 physics_vit_base      mag_r_desi 10  0.079308    0.046875    0.986101 0.783746
 physics_vit_base      mag_r_desi 11  0.012933    0.010100    1.000000 0.794398
 physics_vit_base      mag_r_desi 12 -0.000821    0.000452    1.000000 0.803578
 physics_vit_base      mag_r_desi 13  0.016239   -0.051405    0.966803 0.811705
 physics_vit_base      mag_r_desi 14  0.065804    0.023141    1.000000 0.818971
 physics_vit_base      mag_r_desi 15  0.072738    0.060690    0.896410 0.825578
 physics_vit_base      mag_r_desi 16  0.053521    0.069245    0.788421 0.831572
 physics_vit_base      mag_r_desi 17  0.024817    0.082129    0.570343 0.837040
 physics_vit_base      mag_r_desi 18  0.035158    0.108368    0.199780 0.842065
 physics_vit_base      mag_r_desi 19  0.031400    0.100932    0.290071 0.846709
 physics_vit_base      mag_r_desi 20  0.027525    0.106967    0.215478 0.851005
 physics_vit_base      mag_r_desi 21  0.014392    0.085484    0.512149 0.854986
 physics_vit_base      mag_r_desi 22  0.015332    0.100218    0.299370 0.858688
 physics_vit_base      mag_r_desi 23  0.019362    0.112476    0.163384 0.862152
 physics_vit_base      mag_r_desi 24 -0.007632    0.093681    0.387761 0.865371
 physics_vit_base      mag_r_desi 25 -0.029032    0.073323    0.723428 0.868382
 physics_vit_base      mag_r_desi 26 -0.031218    0.055708    0.941606 0.871248
 physics_vit_base      mag_r_desi 27 -0.030035    0.059457    0.909109 0.873954
 physics_vit_base      mag_r_desi 28 -0.012797    0.061757    0.883912 0.876514
 physics_vit_base      mag_r_desi 29 -0.013364    0.040905    0.997100 0.878931
 physics_vit_base      mag_r_desi 30  0.008921    0.058786    0.915208 0.881238
 physics_vit_base      mag_r_desi 31 -0.046631    0.000353    1.000000 0.883430
 physics_vit_base      mag_r_desi 32  0.000180    0.052319    0.961804 0.885531
 physics_vit_base      mag_r_desi 33 -0.017384    0.058404    0.917808 0.887524
 physics_vit_base      mag_r_desi 34 -0.021675    0.069673    0.781622 0.889431
 physics_vit_base      mag_r_desi 35 -0.032404    0.088295    0.469053 0.891255
 physics_vit_base      mag_r_desi 36 -0.040842    0.079825    0.612739 0.893012
 physics_vit_base      mag_r_desi 37  0.002493    0.115739    0.138486 0.894700
 physics_vit_base      mag_r_desi 38 -0.013302    0.110516    0.180082 0.896307
 physics_vit_base      mag_r_desi 39 -0.025764    0.116721    0.130787 0.897855
 physics_vit_base      mag_r_desi 40 -0.018251    0.123169    0.091391 0.899350
 physics_vit_base      mag_r_desi 41 -0.001201    0.136595    0.036596 0.900781
 physics_vit_base      mag_r_desi 42  0.031899    0.157610    0.007799 0.902167
 physics_vit_base      mag_r_desi 43  0.015919    0.138603    0.031297 0.903508
 physics_vit_base      mag_r_desi 44 -0.006117    0.119509    0.114489 0.904800
 physics_vit_base smooth_fraction  6 -0.039356    0.073900    0.782122 0.721537
 physics_vit_base smooth_fraction  7 -0.102510    0.078749    0.700730 0.741276
 physics_vit_base smooth_fraction  8 -0.022617    0.072693    0.799820 0.757538
 physics_vit_base smooth_fraction  9 -0.022323    0.050185    0.989201 0.771490
 physics_vit_base smooth_fraction 10  0.020921    0.090069    0.507549 0.783746
 physics_vit_base smooth_fraction 11 -0.057665    0.014381    1.000000 0.794398
 physics_vit_base smooth_fraction 12 -0.063984    0.019469    1.000000 0.803578
 physics_vit_base smooth_fraction 13 -0.055685   -0.007144    1.000000 0.811705
 physics_vit_base smooth_fraction 14 -0.127438    0.022162    1.000000 0.818971
 physics_vit_base smooth_fraction 15 -0.146466    0.093090    0.455654 0.825578
 physics_vit_base smooth_fraction 16 -0.222870    0.105447    0.279572 0.831572
 physics_vit_base smooth_fraction 17 -0.282475    0.088994    0.524048 0.837040
 physics_vit_base smooth_fraction 18 -0.306209    0.072257    0.806219 0.842065
 physics_vit_base smooth_fraction 19 -0.319248    0.040574    0.999400 0.846709
 physics_vit_base smooth_fraction 20 -0.308730    0.066017    0.882912 0.851005
 physics_vit_base smooth_fraction 21 -0.307268    0.063040    0.916808 0.854986
 physics_vit_base smooth_fraction 22 -0.301074    0.094944    0.425057 0.858688
 physics_vit_base smooth_fraction 23 -0.303648    0.122790    0.117188 0.862152
 physics_vit_base smooth_fraction 24 -0.332593    0.100347    0.347365 0.865371
 physics_vit_base smooth_fraction 25 -0.323095    0.136656    0.051595 0.868382
 physics_vit_base smooth_fraction 26 -0.345062    0.099662    0.358064 0.871248
 physics_vit_base smooth_fraction 27 -0.318384    0.127347    0.091991 0.873954
 physics_vit_base smooth_fraction 28 -0.317714    0.132789    0.065893 0.876514
 physics_vit_base smooth_fraction 29 -0.304147    0.114292    0.188681 0.878931
 physics_vit_base smooth_fraction 30 -0.305209    0.110913    0.219478 0.881238
 physics_vit_base smooth_fraction 31 -0.334915    0.104789    0.288071 0.883430
 physics_vit_base smooth_fraction 32 -0.312861    0.085029    0.592741 0.885531
 physics_vit_base smooth_fraction 33 -0.316037    0.138905    0.044196 0.887524
 physics_vit_base smooth_fraction 34 -0.347251    0.095620    0.415058 0.889431
 physics_vit_base smooth_fraction 35 -0.362501    0.093785    0.443056 0.891255
 physics_vit_base smooth_fraction 36 -0.354236    0.102660    0.316668 0.893012
 physics_vit_base smooth_fraction 37 -0.349579    0.121790    0.124788 0.894700
 physics_vit_base smooth_fraction 38 -0.309455    0.197612    0.000200 0.896307
 physics_vit_base smooth_fraction 39 -0.334494    0.187134    0.000400 0.897855
 physics_vit_base smooth_fraction 40 -0.306682    0.190467    0.000400 0.899350
 physics_vit_base smooth_fraction 41 -0.327880    0.195407    0.000300 0.900781
 physics_vit_base smooth_fraction 42 -0.284824    0.229283    0.000100 0.902167
 physics_vit_base smooth_fraction 43 -0.313468    0.237771    0.000100 0.903508
 physics_vit_base smooth_fraction 44 -0.306193    0.210864    0.000100 0.904800
 physics_vit_base         photo_z  6 -0.138075   -0.078463    0.677232 0.721537
 physics_vit_base         photo_z  7 -0.087860   -0.051614    0.975602 0.741276
 physics_vit_base         photo_z  8 -0.064557   -0.018583    1.000000 0.757538
 physics_vit_base         photo_z  9 -0.155086   -0.089078    0.507749 0.771490
 physics_vit_base         photo_z 10 -0.200057   -0.114437    0.175482 0.783746
 physics_vit_base         photo_z 11 -0.226772   -0.164411    0.008999 0.794398
 physics_vit_base         photo_z 12 -0.228472   -0.188333    0.001200 0.803578
 physics_vit_base         photo_z 13 -0.143091   -0.117889    0.150885 0.811705
 physics_vit_base         photo_z 14 -0.050003   -0.031865    0.999900 0.818971
 physics_vit_base         photo_z 15 -0.026562   -0.017615    1.000000 0.825578
 physics_vit_base         photo_z 16  0.024220    0.045725    0.991801 0.831572
 physics_vit_base         photo_z 17  0.057920    0.111246    0.203180 0.837040
 physics_vit_base         photo_z 18  0.057248    0.093238    0.437956 0.842065
 physics_vit_base         photo_z 19  0.071089    0.135564    0.059494 0.846709
 physics_vit_base         photo_z 20  0.076226    0.141576    0.042796 0.851005
 physics_vit_base         photo_z 21  0.086761    0.174100    0.004500 0.854986
 physics_vit_base         photo_z 22  0.082213    0.185110    0.001600 0.858688
 physics_vit_base         photo_z 23  0.089291    0.187905    0.001200 0.862152
 physics_vit_base         photo_z 24  0.061748    0.125181    0.103490 0.865371
 physics_vit_base         photo_z 25  0.060911    0.132599    0.069193 0.868382
 physics_vit_base         photo_z 26  0.020500    0.087289    0.536846 0.871248
 physics_vit_base         photo_z 27 -0.017654    0.046428    0.990501 0.873954
 physics_vit_base         photo_z 28 -0.011324    0.054486    0.963104 0.876514
 physics_vit_base         photo_z 29  0.000354    0.067524    0.838616 0.878931
 physics_vit_base         photo_z 30  0.012714    0.086868    0.543446 0.881238
 physics_vit_base         photo_z 31 -0.010803    0.058766    0.932907 0.883430
 physics_vit_base         photo_z 32  0.026024    0.103587    0.290571 0.885531
 physics_vit_base         photo_z 33  0.020809    0.095527    0.400660 0.887524
 physics_vit_base         photo_z 34  0.020402    0.101337    0.318368 0.889431
 physics_vit_base         photo_z 35  0.033063    0.110280    0.213979 0.891255
 physics_vit_base         photo_z 36  0.042332    0.114648    0.173783 0.893012
 physics_vit_base         photo_z 37  0.016861    0.087356    0.536346 0.894700
 physics_vit_base         photo_z 38 -0.025572    0.044941    0.993001 0.896307
 physics_vit_base         photo_z 39 -0.039999    0.029399    1.000000 0.897855
 physics_vit_base         photo_z 40 -0.034161    0.026760    1.000000 0.899350
 physics_vit_base         photo_z 41 -0.027879    0.035722    0.999500 0.900781
 physics_vit_base         photo_z 42 -0.007775    0.057672    0.941806 0.902167
 physics_vit_base         photo_z 43 -0.028872    0.030930    0.999900 0.903508
 physics_vit_base         photo_z 44 -0.019161    0.045920    0.991401 0.904800
 physics_vit_base    stellar_mass  6 -0.134156   -0.004371    1.000000 0.721537
 physics_vit_base    stellar_mass  7 -0.111501    0.049259    0.987001 0.741276
 physics_vit_base    stellar_mass  8 -0.147963   -0.027493    1.000000 0.757538
 physics_vit_base    stellar_mass  9 -0.246772   -0.046585    0.992601 0.771490
 physics_vit_base    stellar_mass 10 -0.249845   -0.011250    1.000000 0.783746
 physics_vit_base    stellar_mass 11 -0.216101    0.000245    1.000000 0.794398
 physics_vit_base    stellar_mass 12 -0.193931   -0.019172    1.000000 0.803578
 physics_vit_base    stellar_mass 13 -0.140489    0.018915    1.000000 0.811705
 physics_vit_base    stellar_mass 14 -0.072685    0.003618    1.000000 0.818971
 physics_vit_base    stellar_mass 15 -0.060589   -0.033273    0.999900 0.825578
 physics_vit_base    stellar_mass 16  0.002671   -0.007138    1.000000 0.831572
 physics_vit_base    stellar_mass 17  0.061739    0.055915    0.961204 0.837040
 physics_vit_base    stellar_mass 18  0.048260    0.015724    1.000000 0.842065
 physics_vit_base    stellar_mass 19  0.060831    0.068847    0.837016 0.846709
 physics_vit_base    stellar_mass 20  0.073518    0.107031    0.268873 0.851005
 physics_vit_base    stellar_mass 21  0.075212    0.058860    0.942806 0.854986
 physics_vit_base    stellar_mass 22  0.076549    0.087846    0.541646 0.858688
 physics_vit_base    stellar_mass 23  0.072547    0.066464    0.869313 0.862152
 physics_vit_base    stellar_mass 24  0.066766    0.068416    0.842716 0.865371
 physics_vit_base    stellar_mass 25  0.061738    0.065993    0.875012 0.868382
 physics_vit_base    stellar_mass 26  0.027893    0.027616    1.000000 0.871248
 physics_vit_base    stellar_mass 27 -0.005846    0.014774    1.000000 0.873954
 physics_vit_base    stellar_mass 28 -0.026803    0.012359    1.000000 0.876514
 physics_vit_base    stellar_mass 29 -0.011774    0.019447    1.000000 0.878931
 physics_vit_base    stellar_mass 30 -0.033538   -0.019869    1.000000 0.881238
 physics_vit_base    stellar_mass 31 -0.008963    0.011286    1.000000 0.883430
 physics_vit_base    stellar_mass 32 -0.000427    0.020784    1.000000 0.885531
 physics_vit_base    stellar_mass 33  0.020744    0.034909    0.999900 0.887524
 physics_vit_base    stellar_mass 34  0.025321    0.031007    1.000000 0.889431
 physics_vit_base    stellar_mass 35  0.059912    0.058401    0.946605 0.891255
 physics_vit_base    stellar_mass 36  0.051481    0.047396    0.991001 0.893012
 physics_vit_base    stellar_mass 37  0.015381    0.010580    1.000000 0.894700
 physics_vit_base    stellar_mass 38 -0.039464   -0.008251    1.000000 0.896307
 physics_vit_base    stellar_mass 39 -0.039897    0.006394    1.000000 0.897855
 physics_vit_base    stellar_mass 40 -0.039321   -0.003378    1.000000 0.899350
 physics_vit_base    stellar_mass 41 -0.031228   -0.018791    1.000000 0.900781
 physics_vit_base    stellar_mass 42 -0.051490   -0.017367    1.000000 0.902167
 physics_vit_base    stellar_mass 43 -0.068077   -0.032004    1.000000 0.903508
 physics_vit_base    stellar_mass 44 -0.050641   -0.023699    1.000000 0.904800
 physics_vit_base             sfr  6 -0.170487   -0.139789    0.998800 0.721537
 physics_vit_base             sfr  7 -0.301581   -0.342161    0.320168 0.741276
 physics_vit_base             sfr  8 -0.152042   -0.304743    0.485551 0.757538
 physics_vit_base             sfr  9 -0.127009   -0.146772    0.997200 0.771490
 physics_vit_base             sfr 10 -0.276153   -0.259816    0.707829 0.783746
 physics_vit_base             sfr 11 -0.417128   -0.371410    0.216878 0.794398
 physics_vit_base             sfr 12 -0.314097   -0.275758    0.628837 0.803578
 physics_vit_base             sfr 13 -0.206192   -0.260211    0.705629 0.811705
 physics_vit_base             sfr 14 -0.003557   -0.150725    0.996400 0.818971
 physics_vit_base             sfr 15  0.140184    0.067062    1.000000 0.825578
 physics_vit_base             sfr 16  0.258235    0.155731    0.994301 0.831572
 physics_vit_base             sfr 17  0.234256    0.163900    0.989801 0.837040
 physics_vit_base             sfr 18  0.292622    0.262451    0.694931 0.842065
 physics_vit_base             sfr 19  0.278524    0.225033    0.856214 0.846709
 physics_vit_base             sfr 20  0.316601    0.317391    0.427857 0.851005
 physics_vit_base             sfr 21  0.246113    0.130962    0.999500 0.854986
 physics_vit_base             sfr 22  0.316206    0.269038    0.662034 0.858688
 physics_vit_base             sfr 23  0.350856    0.311594    0.453455 0.862152
 physics_vit_base             sfr 24  0.383399    0.347167    0.299970 0.865371
 physics_vit_base             sfr 25  0.373386    0.354941    0.271073 0.868382
 physics_vit_base             sfr 26  0.368643    0.353228    0.277772 0.871248
 physics_vit_base             sfr 27  0.401976    0.394335    0.149785 0.873954
 physics_vit_base             sfr 28  0.291041    0.230040    0.837516 0.876514
 physics_vit_base             sfr 29  0.261397    0.256522    0.723028 0.878931
 physics_vit_base             sfr 30  0.289328    0.200000    0.935406 0.881238
 physics_vit_base             sfr 31  0.255731    0.225823    0.854015 0.883430
 physics_vit_base             sfr 32  0.327668    0.302635    0.495650 0.885531
 physics_vit_base             sfr 33  0.460606    0.388669    0.165383 0.887524
 physics_vit_base             sfr 34  0.496179    0.453755    0.048995 0.889431
 physics_vit_base             sfr 35  0.446904    0.377997    0.196980 0.891255
 physics_vit_base             sfr 36  0.438076    0.372991    0.212079 0.893012
 physics_vit_base             sfr 37  0.461660    0.364032    0.241476 0.894700
 physics_vit_base             sfr 38  0.423320    0.339921    0.329467 0.896307
 physics_vit_base             sfr 39  0.353887    0.327404    0.382162 0.897855
 physics_vit_base             sfr 40  0.389328    0.321344    0.409159 0.899350
 physics_vit_base             sfr 41  0.367062    0.311462    0.454655 0.900781
 physics_vit_base             sfr 42  0.299078    0.237681    0.807319 0.902167
 physics_vit_base             sfr 43  0.308959    0.227668    0.846515 0.903508
 physics_vit_base             sfr 44  0.367852    0.291568    0.548845 0.904800
desi_vit_base_hsc          spec_z  5 -0.049183   -0.023333    1.000000 0.707539
desi_vit_base_hsc          spec_z  6  0.070836    0.097117    0.329767 0.734035
desi_vit_base_hsc          spec_z  7  0.017046    0.031242    1.000000 0.755324
desi_vit_base_hsc          spec_z  8 -0.051846   -0.042671    0.994801 0.772496
desi_vit_base_hsc          spec_z  9 -0.115943   -0.137230    0.036796 0.786850
desi_vit_base_hsc          spec_z 10 -0.083768   -0.126839    0.068293 0.798714
desi_vit_base_hsc          spec_z 11 -0.031261   -0.073248    0.720928 0.809027
desi_vit_base_hsc          spec_z 12 -0.102877   -0.166978    0.004000 0.818240
desi_vit_base_hsc          spec_z 13 -0.090853   -0.152228    0.010999 0.826449
desi_vit_base_hsc          spec_z 14 -0.075247   -0.126008    0.071993 0.833778
desi_vit_base_hsc          spec_z 15 -0.156812   -0.211514    0.000300 0.840375
desi_vit_base_hsc          spec_z 16 -0.198597   -0.242762    0.000100 0.846323
desi_vit_base_hsc          spec_z 17 -0.134644   -0.158161    0.008099 0.851690
desi_vit_base_hsc          spec_z 18 -0.163127   -0.193043    0.000500 0.856577
desi_vit_base_hsc          spec_z 19 -0.160787   -0.209415    0.000300 0.861045
desi_vit_base_hsc          spec_z 20 -0.102815   -0.158159    0.008099 0.865145
desi_vit_base_hsc          spec_z 21 -0.069040   -0.111543    0.165983 0.868898
desi_vit_base_hsc          spec_z 22 -0.052509   -0.098393    0.311569 0.872376
desi_vit_base_hsc          spec_z 23 -0.018388   -0.056387    0.934207 0.875615
desi_vit_base_hsc          spec_z 24 -0.012379   -0.048404    0.979502 0.878667
desi_vit_base_hsc          spec_z 25  0.013170   -0.009989    1.000000 0.881558
desi_vit_base_hsc          spec_z 26  0.030426    0.019339    1.000000 0.884272
desi_vit_base_hsc          spec_z 27  0.052792    0.047954    0.981702 0.886858
desi_vit_base_hsc          spec_z 28  0.057035    0.052521    0.960204 0.889290
desi_vit_base_hsc          spec_z 29  0.037593    0.038098    0.998300 0.891607
desi_vit_base_hsc          spec_z 30  0.038328    0.042559    0.994901 0.893806
desi_vit_base_hsc          spec_z 31  0.040647    0.031351    1.000000 0.895888
desi_vit_base_hsc          spec_z 32  0.026015    0.010610    1.000000 0.897882
desi_vit_base_hsc          spec_z 33  0.055675    0.044078    0.992401 0.899787
desi_vit_base_hsc          spec_z 34  0.039984    0.032028    0.999900 0.901608
desi_vit_base_hsc          spec_z 35  0.039382    0.020975    1.000000 0.903364
desi_vit_base_hsc          spec_z 36  0.076401    0.075545    0.681632 0.905040
desi_vit_base_hsc          spec_z 37  0.089535    0.088865    0.458054 0.906649
desi_vit_base_hsc           mag_r  5 -0.107640   -0.069417    0.798920 0.707539
desi_vit_base_hsc           mag_r  6 -0.022117    0.029908    0.999900 0.734035
desi_vit_base_hsc           mag_r  7 -0.086615   -0.025659    1.000000 0.755324
desi_vit_base_hsc           mag_r  8 -0.159754   -0.083725    0.559344 0.772496
desi_vit_base_hsc           mag_r  9 -0.183063   -0.137919    0.034397 0.786850
desi_vit_base_hsc           mag_r 10 -0.120419   -0.080725    0.615338 0.798714
desi_vit_base_hsc           mag_r 11 -0.075159   -0.057487    0.937806 0.809027
desi_vit_base_hsc           mag_r 12 -0.133415   -0.110238    0.184082 0.818240
desi_vit_base_hsc           mag_r 13 -0.118737   -0.106091    0.226977 0.826449
desi_vit_base_hsc           mag_r 14 -0.097424   -0.061683    0.896610 0.833778
desi_vit_base_hsc           mag_r 15 -0.141724   -0.089617    0.454755 0.840375
desi_vit_base_hsc           mag_r 16 -0.192549   -0.155620    0.008299 0.846323
desi_vit_base_hsc           mag_r 17 -0.117751   -0.078051    0.657934 0.851690
desi_vit_base_hsc           mag_r 18 -0.129700   -0.078401    0.651835 0.856577
desi_vit_base_hsc           mag_r 19 -0.129743   -0.094454    0.378862 0.861045
desi_vit_base_hsc           mag_r 20 -0.077933   -0.044114    0.994401 0.865145
desi_vit_base_hsc           mag_r 21 -0.020340    0.034189    0.999700 0.868898
desi_vit_base_hsc           mag_r 22  0.009835    0.073728    0.735526 0.872376
desi_vit_base_hsc           mag_r 23  0.050954    0.108614    0.200280 0.875615
desi_vit_base_hsc           mag_r 24  0.065705    0.102899    0.262274 0.878667
desi_vit_base_hsc           mag_r 25  0.077620    0.116587    0.132487 0.881558
desi_vit_base_hsc           mag_r 26  0.069784    0.097138    0.339766 0.884272
desi_vit_base_hsc           mag_r 27  0.093547    0.111469    0.173283 0.886858
desi_vit_base_hsc           mag_r 28  0.087904    0.093414    0.394661 0.889290
desi_vit_base_hsc           mag_r 29  0.056969    0.059882    0.915308 0.891607
desi_vit_base_hsc           mag_r 30  0.079356    0.101775    0.276872 0.893806
desi_vit_base_hsc           mag_r 31  0.099865    0.121530    0.099890 0.895888
desi_vit_base_hsc           mag_r 32  0.097044    0.119185    0.113289 0.897882
desi_vit_base_hsc           mag_r 33  0.109757    0.107880    0.208679 0.899787
desi_vit_base_hsc           mag_r 34  0.108336    0.124337    0.086191 0.901608
desi_vit_base_hsc           mag_r 35  0.109773    0.111041    0.176982 0.903364
desi_vit_base_hsc           mag_r 36  0.159501    0.173342    0.001800 0.905040
desi_vit_base_hsc           mag_r 37  0.153264    0.147115    0.017698 0.906649

## 8. Which associations survive within-dataset correction?

       dataset_id           label  d  controlled  p_ctl_fwer
 physics_vit_base smooth_fraction 33    0.138905    0.044196
 physics_vit_base smooth_fraction 38    0.197612    0.000200
 physics_vit_base smooth_fraction 39    0.187134    0.000400
 physics_vit_base smooth_fraction 40    0.190467    0.000400
 physics_vit_base smooth_fraction 41    0.195407    0.000300
 physics_vit_base smooth_fraction 42    0.229283    0.000100
 physics_vit_base smooth_fraction 43    0.237771    0.000100
 physics_vit_base smooth_fraction 44    0.210864    0.000100
 physics_vit_base         photo_z 11   -0.164411    0.008999
 physics_vit_base         photo_z 12   -0.188333    0.001200
 physics_vit_base         photo_z 20    0.141576    0.042796
 physics_vit_base         photo_z 21    0.174100    0.004500
 physics_vit_base         photo_z 22    0.185110    0.001600
 physics_vit_base         photo_z 23    0.187905    0.001200
 physics_vit_base             sfr 34    0.453755    0.048995
desi_vit_base_hsc          spec_z  9   -0.137230    0.036796
desi_vit_base_hsc          spec_z 12   -0.166978    0.004000
desi_vit_base_hsc          spec_z 13   -0.152228    0.010999
desi_vit_base_hsc          spec_z 15   -0.211514    0.000300
desi_vit_base_hsc          spec_z 16   -0.242762    0.000100
desi_vit_base_hsc          spec_z 17   -0.158161    0.008099
desi_vit_base_hsc          spec_z 18   -0.193043    0.000500
desi_vit_base_hsc          spec_z 19   -0.209415    0.000300
desi_vit_base_hsc          spec_z 20   -0.158159    0.008099
desi_vit_base_hsc           mag_r  9   -0.137919    0.034397
desi_vit_base_hsc           mag_r 16   -0.155620    0.008299
desi_vit_base_hsc           mag_r 36    0.173342    0.001800
desi_vit_base_hsc           mag_r 37    0.147115    0.017698

## 9. Which survive global dataset × label × dimension correction?

Global confirmatory $p$ (discovery excluded) = 0.0507.
Family: [{'dataset_id': 'physics_vit_base', 'canonical_label': 'smooth_fraction'}, {'dataset_id': 'physics_vit_base', 'canonical_label': 'photo_z'}, {'dataset_id': 'physics_vit_base', 'canonical_label': 'stellar_mass'}, {'dataset_id': 'physics_vit_base', 'canonical_label': 'sfr'}, {'dataset_id': 'desi_vit_base_hsc', 'canonical_label': 'spec_z'}, {'dataset_id': 'desi_vit_base_hsc', 'canonical_label': 'mag_r'}].
Same-object physics labels share one object permutation. DESI is an independent
sample and enters the joint maximum separately.

## 10. Does the ViT-B positive-core / negative-tail transition recur?

       dataset_id           label  mag_like  d_80  d_85  delta_85_80_raw  delta_85_80_ctl predicted_sign sign_consistent_raw
 physics_vit_base      mag_r_desi      True    12    20         0.028346         0.106515       negative               False
 physics_vit_base smooth_fraction     False    12    20        -0.244746         0.046549    not_assumed                 NaN
 physics_vit_base         photo_z     False    12    20         0.304697         0.329909    not_assumed                 NaN
 physics_vit_base    stellar_mass     False    12    20         0.267450         0.126203    not_assumed                 NaN
 physics_vit_base             sfr     False    12    20         0.630698         0.593149    not_assumed                 NaN
desi_vit_base_hsc          spec_z     False    11    17        -0.103383        -0.084913    not_assumed                 NaN
desi_vit_base_hsc           mag_r      True    11    17        -0.042593        -0.020565       negative                True

For magnitude labels with the same documented orientation as `mag_r_desi`, the
discovery-informed contrast $\Delta^{85-80}$ is predicted negative. Redshift
and morphology labels are not assumed to share that sign.

## 11. Absolute rank vs variance explained?

The variance-axis plots (`figures/07_heatmap_variance.png`,
`figures/08_discovery_variance_overlay.png`) are the primary cross-dataset
comparison. Absolute-rank heatmaps leave out-of-range cells blank. A common
numerical $d$ need not mean a common fraction of held-out energy.

## 12. Distribution of $\Delta^{85-80}$

       dataset_id           label  delta_85_80_ctl  mag_like  is_discovery
 physics_vit_base      mag_r_desi         0.106515      True          True
 physics_vit_base smooth_fraction         0.046549     False         False
 physics_vit_base         photo_z         0.329909     False         False
 physics_vit_base    stellar_mass         0.126203     False         False
 physics_vit_base             sfr         0.593149     False         False
desi_vit_base_hsc          spec_z        -0.084913     False         False
desi_vit_base_hsc           mag_r        -0.020565      True         False

## 13. Leave-one-dataset-out stability

         left_out  n  median_delta  frac_negative
desi_vit_base_hsc  0           NaN            NaN

## 14. Neighbourhood scale

Primary $k$ follows the frozen $0.125 n$ rule. Scale-sensitivity ranks are the
predeclared geometry ranks ($d_{80}$, $d_{85}$, $d_{90}$, plateaus), never
chosen from probe $\rho$.

       dataset_id    k   d           label  controlled              source
 physics_vit_base  256  12             NaN         NaN predeclared_pending
 physics_vit_base  256  20             NaN         NaN predeclared_pending
 physics_vit_base  256  41             NaN         NaN predeclared_pending
 physics_vit_base  256 115             NaN         NaN predeclared_pending
 physics_vit_base  512  12             NaN         NaN predeclared_pending
 physics_vit_base  512  20             NaN         NaN predeclared_pending
 physics_vit_base  512  41             NaN         NaN predeclared_pending
 physics_vit_base  512 115             NaN         NaN predeclared_pending
 physics_vit_base  768  12             NaN         NaN predeclared_pending
 physics_vit_base  768  20             NaN         NaN predeclared_pending
 physics_vit_base  768  41             NaN         NaN predeclared_pending
 physics_vit_base  768 115             NaN         NaN predeclared_pending
 physics_vit_base 1024  12             NaN         NaN predeclared_pending
 physics_vit_base 1024  20             NaN         NaN predeclared_pending
 physics_vit_base 1024  41             NaN         NaN predeclared_pending
 physics_vit_base 1024 115             NaN         NaN predeclared_pending
 physics_vit_base 1536  12             NaN         NaN predeclared_pending
 physics_vit_base 1536  20             NaN         NaN predeclared_pending
 physics_vit_base 1536  41             NaN         NaN predeclared_pending
 physics_vit_base 1536 115             NaN         NaN predeclared_pending
 physics_vit_base 2048  12      mag_r_desi    0.000452       primary_scale
 physics_vit_base 2048  12 smooth_fraction    0.019469       primary_scale
 physics_vit_base 2048  12         photo_z   -0.188333       primary_scale
 physics_vit_base 2048  12    stellar_mass   -0.019172       primary_scale
 physics_vit_base 2048  12             sfr   -0.275758       primary_scale
 physics_vit_base 2048  20      mag_r_desi    0.106967       primary_scale
 physics_vit_base 2048  20 smooth_fraction    0.066017       primary_scale
 physics_vit_base 2048  20         photo_z    0.141576       primary_scale
 physics_vit_base 2048  20    stellar_mass    0.107031       primary_scale
 physics_vit_base 2048  20             sfr    0.317391       primary_scale
 physics_vit_base 2048  41      mag_r_desi    0.136595       primary_scale
 physics_vit_base 2048  41 smooth_fraction    0.195407       primary_scale
 physics_vit_base 2048  41         photo_z    0.035722       primary_scale
 physics_vit_base 2048  41    stellar_mass   -0.018791       primary_scale
 physics_vit_base 2048  41             sfr    0.311462       primary_scale
 physics_vit_base 2048 115             NaN         NaN predeclared_pending
desi_vit_base_hsc  256  11             NaN         NaN predeclared_pending
desi_vit_base_hsc  256  17             NaN         NaN predeclared_pending
desi_vit_base_hsc  256  34             NaN         NaN predeclared_pending
desi_vit_base_hsc  256  35             NaN         NaN predeclared_pending
desi_vit_base_hsc  256  87             NaN         NaN predeclared_pending
desi_vit_base_hsc  512  11             NaN         NaN predeclared_pending
desi_vit_base_hsc  512  17             NaN         NaN predeclared_pending
desi_vit_base_hsc  512  34             NaN         NaN predeclared_pending
desi_vit_base_hsc  512  35             NaN         NaN predeclared_pending
desi_vit_base_hsc  512  87             NaN         NaN predeclared_pending
desi_vit_base_hsc  768  11             NaN         NaN predeclared_pending
desi_vit_base_hsc  768  17             NaN         NaN predeclared_pending
desi_vit_base_hsc  768  34             NaN         NaN predeclared_pending
desi_vit_base_hsc  768  35             NaN         NaN predeclared_pending
desi_vit_base_hsc  768  87             NaN         NaN predeclared_pending
desi_vit_base_hsc 1024  11             NaN         NaN predeclared_pending
desi_vit_base_hsc 1024  17             NaN         NaN predeclared_pending
desi_vit_base_hsc 1024  34             NaN         NaN predeclared_pending
desi_vit_base_hsc 1024  35             NaN         NaN predeclared_pending
desi_vit_base_hsc 1024  87             NaN         NaN predeclared_pending
desi_vit_base_hsc 1536  11             NaN         NaN predeclared_pending
desi_vit_base_hsc 1536  17             NaN         NaN predeclared_pending
desi_vit_base_hsc 1536  34             NaN         NaN predeclared_pending
desi_vit_base_hsc 1536  35             NaN         NaN predeclared_pending
desi_vit_base_hsc 1536  87             NaN         NaN predeclared_pending
desi_vit_base_hsc 2048  11          spec_z   -0.073248       primary_scale
desi_vit_base_hsc 2048  11           mag_r   -0.057487       primary_scale
desi_vit_base_hsc 2048  17          spec_z   -0.158161       primary_scale
desi_vit_base_hsc 2048  17           mag_r   -0.078051       primary_scale
desi_vit_base_hsc 2048  34          spec_z    0.032028       primary_scale
desi_vit_base_hsc 2048  34           mag_r    0.124337       primary_scale
desi_vit_base_hsc 2048  35          spec_z    0.020975       primary_scale
desi_vit_base_hsc 2048  35           mag_r    0.111041       primary_scale
desi_vit_base_hsc 2048  87             NaN         NaN predeclared_pending

## 15. Reliability, shrinkage, effective degrees of freedom

See `curvature_reliability.csv` and figure 15. Differences that appear only
where $R_H$ fails or $m(d)/n$ is extreme are not interpreted as physics-label
geometry.

## 16. What can and cannot be concluded

- $K_H^{(d)}$ is rank-conditioned curvature under a $d$-chart, not one
  geometric object.
- A maximizing $d$ is **not** intrinsic dimension, tangent dimension, or a
  claim that the manifold is $d$-dimensional.
- Smith42/galaxies and Smith42/DESI are the aligned ViT-B physics-labelled
  datasets used here. JWST/Legacy/SDSS/CosmosWeb remain in the manifest with
  explicit exclusion reasons.
- Discovery `mag_r_desi` is a reference, not a confirmatory replicate.
- Other physics labels on the same 16k galaxies are dependent; they are not
  independent studies.

Runtime: 44056.0 s. Permutations: 10000.
Bootstraps: 2000.

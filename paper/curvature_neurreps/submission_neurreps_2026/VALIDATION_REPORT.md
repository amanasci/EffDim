# VALIDATION_REPORT

{
  "parity": {
    "n_anchors": 512,
    "k": 2048,
    "target_id": "mag_r_desi_local_oof_r2",
    "catalog_field": "mag_r_desi_catalog_value",
    "not_catalog": true,
    "delta_20_12": -0.37631527091129646,
    "delta_20_12_frozen": -0.376315,
    "delta_match": true,
    "all_raw_match": true,
    "all_ctl_match": true,
    "exact_parity": true,
    "hashes": {
      "per_anchor_rank_curve": "badeac8ce6197b3f8e7a8f68f3d05b64e8d965d3d146120f09d04c9f7e1c0f4a",
      "local_probe_fields": "8deccdeba7401d73555132e9817b3a3c1f423c30f2759ad2c7c4934b1eb05df6",
      "vit_base_npz": "7635df38659bce31a3fd7bdff59c4d8f107dc674b33a43abd54e5fe4d9d7235a",
      "knn2048": "f0a6a730ba909b5869c0c624228620edac65cf6fd53d2d0590670aa63f4f5962"
    },
    "controls": [
      "log_knn_radius",
      "local_label_variance",
      "local_evaluation_count"
    ],
    "neighbourhood": "vit_base_kmax2048.npz k=2048, aligned by sample_id"
  },
  "probe": {
    "r2_ok": true
  },
  "scale": {
    "n_pending": 0
  },
  "decision": {
    "label": "claim_supported_but_scale_dependent",
    "reasons": [
      "scale_direction_or_magnitude_varies"
    ],
    "parity_ok": true,
    "primary_survives_fwer": true,
    "direct_error_ok": true,
    "error_hits": [
      {
        "target": "mag_r_desi_oof_sse",
        "rho16": 0.2270478922763529,
        "p_fwer": 0.0
      },
      {
        "target": "mag_r_desi_oof_mae",
        "rho16": 0.250721159348142,
        "p_fwer": 0.0
      },
      {
        "target": "mag_r_desi_oof_mse",
        "rho16": 0.2270478922763529,
        "p_fwer": 0.0
      },
      {
        "target": "mag_r_desi_oof_nmse",
        "rho16": 0.2404841119636992,
        "p_fwer": NaN
      },
      {
        "target": "mag_r_desi_normalized_mse",
        "rho16": 0.2270478922763529,
        "p_fwer": NaN
      }
    ],
    "denom_rhos_d16": {
      "mag_r_desi_local_sst": -0.0245571844475716,
      "mag_r_desi_local_target_var": -0.0245571844475716
    },
    "error_rhos_d16": {
      "mag_r_desi_oof_sse": 0.2270478922763529,
      "mag_r_desi_oof_mae": 0.250721159348142,
      "mag_r_desi_oof_mse": 0.2270478922763529,
      "mag_r_desi_oof_nmse": 0.2404841119636992,
      "mag_r_desi_normalized_mse": 0.2270478922763529
    },
    "shuffle_ok": true,
    "scale_sign_frac_mid_upper": 1.0,
    "scale_recurrent": true,
    "scale_magnitude_varies": true,
    "n_pending_scale": 0,
    "leakage_ok": true
  }
}

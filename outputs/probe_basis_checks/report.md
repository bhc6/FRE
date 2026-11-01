# Probe basis CPU checks

All 47 checks passed. NumPy 2.4.2, float64, seed 20261101, one process, two library threads.

These are synthetic numerical checks, not Qwen/VLM or QA results.

- Complete orthogonal Fourier versus time-domain ridge: maximum test-prediction discrepancy 1.056e-11.
- Missing sqrt(2) scaling with unchanged isotropic penalty: prediction discrepancy 2.40706; corrected penalty error 5.329e-15.
- Frequency-anisotropic penalty versus its rotated time penalty: error 1.421e-14.
- Reversal/circular-shift power discrepancies: 1.332e-15 / 1.332e-15.
- Time-varying local SwiGLU Jacobian: off-diagonal frequency-operator norm 1.91925; constant-J control 9.501e-16.

Full metrics: checks.json. Exact acceptance criteria: source check conditions. Small synthetic examples: examples.npz.

| Check | Status |
| --- | --- |
| complete_real_fourier_is_orthogonal_T1 | PASS |
| dc_is_single_unscaled_endpoint_T1 | PASS |
| complete_real_fourier_is_orthogonal_T2 | PASS |
| dc_is_single_unscaled_endpoint_T2 | PASS |
| even_nyquist_is_single_unscaled_endpoint_T2 | PASS |
| complete_real_fourier_is_orthogonal_T3 | PASS |
| dc_is_single_unscaled_endpoint_T3 | PASS |
| odd_last_rfft_bin_retains_imaginary_component_T3 | PASS |
| complete_real_fourier_is_orthogonal_T7 | PASS |
| dc_is_single_unscaled_endpoint_T7 | PASS |
| odd_last_rfft_bin_retains_imaginary_component_T7 | PASS |
| complete_real_fourier_is_orthogonal_T8 | PASS |
| dc_is_single_unscaled_endpoint_T8 | PASS |
| even_nyquist_is_single_unscaled_endpoint_T8 | PASS |
| complete_real_fourier_is_orthogonal_T31 | PASS |
| dc_is_single_unscaled_endpoint_T31 | PASS |
| odd_last_rfft_bin_retains_imaginary_component_T31 | PASS |
| complete_real_fourier_is_orthogonal_T32 | PASS |
| dc_is_single_unscaled_endpoint_T32 | PASS |
| even_nyquist_is_single_unscaled_endpoint_T32 | PASS |
| linear_function_class_and_l2_norm_equivalent_T7 | PASS |
| ridge_same_predictions_T7_lambda0.03 | PASS |
| ridge_same_predictions_T7_lambda0.7 | PASS |
| ridge_same_predictions_T7_lambda12.0 | PASS |
| zero_ridge_minimum_euclidean_norm_solution_equivalent_T7 | PASS |
| linear_function_class_and_l2_norm_equivalent_T8 | PASS |
| ridge_same_predictions_T8_lambda0.03 | PASS |
| ridge_same_predictions_T8_lambda0.7 | PASS |
| ridge_same_predictions_T8_lambda12.0 | PASS |
| zero_ridge_minimum_euclidean_norm_solution_equivalent_T8 | PASS |
| linear_function_class_and_l2_norm_equivalent_T32 | PASS |
| ridge_same_predictions_T32_lambda0.03 | PASS |
| ridge_same_predictions_T32_lambda0.7 | PASS |
| ridge_same_predictions_T32_lambda12.0 | PASS |
| zero_ridge_minimum_euclidean_norm_solution_equivalent_T32 | PASS |
| omitting_sqrt2_changes_isotropic_ridge_but_not_invertibility | PASS |
| anisotropic_frequency_regularizer_equals_rotated_time_regularizer | PASS |
| truncated_spectral_probe_equals_same_time_subspace | PASS |
| time_mean_removal_equals_dc_projection_not_intercept_removal | PASS |
| independent_per_coordinate_standardization_changes_ridge_prior | PASS |
| unwindowed_power_cannot_distinguish_reversal_or_circular_shift | PASS |
| phase_preserving_linear_probe_can_separate_one_reversal_pair | PASS |
| linear_power_probe_is_a_restricted_time_quadratic_probe | PASS |
| trajectory_local_swiglu_jacobian_is_generally_time_varying | PASS |
| constant_local_jacobian_is_frequency_diagonal | PASS |
| time_varying_jacobian_fourier_mixing_formula | PASS |
| invalid_complex_empty_nonfinite_scalar_inputs_rejected | PASS |

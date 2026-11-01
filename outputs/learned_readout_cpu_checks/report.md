# Learned readout CPU checks

All 60 checks passed; two CPU threads and one interop thread, no GPU or model loading.

- Each MLP has 18,584 parameters; the constant spectral controller has 8.
- All four methods start at exactly zero residual and satisfy per-head gain and residual caps.
- Gradients are finite; at zero initialization the output layer learns first, then the input layer receives gradients after an update.
- Parameter directional-derivative error: 4.502e-14.
- DC/Nyquist/odd-length cases, context reconstruction and circular-lag equivalence are checked.
- The magnitude conditioner preserves complex-bin magnitude; it does not remove phase from the signed intervention.
- Uniform-attention counterexample: cross-spectrum collision error 0.000e+00, time-feature difference 0.52008.

The time and spectral implementations are matched in parameter count, but their feature classes differ. Complete FFT of the time-contribution basis is only a coordinate rotation; it is not the cross-spectrum used here.

Exact zero delta does not certify cached-tail BF16 logit parity. Final evaluation must use the original forward with the local attention-output injection.

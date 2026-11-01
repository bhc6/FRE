# Temporal spectral adapter V3 protocol

This module is a parameter-matched development control, not a confirmed
memory method. Both adapters accept `[B,T,P,D]` or `[B,T,H,W,D]` and return
the identical layout. They use the same `D→rank→D` projections, `T` gate
parameters, zero-initialized output projection, gate bound and residual cap.

The time control applies the bottleneck independently at each time group. The
spectral control applies it to the orthonormal real FFT along T and reconstructs
with an inverse FFT. Neither operation transforms the feature channel axis.
The spectral control therefore tests a temporal Fourier parameterization under
matched capacity; it does not establish that Fourier coordinates contain more
information than an equivalent time-domain basis.

Required checks:

- zero-initialized output is an exact identity in float32;
- both adapters have equal parameter counts for the same configuration;
- `[B,T,P,D]` and `[B,T,H,W,D]` layouts round-trip identically;
- odd and even T values are supported;
- finite floating inputs are required and non-floating/nonfinite inputs fail;
- residual norm never exceeds `residual_limit * input_norm` per token;
- gate values remain within `[-max_gate,max_gate]`;
- output dtype and shape match the input;
- spectral forward uses only T FFT and preserves the channel axis.

The adapter is deliberately independent of model-specific hooks. A later
Qwen/Smol runner must inject it at a frozen encoder→merger/connector boundary,
record model revision and feature hashes, and compare it with the matched
time-domain control before any new confirmation-set run.

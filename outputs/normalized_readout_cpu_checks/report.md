# Normalized readout CPU checks

108 / 108 passed. Source SHA256: `6c8ab1f65ec8b50b3de318402b465de96ba9ac0e28b2d25226fb0d6b61eaad15`.

Checks cover parameter counts, exact zero initialization, runtime diagnostic compatibility, independent NumPy arithmetic, positive per-direction scaling invariance above epsilon, common basis/context scaling equivariance, signed spectral input, zero directions/context, finite and nonzero parameter gradients, staged first-layer gradients, state-dict replay, 10%/30% per-head caps, relative-epsilon attenuation, finite differences with active/inactive caps, and invalid dimensions/data. V1 sources were unchanged.

No GPU, pretrained backbone, or task evaluation was used. Normalization changes the scale hypothesis; it is not a claim of a new spectral-memory mechanism.

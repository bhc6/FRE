"""CPU verification of residual scaling versus reciprocal update damping."""

from __future__ import annotations

import json
from pathlib import Path

from .resource_guard import configure_threads


def main():
    configure_threads()
    import torch
    from transformers import LlamaConfig, LlamaModel
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
    from .controller_runtime import digest, write_json

    torch.set_num_threads(2)
    torch.manual_seed(20261101)
    rows = []
    states = torch.randn(7, 64, dtype=torch.float64)
    weights = torch.randn(64, dtype=torch.float64)
    for epsilon in (0., 1e-6):
        for scale in (.5, 2., 4.):
            observed = weights * scale * states / (scale**2 * states.square().mean(dim=-1, keepdim=True) + epsilon).sqrt()
            predicted = weights * states / (states.square().mean(dim=-1, keepdim=True) + epsilon / scale**2).sqrt()
            error = float((observed - predicted).abs().max())
            assert error < 1e-12
            rows.append({"test": "RMSNorm_exact_relation", "epsilon": epsilon, "scale": scale, "maximum_error": error})
    for model_name in ("llama", "qwen3"):
        for epsilon in (0., 1e-6):
            torch.manual_seed(20261101)
            settings = {"vocab_size": 128, "hidden_size": 64, "intermediate_size": 96,
                        "num_hidden_layers": 4, "num_attention_heads": 4,
                        "num_key_value_heads": 2, "head_dim": 16, "attention_dropout": 0.,
                        "rms_norm_eps": epsilon}
            if model_name == "llama":
                configuration = LlamaConfig(**settings)
                configuration._attn_implementation = "sdpa"
                model = LlamaModel(configuration).eval()
            else:
                configuration = Qwen3VLTextConfig(**settings, rope_scaling={"rope_type": "default", "mrope_section": [4, 2, 2]})
                configuration._attn_implementation = "sdpa"
                model = Qwen3VLTextModel(configuration).eval()
            input_ids = torch.randint(0, 128, (1, 24))
            selected_positions = torch.tensor([2, 6, 10, 14])

            def forward(mode, scale):
                handles = []

                def scale_state(module, arguments, output):
                    values = output[0] if isinstance(output, tuple) else output
                    replaced = values.clone()
                    replaced[:, selected_positions] *= scale
                    return (replaced, *output[1:]) if isinstance(output, tuple) else replaced

                def damp_update(module, arguments, output):
                    values = output[0] if isinstance(output, tuple) else output
                    replaced = values.clone()
                    replaced[:, selected_positions] /= scale
                    return (replaced, *output[1:]) if isinstance(output, tuple) else replaced

                try:
                    if mode == "scale":
                        handles.append(model.layers[1].register_forward_hook(scale_state))
                    if mode == "damp":
                        for layer in model.layers[2:]:
                            handles.extend((layer.self_attn.register_forward_hook(damp_update),
                                            layer.mlp.register_forward_hook(damp_update)))
                    with torch.inference_mode():
                        return model(input_ids=input_ids, use_cache=False, return_dict=True).last_hidden_state.clone()
                finally:
                    for handle in handles:
                        handle.remove()

            native = forward("native", 1.)
            for scale in (.5, 2., 4.):
                scaled = forward("scale", scale)
                damped = forward("damp", scale)
                error = float((scaled - damped).abs().max())
                threshold = 2e-5 if epsilon == 0. else .03
                assert error < threshold, (model_name, epsilon, scale, error)
                assert torch.equal(native, forward("native", 1.))
                rows.append({"test": "actual_tiny_pre_norm_architecture", "model": model_name,
                             "epsilon": epsilon, "scale": scale, "maximum_error": error,
                             "last_answer_maximum_error": float((scaled[:, -1] - damped[:, -1]).abs().max()),
                             "native_restored_exact": True, "dtype": "float32", "threshold": threshold})
    output = Path("outputs/delimiter_gauge_cpu_v2")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "report.json", {"status": "passed", "checks": rows,
                                        "source_sha256": digest(Path(__file__)),
                                        "scope": "mathematics and tiny real text architectures; no checkpoint/GPU task evidence"})
    print(json.dumps({"status": "passed", "checks": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()

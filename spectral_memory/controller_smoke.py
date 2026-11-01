"""Non-selection smoke checks on one already-seen training source."""

from __future__ import annotations

import argparse
from pathlib import Path

from .resource_guard import ResourceGuard, configure_threads


def run(arguments):
    configure_threads()
    import numpy as np
    import torch
    from PIL import Image
    from .controller_runtime import FrozenTail, controlled_forward, digest, read_rows, save_tail, write_json
    from .controller_train import METHODS
    from .endpoint_control import check_deadline
    from .evaluate_small import build_prompt
    from .learned_readout import ReadoutController
    from .passive_probe import forward_logits
    from .qwen3_backend import DEFAULT_MAX_PIXELS, load_backend

    output = arguments.output_dir.resolve()
    if output.exists():
        raise ValueError("Smoke output must be new")
    output.mkdir(parents=True)
    question = read_rows(Path("data/internal_controller_v1/train.questions.jsonl"))[0]
    sources = {row["video_id"]: row for row in read_rows(Path("data/internal_controller_v1/frames16/train.frames.manifest.jsonl"))}
    source = sources[question["video_id"]]
    report = {"purpose": "nonselection_implementation_smoke_on_seen_train_source", "query_id": question["query_id"],
              "source_sha256": {name: digest(Path(__file__).with_name(name)) for name in
                                 ("controller_smoke.py", "controller_runtime.py", "learned_readout.py")}, "checks": []}
    with ResourceGuard(output, "cuda", timeout_seconds=300) as guard:
        guard.configure_torch()
        backend = load_backend(device="cuda")
        option_ids = [backend.processor.tokenizer.encode(label, add_special_tokens=False)[0] for label in "ABC"]
        manifest = save_tail(backend, output / "frozen_tail.pt", option_ids)
        write_json(output / "frozen_tail_manifest.json", manifest)
        tail = FrozenTail(torch.load(output / "frozen_tail.pt", map_location="cpu", weights_only=True)).to("cuda").eval()
        frames = [Image.open(path).convert("RGB") for path in source["frames"]]
        for condition in ("real", "static_first", "static_last"):
            check_deadline(guard)
            guard.phase = "smoke:" + condition
            images = frames if condition == "real" else [frames[0 if condition == "static_first" else -1]] * 16
            batch = backend.prepare_video(images, source["timestamps_s"], build_prompt(question), DEFAULT_MAX_PIXELS)
            native = forward_logits(backend, batch)
            passive, arrays, metadata = controlled_forward(backend, batch, retain_features=True)
            if not np.array_equal(native, passive):
                raise RuntimeError("Passive smoke parity failed")
            tensors = {name: torch.from_numpy(value.copy()).to("cuda") for name, value in arrays.items() if value.dtype.kind == "f"}
            for method in METHODS:
                controller = ReadoutController(method).to("cuda").eval()
                identity, _, zero_metadata = controlled_forward(backend, batch, controller)
                if not np.array_equal(native, identity) or zero_metadata["delta_context_l2"] != 0.0:
                    raise RuntimeError("Controller smoke identity failed")
                with torch.no_grad():
                    parameter = controller.constant_gains if method == "constant_spectral" else controller.network[-1].bias
                    parameter.copy_(torch.linspace(-0.3, 0.3, 8, device="cuda"))
                online, _, intervention = controlled_forward(backend, batch, controller)
                bases = {name: tensors["basis_" + name][None] for name in ("time", "spectral", "phase_ablation")}
                delta, _ = controller(tensors["query"][None], bases, tensors["context"][None])
                replay = tail(tensors["decoder_input"][None], tensors["attention_delta"][None], delta)
                loss = torch.nn.functional.cross_entropy(replay, torch.as_tensor([question["answer_index"]], device="cuda"))
                loss.backward()
                gradients = [parameter.grad for parameter in controller.parameters() if parameter.grad is not None]
                gradient_l1 = sum(float(gradient.abs().sum()) for gradient in gradients)
                if not gradients or not all(torch.isfinite(gradient).all() for gradient in gradients) or gradient_l1 == 0:
                    raise RuntimeError("Frozen-tail controller gradients failed")
                report["checks"].append({"condition": condition, "method": method, "whole_vocabulary_zero_exact": True,
                                         "full_model_vs_tail_max_option_error": float(np.abs(online[option_ids] - replay.detach().float().cpu().numpy()[0]).max()),
                                         "full_model_vs_tail_prediction_equal": int(online[option_ids].argmax()) == int(replay.argmax()),
                                         "gradient_l1": gradient_l1, "intervention": intervention,
                                         "reference_context_relative_l2_error": metadata["reference_context_relative_l2_error"]})
                del controller, gradients, delta, replay, loss
            if not np.array_equal(native, forward_logits(backend, batch)):
                raise RuntimeError("Restoration smoke failed")
            del batch, tensors, arrays
        report["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
        report["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        del backend, tail
        torch.cuda.empty_cache()
    report["status"] = "complete"
    write_json(output / "report.json", report)
    print("Controller smoke complete: twelve zero identities, twelve nonzero full-forward/replay gradient checks.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

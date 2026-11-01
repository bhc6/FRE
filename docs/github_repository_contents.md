# GitHub repository contents

This repository contains the research code, protocols, literature audits, mathematical derivations, and a snapshot of available analysis artifacts. The initial GitHub upload uses a private repository.

`spectral_memory/` contains executable research modules; `docs/` contains protocols and interpretation, including negative results. Experiment status and scientific claims remain as recorded in each original report. A report or preview is not evidence that all later planned experiments have completed.

Selected artifacts retain their original paths under `outputs/`, so report and figure links can be read on GitHub. Included artifacts are Markdown reports, LaTeX tables, PNG/SVG figures, CSV tables, and selected JSON summaries/checks, each no larger than 10 MiB. Source snapshots embedded in experiment outputs are excluded. The exact uploaded artifact list, byte counts, and SHA256 digests are recorded in `github_upload_manifest.json`.

Git attributes preserve file bytes without line-ending normalization so the published artifacts retain their recorded SHA256 digests.

The repository does not contain the downloaded datasets, source videos, frame collections, model weights, virtual environment, caches, complete activation archives, or the full raw prediction/logit archives. Those remain local and must be obtained or regenerated using the relevant source documentation and experiment protocols. The 17 MB full passive-probe analysis JSON is also omitted from this initial upload; its Markdown report remains available. Consequently, this upload is a code and report archive, not a complete raw-data replication bundle.

Local runs may create further outputs after this snapshot. Because `outputs/` is ignored by default, updating the published reports requires explicitly selecting the intended artifacts again.

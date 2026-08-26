---
search:
  exclude: true
robots: noindex, follow
---

# Data and model assets

This page preserves the repository's original `docs/ASSETS.md` link. See the
expanded [Data & model assets guide](reference/assets.md) for the current asset
map, integrity checklist, environment overrides, and license boundaries.

## Key boundary

- Participant recordings are **not distributed** in this GitHub repository.
- Authorized recordings remain local under ignored `data/` directories.
- Python checkpoints, normalization files, and Android ONNX weights are public
  in the [Hugging Face Model repository](https://huggingface.co/config-h/Wearable-IMU-Activity-Segmentation-Pipeline).
- GitHub tracks their manifest and small configuration files, not the model
  binaries. Missing assets are downloaded and verified before use.
- Additional locally generated checkpoints, results, figures, and logs remain
  ignored unless intentionally curated.

Dataset access and the canonical local layout are documented on the
[Data page](guide/data.md).

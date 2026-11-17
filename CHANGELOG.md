# Changelog

All notable changes to this repository are documented in this file.

## v0.0.0.1 - 2026-06-15

### Added
- English and Japanese release notes for the first GitHub Release.
- Scientific experiment journey documents describing how the 0.7527 hold-out Dice recipe was derived.
- Release links from the top-level English and Japanese READMEs.
- Per-case evaluation evidence under `artifacts/eval_runs/cross_arch_v0.0.0.1/`:
  CSV with per-case Dice for all 25 cases (broken down by stage),
  `metrics_summary.json` reproducing the headline 0.7527 mean Dice +0.0140 adaptive-threshold gain,
  and a README explaining how to verify the numbers from cached probability maps.
- `MODEL_CARD.md` describing the 13 trained checkpoints (8 ConvNeXt + 3 2D nnU-Net + 2 3D nnU-Net),
  intended use, out-of-scope use, limitations, and the per-case failure modes called out in the journey doc.
- `docs/weights.md` with SHA-256 checksums and inventory for three weight tarballs
  (≈ 957 MB + 331 MB + 269 MB) prepared locally for future GitHub Release attachment.

### Changed
- Aligned repository metadata and public manifest versioning to v0.0.0.1.
- Aligned project title to `isles2022-2d3d-blend-reproducible-pipeline` to match
  the `yskfuji/<dataset>-<method>-reproducible-pipeline` convention used by the
  sibling portfolio repositories.
- Marked the initial GitHub Release notes milestone complete in the roadmap.

## v0.0.0.0 - 2026-06-15

### Added
- Initial public bundle for the ISLES 2022 2D+3D heterogeneous-ensemble pipeline.
- Source code, configs, smoke test, audit map, citation metadata, and bilingual entry READMEs.
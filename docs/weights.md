# Trained weights — distribution plan and integrity checksums

This document tracks the trained weights that produced the v0.0.0.1 headline
result of **mean Dice 0.7527** on the 25-case custom ISLES 2022 hold-out test
split. For the architectural / training description of each model, see
[`MODEL_CARD.md`](../MODEL_CARD.md).

## Status

> ⚠️ **As of v0.0.0.1, weights are not bundled in the GitHub Release.**
> They have been packaged and checksummed locally; the planned distribution
> path is GitHub Release attachments on a follow-up release. The hashes below
> let any downstream consumer verify the weights once they are published.

## Tarball inventory

The full ensemble (13 checkpoints) is packaged into three tarballs so each
sub-component can be downloaded independently.

### `isles2022_2d3d_blend_v0.0.0.1_convnext_8model.tar.gz` (≈ 957 MB)

8 ConvNeXt-Tiny checkpoints — the 2.5D branch.

```
convnext_8model/
  v2_1mm/         { best.pt, config.yaml }
  v2_1mm_s0/      { best.pt, config.yaml }
  v2_aug_1mm/     { best.pt, config.yaml }
  v2_aug_1mm_s0/  { best.pt, config.yaml }
  v3_aug_1mm/     { best.pt, config.yaml }
  v3_aug_1mm_s0/  { best.pt, config.yaml }
  v3_dilated_1mm/    { best.pt, config.yaml }
  v3_dilated_1mm_s0/ { best.pt, config.yaml }
```

### `isles2022_2d3d_blend_v0.0.0.1_nnunet_2d_3fold.tar.gz` (≈ 331 MB)

3 nnU-Net v2 2D fullres fold checkpoints — the 2D nnU-Net branch.

```
nnunet_2d_3fold/
  fold_0/  { checkpoint_best.pth, checkpoint_final.pth }
  fold_1/  { checkpoint_best.pth, checkpoint_final.pth }
  fold_2/  { checkpoint_best.pth, checkpoint_final.pth }
  dataset.json
  plans.json
```

### `isles2022_2d3d_blend_v0.0.0.1_nnunet_3d_2fold_mps.tar.gz` (≈ 269 MB)

2 MPS-compatible nnU-Net v2 3D fullres fold checkpoints — the 3D nnU-Net branch.

```
nnunet_3d_2fold_mps/
  fold_0/  { checkpoint_best.pth }
  fold_1/  { checkpoint_best.pth }
  dataset.json
  plans.json
```

## SHA-256 checksums

```text
7bfc98c73bc626dfcc17c78a24205eb9bfd67dc32bf0c5e1a80e64b78763995e  isles2022_2d3d_blend_v0.0.0.1_convnext_8model.tar.gz
e08c9597d6622c13f5c5b03383c545b0d5ac5fdb15b6ff207a4c76d1392eb8b1  isles2022_2d3d_blend_v0.0.0.1_nnunet_2d_3fold.tar.gz
5edb9e686c4d944cb016f8ebdbc7123ad8e4c5712dbe73f0f5a9014ec07ea0ee  isles2022_2d3d_blend_v0.0.0.1_nnunet_3d_2fold_mps.tar.gz
```

Verify after downloading:

```bash
shasum -a 256 -c <<'EOF'
7bfc98c73bc626dfcc17c78a24205eb9bfd67dc32bf0c5e1a80e64b78763995e  isles2022_2d3d_blend_v0.0.0.1_convnext_8model.tar.gz
e08c9597d6622c13f5c5b03383c545b0d5ac5fdb15b6ff207a4c76d1392eb8b1  isles2022_2d3d_blend_v0.0.0.1_nnunet_2d_3fold.tar.gz
5edb9e686c4d944cb016f8ebdbc7123ad8e4c5712dbe73f0f5a9014ec07ea0ee  isles2022_2d3d_blend_v0.0.0.1_nnunet_3d_2fold_mps.tar.gz
EOF
```

## Planned distribution

When weights are published, they will be attached to the GitHub Release for
the same tag (`v0.0.0.1`). Download instructions will be:

```bash
TAG=v0.0.0.1
BASE=https://github.com/yskfuji/isles2022-2d3d-blend-reproducible-pipeline/releases/download/${TAG}

curl -L -o convnext_8model.tar.gz       ${BASE}/isles2022_2d3d_blend_${TAG}_convnext_8model.tar.gz
curl -L -o nnunet_2d_3fold.tar.gz       ${BASE}/isles2022_2d3d_blend_${TAG}_nnunet_2d_3fold.tar.gz
curl -L -o nnunet_3d_2fold_mps.tar.gz   ${BASE}/isles2022_2d3d_blend_${TAG}_nnunet_3d_2fold_mps.tar.gz

# Verify (see SHA-256 checksums above)
# Extract
for f in *.tar.gz; do tar -xzf $f; done
```

## DUA note

The trained weights are derivative of ISLES 2022 data. Users downloading these
weights are expected to comply with the ISLES 2022 Data Use Agreement and to
treat the weights as having the same data-use constraints as the source data.
This NOTICE does not modify the Apache 2.0 license that covers the repository
source code.

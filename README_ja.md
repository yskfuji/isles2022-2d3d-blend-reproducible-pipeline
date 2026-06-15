# isles2022-2d3d-blend-pipeline

**言語:** [English](README.md) | 日本語

ISLES 2022 急性期脳梗塞病変セグメンテーションのための異種アーキ・アンサンブルパイプライン。2D nnU-Net (3-fold) + MPS 互換 3D nnU-Net (2-fold) + 2.5D ConvNeXt (8-model) を DWI native 空間で確率融合し、過小予測された大病変を救済するケース適応的閾値処理を組み合わせる。

**クイックリンク**
- エントリーガイド (EN): [isles2022_2d3d_blend/README_en.md](isles2022_2d3d_blend/README_en.md)
- エントリーガイド (JA): [isles2022_2d3d_blend/README.md](isles2022_2d3d_blend/README.md)
- **実験ジャーニー (0.7527 への科学的経緯)**: [EN](docs/experiment_journey.md) | [JA](docs/experiment_journey_ja.md)
- リリースノート: [EN](docs/releases/v0.0.0.1.md) | [JA](docs/releases/v0.0.0.1_ja.md)
- 監査マップ: [AUDIT_MAP.md](AUDIT_MAP.md)
- ロードマップ: [ROADMAP.md](ROADMAP.md)
- 引用情報: [CITATION.cff](CITATION.cff)

## このリポジトリで提供されるもの

- 2D、3D、2.5D セグメンテーションアーキを組み合わせた再現可能な異種アンサンブルレシピ
- MPS 互換 nnU-Net 3D 訓練（Apple Silicon の `ConvTranspose3d` 未対応問題の回避策）
- CC-overlap rescue 付きの cross-architecture 確率融合
- **ケース適応的閾値**: 基底閾値での予測ボリュームが大きいときに低閾値に切替え、過小セグメンテーションされた大病変を救済（小病変には影響なし）
- データ不要の smoke test（1 分以内に公開バンドルを検証）

## 主要結果

ISLES 2022 の 25 ケース独自 hold-out test split:

| 構成 | mean Dice |
|---|---:|
| ベースライン (8× ConvNeXt 2.5D + 2D nnU-Net 3-fold cross-arch) | 0.7352 |
| + 3D nnU-Net 2-fold blend (2D 60% / 3D 40%) | 0.7387 |
| **+ 適応閾値 post-processing** | **0.7527** |

適応閾値は、基底閾値での予測ボリュームが 4000 voxel を超えるケースだけで低閾値に切り替わるため、25 ケース中 2 ケース（大病変が過小予測されていたケース）にのみ作用し、残り 23 ケースは不変。+0.014 mean Dice の純増。

## 最終レシピ

```
nnUNet 確率 = 0.6 × 2D-3fold 平均 + 0.4 × 3D-2fold 平均 (fold 0,1)
combined    = 0.20 × ConvNeXt-8 平均 + 0.80 × nnUNet 確率
rescue      = alpha=0.7, case_gate_agree=0.4, cc_min_overlap=20
post-proc   = base_thr=0.30 → 予測ボリューム > 4000 のとき 0.03 で再閾値化
```

## MPS 互換 3D nnU-Net

`ConvTranspose3d` は Apple MPS で未実装。カスタムトレーナー `nnUNetTrainer_MPS3D_500epochs` は `get_matching_convtransp` を monkey-patch し、3D アップコンボリューションを「nearest-neighbor upsample（`view`+`expand`+`reshape` の MPS native オペレーション）+ 3×3×3 Conv3d」で置換する。reshape ベースの upsample は整数 scale factor で nearest-neighbor と数学的に同一、かつ `F.interpolate(mode='nearest')`（暗黙的に CPU フォールバック）の ~2 倍高速。

詳細は [`core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py`](core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py) を参照。

## 対象読者

- 医療 AI セグメンテーションポートフォリオを評価する採用担当者
- 監査可能な MRI セグメンテーションアンサンブルの基盤を必要とする ML エンジニア
- MPS の `ConvTranspose3d` 未対応で詰まっている Apple Silicon 研究者

## クイックスタート

### 1. 医療データなしでリポジトリ検証

```bash
python scripts/smoke_test.py --use_dummy_data
```

### 2. 公開バンドル manifest 確認

```bash
cd core/pipeline
python tools/make_manifest.py
```

### 3. 自分のデータで訓練 / 評価実行

- 完全ガイド (EN): [isles2022_2d3d_blend/README_en.md](isles2022_2d3d_blend/README_en.md)
- 完全ガイド (JA): [isles2022_2d3d_blend/README.md](isles2022_2d3d_blend/README.md)

## 同梱 / 除外

同梱:
- ソースコード（モデル、訓練、評価、cross-arch 融合）
- MPS 用 nnU-Net trainer variant
- 設定ファイル
- 監査・再現性ドキュメント

除外:
- `Datasets/`
- 訓練済み重み (`*.pt`, `*.pth`)
- `runs/`, `results/`, `logs/`

## 引用方法

[CITATION.cff](CITATION.cff) を参照。

## ライセンス

Apache License 2.0。[LICENSE](LICENSE) と [NOTICE](NOTICE) を参照。

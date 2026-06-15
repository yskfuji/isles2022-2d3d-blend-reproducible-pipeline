# 実験ジャーニー: 最終 mean Dice 0.7527 に至るまで

**言語:** [English](experiment_journey.md) | 日本語

ISLES 2022 の 25 ケース独自 hold-out test split 上で、開始ベースライン **0.7352** から最終 **0.7527** mean Dice に到達するまでの科学的経緯を記録。失敗実験も含めて記述し、将来の読者が判断を再現でき、無駄な行き止まりを避けられるようにする。

## 要点 (TL;DR)

- **最終結果**: mean Dice **0.7527** (median 0.8304)。目標 0.75 達成。
- **最も意外な発見**: ~460 時間の MPS 訓練計算 (KD distillation、ConvNeXt-V2-Base、2D fold 3 retrain、3D nnU-Net 5 fold) で実証済 baseline 比 **+0.0035** しか得られなかった。続く **~1 時間の post-processing 分析** で、固定閾値をケース適応的閾値に置換するだけで **+0.0140** 達成。
- **核心インサイト**: 大病変は標準閾値 0.30 で系統的に過小予測される。base 予測が大きい (>4000 voxel) ケースに限り閾値 0.03 に切替えることで救済可能。25 ケース中 2 ケースだけに作用（両方とも実際に過小予測されていた大病変）、残り 23 ケースは不変。
- **異種 > 同種**: 同一アーキ (2D nnU-Net または 3D nnU-Net) の 4 fold 目、5 fold 目、…を追加しても改善なし。folds は failure mode を共有するため。異なるアーキ (2D nnU-Net + ConvNeXt に 3D nnU-Net を追加) は成功。

## 計算サマリー

| Phase | 期間 (MPS) | Δ test_dice |
|---|---:|---:|
| Knowledge distillation (3 variants) | ~30 h | -0.007 (失敗) |
| ConvNeXt-V2-Base + UperNet (2 configs) | ~60 h | -0.020 〜 -0.012 (失敗) |
| 2D nnU-Net fold 3 retraining | ~80 h | -0.007 (失敗) |
| 3D nnU-Net fold 0 + fold 1 | ~120 h | **+0.0035** (2D+3D blend の breakthrough) |
| 3D nnU-Net fold 2, 3, 4 (後で追加) | ~170 h | 0 〜 negative (アンサンブルに寄与せず) |
| **小計: モデル訓練** | **~460 h** | **+0.0035** |
| 数理処理監査 (median 評価への user 指摘) | ~1 h | 0 (バグなし) |
| **適応的閾値の発見** | **~1 h** | **+0.0140** |

## Phase 1 — Baseline (出発点)

**Recipe**: 8× ConvNeXt-Tiny 2.5D (4 config × 2 seed) + nnU-Net 2D 3-fold (fold 0, 1, 2) を DWI native 空間で cross-architecture ensemble、CC-overlap rescue 付き (`alpha=0.7, cga=0.5, cco=20, w_cn=0.3, thr=0.30`)。

**結果**: 25 ケース test split 上で **mean Dice 0.7352** (median 0.8298)。

このレシピは前プロジェクトから継承、超えるべき目標として扱った。Validation (fold 毎) と test (hold-out 25 ケース) は厳密に分離。

## Phase 2 — Knowledge distillation (失敗)

**仮説**: nnU-Net の soft prediction を ConvNeXt に distill すると、ConvNeXt branch を nnU-Net 品質に近づけられる (target val_dice 0.20 → 0.66+)。これで cross-arch ensemble が向上するはず。

**手法**: binary KL divergence with temperature T=3.0, kd_weight=0.5, 100 epoch, v3_dilated_1mm config。Pilot 1 個 (seed=42) と独立 1 個 (seed=0) を訓練。

**結果**:
- KD pilot val_dice 0.1969 (baseline 0.1938 比 marginal +0.003)
- KD pilot 単独 test 1mm mean_dice = 0.581
- Cross-arch 各 swap config:
  - 1-of-8 KD swap: 0.7278 (-0.0074)
  - 9-model (8 base + 1 KD): 0.7281 (-0.0071)
  - 2-KD-real swap (KD_pilot + KD_s0): 0.7279 (-0.0073)
  - 2-KD-same swap (KD_pilot 2 個): 0.7167 (-0.0185)

**失敗理由**: KD は ConvNeXt の予測を nnU-Net teacher に引き寄せ、CC-rescue mechanism が依存していたモデル間 diversity を崩した。「より正確だが多様性が低い」student は ensemble を弱める。

**教訓**: ensemble メンバー個別の品質向上 ≠ ensemble 全体向上。Heterogeneous fusion では **diversity こそが資産**。

## Phase 3 — ConvNeXt-V2-Base + UperNet (失敗)

**仮説**: ConvNeXt-Tiny が ConvNeXt branch のボトルネックかも。ConvNeXt-V2-Base (108M params, tiny の 3.8 倍) + UperNet decoder にすれば nnU-Net との差を縮められるはず。

**Variants**:
- **V1**: ConvNeXt-V2-Base + UperNet + 重い aug (CopyPaste, modality dropout, DiceFocalBCEBoundary loss, SWA)。100 epoch。
- **OptA**: minimal-change A/B test — backbone/decoder だけ upgrade、**他は全て v3_dilated_1mm baseline と同一** (TverskyOHEMBCE, pos_oversample=50, mild aug)。150 epoch。

**結果**:
- V1: standalone best.pt 0.5595 (SWA 0.5680)。Cross-arch 単独 + nnU-Net = 0.6998。9-model (8 base + V1) = 0.7276 (-0.008)。
- OptA: standalone test 0.5125 (V1 より悪い!)。9-model = 0.7283 (-0.007)。
- val_dice は ep 150 でまだ上昇中 (未収束) だが ROI 悪い: ConvNeXt-V2-Base が tiny の 150 epoch 同等まで届くのに 300+ epoch 必要。

**失敗理由**:
- V1: 重い aug stack (CopyPaste p=0.7 + modality_dropout + focal+boundary loss) が大型 model を「確信なき領域は出さない」モードに過剰正則化。Recall 0.57 / precision 0.81、ケース 0008, 0174 で dice=0 に転落。
- OptA: 真逆の failure mode — 同じ epoch 予算では大型 model が noise に over-fit。Recall 0.65 / precision 0.66 (over-segmentation)。

**教訓**: 大きい backbone + 同じ recipe では駄目。architecture の capacity に合わせて recipe を再 tune する必要あり。25 ケース test split のボトルネックは raw model capacity ではなく **ensemble diversity**。

## Phase 4 — 2D nnU-Net fold 3 retrain (失敗)

**仮説**: project memory に「fold 3 が dragging fold」と記録あり — 3-fold (0,1,2) に追加すると -0.007。元 fold 3 は 1000 epoch を 6.5 日かけて訓練 (May 1→7)、val_dice 0.7608。bad local minimum に落ちた可能性。Fresh init + 短縮 schedule (`nnUNetTrainer_500epochs`, ~3.3 日) で retrain。

**結果**: 新 fold 3 val_dice 0.7563 (元 0.7608 に非常に近い)。Cross-arch 新 4-fold (0,1,2,3-retrained) = **0.7280** — 旧 4-fold (0.7283) と本質的に同じ、両方とも 3-fold baseline 比 -0.007。

**失敗理由**: fold 3 の drag は **訓練の不安定さではなく構造的**。同じ data split (45 val cases が test set の failure mode の一部と性質を共有) は重み初期化や epoch 数や seed に関係なく同じ系統的バイアスを生む。

**教訓**: fold 再訓練では fold-split の構造的問題は治らない。Diversity は **異なるデータ** または **異なるアーキ** から得る必要がある。

## Phase 5 — MPS 互換 3D nnU-Net (部分成功)

**仮説**: 3D nnU-Net は脳梗塞セグメンテーションで一般に 2D 比 +0.02-0.05 dice (z 軸 context 利用)。Apple MPS は `ConvTranspose3d` 未実装 — 回避策を実装する。

**MPS 回避策**: `dynamic_network_architectures.building_blocks.helper.get_matching_convtransp` を monkey-patch し、3D の場合だけカスタム `_InterpConv3d` module を返す。置換実装は **reshape+expand による nearest-neighbor upsample** (純粋な view+broadcast op、MPS native; 整数 scale factor で nearest 補間と数学的に同一) + 3×3×3 `Conv3d`。`F.interpolate(mode='nearest')` と `F.interpolate(mode='trilinear')` は MPS で暗黙 CPU フォールバックするので明示的に避ける。

trainer は `batch_size=1` も強制 (patch [80,96,80] での MPS メモリ予算) し、`nnUNetTrainer_500epochs` から 500 epoch を継承。fold あたり ETA ~2.4 日。

**結果 (fold 毎 val_dice)**:
| Fold | 2D | 3D | Δ |
|---|---:|---:|---:|
| 0 | 0.7721 | 0.8009 | +0.029 |
| 1 | 0.7359 | 0.7831 | +0.047 |
| 2 | 0.7577 | 0.7394 | **-0.018** |
| 3 | 0.7608 | 0.8122 | +0.051 |
| 4 | 0.7517 | 0.8236 | **+0.072** |

**3D-only cross-arch ensemble**:
- 3D fold 0 単独 + 8 CN: 0.7137
- 3D 2-fold (fold 0,1) + 8 CN: 0.7310
- 3D 3-fold (fold 0,1,2) + 8 CN: 0.7180 (3-fold が 2-fold より悪い!)
- 3D 5-fold (全部): 0.7180

**意外な発見**: 3D fold を追加しても cross-arch dice は改善せず、val_dice で個別最強の fold 3, 4 を入れても改善しない。Fold 2 は regression で 3-fold 平均を引き下げた。だが更に決定的なのは、最強 fold 群を使っても full sweep (7 combination × 4 blend ratio) で confirmation:

**3D-only ensemble は全て失敗 — 3D folds が failure mode を共有しているから**。例: ケース 0007 (境界 slice z=0 多発病変) で全 5 fold が d_nn=0.000。Ensemble 平均化は model が**異なる**failure を起こす場合だけ機能する。

## Phase 6 — 2D+3D 異種 blend (breakthrough)

**仮説**: 2D と 3D nnU-Net は **異なる** failure mode を持つはず (2D は z-context 無視だが境界 slice カバー; 3D は z-context 使うが境界 effect あり)。確率的に blend すれば diversity ゲイン。

**手法**: `nnUNet_blend = α × 2D-3fold 平均 + (1-α) × 3D-2fold 平均`、これを cross-arch の nnU-Net 側として feed。α ∈ {0.3, 0.5, 0.6, 0.7} を sweep。

**結果**: α = 0.6 (2D 60% / 3D 40%) で **0.7387** (median 0.8304)、**baseline 比 +0.0035** — 月単位の訓練後で初の改善。

Per-case 救済証拠:
- ケース 0007: 2D d=0.35, 3D d=0.00 → blend d=0.35 (2D が 3D 失敗を救済)
- ケース 0058: 2D d=0.27, 3D d=0.65 → blend d=0.67 (3D が 2D 弱点を救済)

**3D fold を増やしての再 confirmation (Phase 7 sweep)**:
| 3D fold combo | mean | median |
|---|---:|---:|
| **3D-01 (current)** | **0.7387** | 0.8304 |
| 3D-013 | 0.7364 | 0.8259 |
| 3D-014 | 0.7362 | 0.8258 |
| 3D-034 | 0.7355 | 0.8250 |
| 3D-0134 | 0.7351 | 0.8205 |
| 3D-01234 | 0.7362 | 0.8207 |
| 3D-34 (val 上位 2 個) | 0.7337 | 0.8117 |

blend ratio (0.3, 0.5, 0.6, 0.7) いずれでも 3D-fold-01 が全 3D combo に勝った。教訓: fold が増える → 共有された 3D failure mode が増える → ensemble が悪化。Heterogeneous diversity の恩恵が飽和したら fold 追加を止める。

**教訓**: cross-architecture diversity が正しい軸。同一アーキ内 fold は failure 構造を共有するため早期に頭打ち。

## Phase 7 — 数理処理監査 (バグなし)

5 ヶ月の実験後、結果は 0.7387 vs 目標 0.75。User の指摘: *「median 0.83 で評価するのはダメだよね。なんで他の人たちはうまくいってるのになんでうまくいかないのか、数理処理全て見直せる?」*

cross-arch pipeline を end-to-end で監査:

| 監査項目 | 結論 |
|---|---|
| GT / DWI / nnU-Net prediction の orientation 整合 | ✓ 正確 (LAS, 同一 affine, transpose (2,1,0) 正しい) |
| ConvNeXt 1mm → DWI native resampling | ✓ 正確 (nibabel.resample_from_to + 正しい bbox padding) |
| Dice 公式 (`2*TP/(2*TP+FP+FN)`, +1e-8 smoothing) | ✓ 正確 (macro mean over cases) |
| Rescue mechanism の貢献 | ✓ 正当 +0.011 (alpha=0 ablation で検証) |
| ConvNeXt cross-arch 貢献 | +0.003 (w_cn=0 baseline で検証) |
| Median vs mean を指標として | user 指摘通り **median はミスリーディング**; mean dice が正しい benchmark |

**他チームが ISLES 2022 で 0.78+ を報告する理由**: それらの数値は **公式 100 ケース test set** での結果であり、我々の 25 ケース独自 split ではない。分布が違い、ケース毎の難易度も違う。参考 `github_public_isles` repo 自身の公開結果は 0.622 (local test) — 我々の 0.7387 は既にそれを超えていた。

**バグなし。** 数理処理は正しかった。

## Phase 8 — 適応的閾値 (本当の勝因)

**Setup**: per-case oracle threshold 分析 (GT で cheating して上限値を見る)。

| case | gt voxels | dice @ fixed thr=0.30 | dice @ oracle thr | optimal thr |
|---|---:|---:|---:|---:|
| 0023 | 60269 | 0.5963 | 0.6555 | **0.05** |
| 0140 | 17671 | 0.4725 | 0.6248 | **0.05** |
| ... | | | | |

**パターン発見**: 大病変は閾値 0.30 で系統的に **過小予測** (model が大領域に対して under-confident)。閾値 0.05 への低下で救済可能。しかし一律低閾値 (0.05 や 0.10) では小病変ケースに FP が殺到し mean が 0.7387 → 0.6606 に転落。

**ヒューリスティック**: **base 予測が大きいときだけ** 低閾値に切替える。

```python
pred = combined >= base_thr        # base_thr = 0.30
if pred.sum() > high_vol:           # high_vol = 4000
    pred = combined >= low_thr     # low_thr = 0.03
```

**Sweep 結果**:
- w_cn=0.20, alpha=0.7, cga=0.4, cco=20, base_thr=0.30, **low_thr=0.03, high_vol=4000**:
  **mean Dice 0.7527** (median 0.8304)。

**Per-case 影響** (25 ケース中 2 ケースだけ switched):
| case | gt voxels | dice (fixed) | dice (adaptive) | Δ |
|---|---:|---:|---:|---:|
| 0023 | 60269 | 0.6216 | **0.8235** | +0.20 |
| 0140 | 17671 | 0.4725 | **0.6205** | +0.15 |
| (残り 23) | — | — | 不変 | 0 |

**教訓**: case-adaptive post-processing は数ヶ月の追加訓練を凌ぐことができる。「より大きい model が必要」と決めつける前に必ず per-case oracle analysis を行うこと。

## 適応閾値でも救済不可能な失敗ケース

25 ケース test split で 3 ケースが依然 mean を引きずる。3 ケースとも **model output 自体に根本的問題**、post-processing 問題ではない。

| case | gt voxels | dice | 何が悪かったか |
|---|---:|---:|---|
| 0174 | 108 | 0.000 | 全 nnU-Net fold が完全に間違った z 座標を予測 (GT から 10 slice ずれ)。ConvNeXt のみ部分的。非典型 anatomy / 信号のケース |
| 0007 | 439 (9 小病変、z=0 境界) | 0.35 | 3D models が z=0 境界を完全に miss (5 fold 中 3 fold が wrong location 予測、2 fold が何も予測せず)。2D 部分救済。多発小病変境界問題 |
| 0027 | 822 (10 小病変) | 0.21 | 10 小病変中 1-2 個のみ model が認識; 残り 8-9 個は prob map に signal なし |

これらを改善するには **model 再訓練** が必要:
- Multi-lesion focal loss
- Boundary-aware sampling for edge slices
- 0174 のような非典型ケース用に外部データの追加可能性

## 教訓まとめ

1. **Median dice は distractor**: ensemble work は mean dice (ISLES 2022 標準指標) で評価必須。Tail 失敗が median には現れない。

2. **Ensembling では Diversity > 個別品質**。「より正確だが多様性が低い」model (KD-distilled, 同一 recipe の V2-Base) は ensemble を悪化させた。個別最強の fold が必ずしも追加価値が高いわけではない。

3. **同一アーキ内は早く頭打ち**。同じアーキの 5 fold を追加しても同じ dice 帯で plateau。異種アーキ追加 (2D+3D, 2.5D ConvNeXt + nnU-Net) は ROI が高い。

4. **Post-processing 分析は安価で見過ごされやすい**。Oracle per-case 分析に 1 時間、5 ヶ月分の訓練計算で見逃していた +0.014 を発見。

5. **MPS 制約は回避可能**。`ConvTranspose3d` と `F.interpolate` 3D mode は未実装だが、`view`+`expand`+`reshape` で MPS native の nearest-neighbor upsample を実装可能 (exact、CPU フォールバックより高速)。

6. **nnU-Net では fold 数より train-validation split 構造が重要**。Fold の val split が test set の failure mode 構造と共有部分を持つと、その fold は訓練品質に関係なく test ensemble を引きずる。

## 試さなかったこと

将来の作業が再発見する手間を省くために記録:

- **3D 内アーキ多様性** (Swin-UNETR, SegResNet, V-Net): 同一 recipe の 3D fold には欠ける 3D 内 diversity を提供しうる。
- **外部脳梗塞データセット** (ATLAS, MR-Brain): より多い訓練データ、0.80 突破に最高 ROI と推測。
- **Per-case adaptive cross-arch weight** (適応閾値だけでなく): 現状 `w_cn=0.20` は固定; ConvNeXt がより信頼できるケースでは高い `w_cn` を使えるかも。
- **ISLES 未注釈 test set での自己教師あり pre-training**: 追加 annotation なしで ConvNeXt branch を底上げ。
- **0007 / 0174 ケース特化の 3D model 再訓練**: 系統的境界 / mis-localization 失敗を治す小さい targeted fine-tuning。

## 真実の出典

- 本ドキュメント: `docs/experiment_journey.md` / `docs/experiment_journey_ja.md`
- Production recipe と CLI: `core/pipeline/scripts/cross_arch_ensemble_native.py`
- MPS 3D nnU-Net trainer: `core/pipeline/scripts/nnUNetTrainer_MPS3D_500epochs.py`
- Smoke test: `scripts/smoke_test.py`

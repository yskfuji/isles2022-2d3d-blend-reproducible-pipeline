# Public repo polish checklist — isles2022-2d3d-blend-reproducible-pipeline

テンプレ由来: RSNA `docs/audit/PUBLIC_REPO_POLISH_CHECKLIST.md`  
監査日: 2026-11-11

## 1. 証拠と主張

| 項目 | 状態 | メモ |
|---|---|---|
| 数値の一次成果 | 部分 | `artifacts/eval_runs/cross_arch_v0.0.0.1/` を一次根拠として README が参照 |
| 制約と数値の同一節 | 完了 | Not a medical device + Dice 表 |
| claims 検査スクリプト | 該当なし | 未移植 |

## 2. 公開衛生

| 項目 | 状態 | メモ |
|---|---|---|
| 機微パス検査 | 部分 | 重み・Datasets 除外。専用スクリプトなし |
| 相対リンク | 完了 | |
| 無視規則 | 完了 | |

## 3. 図

| 項目 | 状態 | メモ |
|---|---|---|
| PNG + `width="100%"` | 完了 | architecture / results_snapshot（新規生成） |
| `*_ja.png` | 完了 | |
| Mermaid 非依存 | 完了 | |
| raw ハッシュ照合 | 完了 | 下記参照 |

## 4. 日本語

| 項目 | 状態 | メモ |
|---|---|---|
| である調 | 完了 | README_ja 書き換え |
| textlint 設定 | 完了 | `.textlintrc.json` |
| 口語時間売り禁止 | 完了 | |

## 5. 完了定義

| 項目 | 状態 | メモ |
|---|---|---|
| ローカル図再生成 | 完了 | 空だった assets を埋め |
| raw 照合 | 完了 | 下記参照 |
| ブラウザ確認 | 未 | |

## 監査時ギャップ（整備前）

- docs/assets が空（図なし）
- README_ja がです・ます／未整備
- PNG 埋め込みなし

## 適用コミット

`38720d4` — PNG/JA/caveat/checklist

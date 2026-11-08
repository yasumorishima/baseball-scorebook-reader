# cv/ — 生成 AI を使わずにスコアブックを読む部品

OpenCV と NumPy だけで、手書きのスコアブックの写真からマスの中の印を読む部品です。Raspberry Pi 5 の上で開発しました。同じ写真はいつ読んでも同じ結果になります。上の `scorebook/`（Claude で読むツール）とは別の、研究開発の経路です。

- **対象の用紙**：成美堂の 9104 番のスコアブックです。ほかの用紙は格子の位置が違うので、`grid_detect.py` の定数を合わせる必要があります。
- **含まないもの**：正解データ（選手名を含む）と画像です。多くの部品は、正解データと、そこから作る見本（テンプレート）があることを前提にしているので、このままでは動きません。読み方の参考や、自分の用紙に合わせて作り直すときの土台として置いています。
- 開発の経緯と測った成績は [baseball-scorebook-ocr-docs](https://github.com/yasumorishima/baseball-scorebook-ocr-docs) にあります。記号の読み方は同じ repo の [notation.md](https://github.com/yasumorishima/baseball-scorebook-ocr-docs/blob/main/notation.md) にまとめています。

## 流れ

1. **格子を見つける**（`grid_detect.py`）：写真の傾きを直し、打順の行と回の列の線を見つける。
2. **手書きだけを取り出す**（`cell_diff.py`）：印刷された白紙の用紙との差をとって、鉛筆のインクだけを残す。
3. **列を回に分ける**（`split_innings.py`、`header_scan.py`）：どの列が何回かを、アウトの数字と打順のつながりから決める。手で振り直された回の番号も色で見分ける。
4. **1 打席ずつの事実を読む**：打席があるか（`pa_detect.py`）、何番目のアウトか（`out_detect.py`）、どの塁まで進んだか（`edge_score.py`）。
5. **塁の区画の印を読む**（`annot_zones.py`、`corner_knn.py`、`token_detect.py`）：区画ごとにインクの塊を切り出し、番号つきの印（何番打者のときに進んだか）と番号のない印（`WP`・`S` など）を見本との近さで読む。
6. **野球の決まりと一緒に解く**（`solve_corners.py`）：「進塁の番号は、その走者より後に打った打者の番号しかありえない」「1 つの塁に走者は 1 人で、追い越さない」「四球で進むのは押し出される走者だけ」などの決まりに合う読み方の中から、見本にいちばん近いものを選ぶ。

## 部品

| ファイル | すること |
|---|---|
| `grid_detect.py` | 格子の検出 |
| `grid_batch.py` | フォルダの写真に格子の検出をまとめてかける |
| `cell_diff.py` | 白紙との差で手書きのインクを取り出す |
| `split_innings.py` | 列を回に分ける（どの列が何回かを読まずに切る） |
| `header_scan.py` | 手で振り直された回の番号を見つける |
| `pa_detect.py` | 打席があるか |
| `out_detect.py` | 何番目のアウトか |
| `edge_score.py` | どの塁まで進んだか |
| `annot_zones.py` | インクの塊がマスのどの区画にあるか |
| `corner_knn.py` | 区画の番号の印・真ん中の印を読む |
| `token_detect.py` | 番号のない印（`WP`・`S` など）を見分ける |
| `solve_corners.py` | 読みと野球の決まりを合わせて解く |
| `count_strip.py` | 球のカウントの帯の印を読む |
| `tally_read.py`、`stats_crop.py` | 右側の個人成績の欄を読む・切り出す |
| `gt_score.py` | 正解データと比べて採点する |
| `cell_zoom.py`、`montage.py` | 人や Claude が目で確かめるための拡大画像を作る |

## 動かすのに要るもの

```
pip install numpy opencv-python-headless
```

正解データと見本の置き場所（`cv/ground-truth/`、`eval/`）はこの repo にありません。自分の用紙で使うときは、数枚を書き起こして同じ形の正解データを作るところから始めます。

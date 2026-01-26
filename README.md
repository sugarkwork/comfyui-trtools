# ComfyUI-TRTools

画像のタグ化のモデルと、アップスケーラーモデルをTensorRT を活用して NVIDIA GPU に最適化する事で、処理が凄く速くなります！

## 🚀 速度比較

### Tagger (画像タグ付け)
WD14 Tagger (3.863s) -> **TRT Tagger (0.240s)**

<img width="1322" height="1064" alt="image" src="https://github.com/user-attachments/assets/b1edba60-68c2-440f-a493-08a88d0e0ed1" />

### Upscaler (高画質化)
標準アップスケーラー（0.182s + 3.523s） -> **TRT Upscaler (1.648s)**

<img width="1111" height="1107" alt="image" src="https://github.com/user-attachments/assets/dea76fa5-43bb-4f43-9d9a-b78545cef131" />


## ✨ 特徴

- **自動セットアップ**: 使用するモデルを自動的にダウンロードし、TensorRTエンジンに変換します。
- **グローバルキャッシュ**: 一度ロードしたエンジンはメモリ上で共有されるため、複数のノードを使用してもメモリ効率が良く、2回目以降の実行が爆速です（たぶんね）
- **簡単インストール**: システムへの複雑なインストール作業を極力減らしました。

## 📦 インストール

1. ComfyUIの `custom_nodes` フォルダにこのリポジトリをクローンします。
2. 必要なライブラリをインストールします。

```bash
pip install -r requirements.txt
```

※ `tensorrt` は環境に合わせてインストールしてください（CUDA 12.x の場合は、通常は `pip install tensorrt-cu12` CUDA 13.x の場合は、通常は `pip install tensorrt-cu13` か `pip install tensorrt`）。

## 🛠️ 使い方

### TRT Tagger
画像を接続し、モデルを選択するだけでタグ（String）が出力されます。
初回実行時はモデルのダウンロードと変換が行われるため、環境によっては ** ２～５分間ほど ** 時間がかかります。

### TRT Upscaler
画像を接続し、モデルと解像度設定を行うだけで高速にアップスケールされます。
こちらも初回はエンジンビルドに時間がかかります。

## ⚠️ 注意点

- **初回はフリーズしたように見えます**: モデルのダウンロード＋TensorRT への変換のためです。コンソールを確認してください。
- **2回目以降が本番**: 変換されたエンジンはキャッシュされ、次回以降は超高速に動作します。

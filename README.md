# Whisper Voice Typer

[[Blog]](https://openai.com/blog/whisper)
[[Paper]](https://arxiv.org/abs/2212.04356)
[[Model card]](https://github.com/openai/whisper/blob/main/model-card.md)
[[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)



OpenAIのWhisperを高速化した「faster-whisper」を利用した、Windows向けの常駐型音声入力・自動ペーストツールです。
特定のキー（デフォルトはF8）を押している間だけ録音し、離すと即座にテキスト化してアクティブなウィンドウに貼り付けます。

## 主な機能

- **プッシュ・トゥ・トーク**: 設定したキーを押し続けている間だけマイク入力をキャプチャ。
- **自動ペースト**: 解析完了後、クリップボード経由で即座に `Ctrl + V` を実行。
- **タスクトレイ常駐**: 邪魔にならないようタスクトレイに格納。右クリックから設定や終了が可能。
- **視覚的なフィードバック**: 録音中は「● REC」、解析中は「⌛ 分析中...」と画面端にインジケーターを表示。
- **GUI設定画面**: 使用モデル、ショートカットキー、初期プロンプトを自由に変更可能。
- **CPU最適化**: `int8` 量子化により、ノートPC（Core i5-8250U等）でも実用的な速度で動作。

## 動作環境

- **OS**: Windows 10 / 11
- **Python**: 3.8以上
- **必須外部ソフト**: **FFmpeg**
  - インストールされていない場合、音声の読み込みでエラーになります。
  - 管理者用PowerShellで `winget install ffmpeg` を実行してインストールしてください。

## セットアップ

### 1. ライブラリのインストール
```bash
pip install -r requirements.txt
```
※ pyaudio のインストールでエラーが出る場合は、pip install pipwin の後に pipwin install pyaudio を試してください。


## 使い方

1. python app.py を実行します。

2. タスクトレイにアイコンが表示され、ステータスが「準備完了」になったら使用可能です。

3. [F8] キーを押し続けながらマイクに向かって話します。

4. キーを離すと解析が始まり、数秒後にカーソル位置に文字が自動入力されます。

## 設定
+ タスクトレイのアイコンを右クリックし、「設定」を選択すると以下の項目を変更できます。

+ 使用モデル: 速度重視なら base、精度重視なら small や large-v3-turbo を推奨。

+ 録音キー: 好きなキーに変更可能（ボタンクリック後にキーを押す）。

+ 初期プロンプト: 日本語の認識精度や句読点の有無を調整するためのヒント文章。

## ライセンス
MIT License
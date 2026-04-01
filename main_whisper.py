import whisper
import speech_recognition as sr
import io
import numpy as np
import torch

# モデルのロード (base, small, mediumなど)
# 速度重視（まずはここから）
# model = whisper.load_model("base")

# 精度を上げたい場合（少し待ち時間が出るかもしれません）
model = whisper.load_model("small")

# マイク設定
r = sr.Recognizer()


def main():
    print("マイクに向かって話してください... (Ctrl+C で終了)")

    with sr.Microphone(sample_rate=16000) as source:
        # 周囲のノイズに合わせて調整
        r.adjust_for_ambient_noise(source)

        while True:
            try:
                # 音声を待機
                audio = r.listen(source)
                print("解析中...")

                # 音声データをnumpy形式に変換してWhisperに渡す
                wav_bytes = audio.get_wav_data()
                wav_stream = io.BytesIO(wav_bytes)

                # 録音データを一時的なファイルとして扱わずにWhisperで処理
                # (簡易的な方法として一度ファイルに書き出す方法もありますが、ここではメモリ上で処理)
                import tempfile
                import os

                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                    temp_file.write(wav_bytes)
                    temp_file_path = temp_file.name

                # 文字起こし実行
                result = model.transcribe(temp_file_path, language="ja")
                print(f"結果: {result['text']}")

                # 一時ファイルの削除
                os.remove(temp_file_path)

            except KeyboardInterrupt:
                print("終了します。")
                break
            except Exception as e:
                print(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
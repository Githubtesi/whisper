import whisper # Whisperのインポート
model = whisper.load_model('large') # 利用するモデルを指定（ここでは最も精度の高い'large'を指定）
result = model.transcribe('音声データ.mp3') # resultに文字起こししたテキストデータを格納
with open("書き起こしテキスト.txt", "w") as f:
    print(result['text'], file=f) # テキストデータを書き出し
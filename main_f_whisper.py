# import os
# import wave
# import threading
# import collections
# import pyaudio
# import pyperclip
# from pynput import keyboard
# from faster_whisper import WhisperModel
#
# # --- 設定 ---
# # MODEL_SIZE = "small"  # i5-8250Uならsmallが精度的におすすめ
# # MODEL_SIZE = "medium"  # i5-8250Uならsmallが精度的におすすめ
# # MODEL_SIZE = "large"  # i5-8250Uならsmallが精度的におすすめ
# MODEL_SIZE = "large-v3-turbo"  # large-v3 より圧倒的に速く、精度は medium よりも高いという「いいとこ取り」を狙っています。
# TARGET_KEY = keyboard.Key.f8  # 録音に使用するキー
# CHUNK = 1024
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
#
# # モデルの初期化
# model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
# controller = keyboard.Controller()
#
#
# class VoiceTyper:
#     def __init__(self):
#         self.is_recording = False
#         self.frames = []
#         self.p = pyaudio.PyAudio()
#
#     def start_recording(self):
#         if self.is_recording: return
#         self.is_recording = True
#         self.frames = []
#         print("● 録音中...")
#
#         # 録音用スレッド開始
#         self.thread = threading.Thread(target=self._record_task)
#         self.thread.start()
#
#     def _record_task(self):
#         stream = self.p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
#                              input=True, frames_per_buffer=CHUNK)
#         while self.is_recording:
#             data = stream.read(CHUNK)
#             self.frames.append(data)
#         stream.stop_stream()
#         stream.close()
#
#     def stop_recording(self):
#         if not self.is_recording: return
#         self.is_recording = False
#         print("○ 解析中...")
#
#         # 一時保存して文字起こし
#         temp_file = "temp_voice.wav"
#         wf = wave.open(temp_file, 'wb')
#         wf.setnchannels(CHANNELS)
#         wf.setsampwidth(self.p.get_sample_size(FORMAT))
#         wf.setframerate(RATE)
#         wf.writeframes(b''.join(self.frames))
#         wf.close()
#
#         # Whisperで解析
#         segments, _ = model.transcribe(temp_file, language="ja", beam_size=5,
#                                        initial_prompt="句読点をつけて自然な日本語で。")
#         text = "".join([s.text for s in segments])
#
#         if text.strip():
#             print(f"結果: {text}")
#             # クリップボードにコピーして貼り付け
#             pyperclip.copy(text)
#             controller.press(keyboard.Key.ctrl)
#             controller.press('v')
#             controller.release('v')
#             controller.release(keyboard.Key.ctrl)
#
#         if os.path.exists(temp_file):
#             os.remove(temp_file)
#
#
# # 制御用インスタンス
# typer = VoiceTyper()
#
#
# def on_press(key):
#     if key == TARGET_KEY:
#         typer.start_recording()
#
#
# def on_release(key):
#     if key == TARGET_KEY:
#         typer.stop_recording()
#
#
# print(f"{TARGET_KEY} を押している間だけ録音します...")
# with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
#     listener.join()



# main_f_whisper.py の中身を整理
from faster_whisper import WhisperModel

class WhisperEngine:
    def __init__(self, model_size="large-v3-turbo"):
        self.model = WhisperModel(model_size, device="cpu", compute_type="int8")

    def transcribe(self, audio_path):
        segments, _ = self.model.transcribe(audio_path, language="ja")
        return "".join([s.text for s in segments])
import io
import os
import sys
import asyncio
import edge_tts
import time

import numpy as np

# EXE化された環境（frozen）かどうかを判定
if getattr(sys, 'frozen', False):
    # EXE実行時のパス（_internalフォルダを指すように調整）
    base_path = sys._MEIPASS
    # ライブラリがある場所を明示的に指定
    torch_lib_path = os.path.join(base_path, "torch", "lib")
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
    # EXEの直下も検索対象にする
    os.add_dll_directory(os.path.dirname(sys.executable))
import pygame
import json
import threading
import wave
import pyaudio
import pyperclip
import customtkinter as ctk
from pynput import keyboard
from pystray import Icon, Menu, MenuItem
from PIL import Image, ImageDraw
from faster_whisper import WhisperModel
from openai import OpenAI
import io

# クライアントの初期化（APIキーを設定）
client = OpenAI(api_key="あなたのAPIキー")
# --- DEFAULT_CONFIG に新しい項目を追加 ---
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "model_size": "large-v3",
    "device": "cuda",
    "compute_type": "float16",
    "hotkey": ["ctrl_l", "cmd"],
    "initial_prompt": "こんにちは。日本語の文字起こしです。句読点をつけて自然に変換してください。",
    "input_language": "ja",          # 新規追加: "ja", "en", "auto"
    "output_language": "ja",          # 新規追加: "ja"（日本語） or "en"（英語翻訳）
    "show_both_languages": False,  # 両方貼り付けチェック
    "enable_audio_output": False,  # 音声出力チェック
}

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
TEMP_FILE = "temp_recording.wav"


class WhisperApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # 1. 設定の読み込み
        self.config = self.load_config()

        # --- 2. GUIの初期設定 ---
        self.title("Whisper Voice Settings")
        self.geometry("400x820")  # 470から520へ変更
        self.protocol('WM_DELETE_WINDOW', self.hide_window)

        # 3. 状態管理
        self.is_recording = False
        self.is_loading_model = False
        self.waiting_for_key = False
        self.frames = []
        self.model = None
        self.kb_controller = keyboard.Controller()
        self.audio = pyaudio.PyAudio()

        # キー判定用の状態
        self.pressed_keys = set()
        self.captured_keys = []
        self.target_hotkeys = self.get_target_keys()

        # 4. UIコンポーネントの作成
        self.create_widgets()

        # 5. インジケーター (REC表示用)
        self.setup_indicator()

        # 6. エンジンの初期化 (別スレッド)
        # self.update_model_async(
        #     self.config["model_size"],
        #     self.config["device"],
        #     self.config["compute_type"]
        # )
        
        # 7. 裏方の起動
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.setup_tray()
        self.pygame_initialized = False

    async def speak_english(self, english_text: str):
        """ファイルを生成せず、メモリ上のバッファから再生する"""
        if not english_text or not english_text.strip():
            return

        voice = "en-US-AndrewNeural"

        try:
            if not self.pygame_initialized:
                pygame.mixer.pre_init(frequency=24000, size=-16, channels=1, buffer=1024)
                pygame.mixer.init()
                self.pygame_initialized = True
                self.pygame_module = pygame

            # 1. edge-ttsの結果をメモリ上のBytesIOバッファに書き込む
            mp3_buffer = io.BytesIO()
            communicate = edge_tts.Communicate("   " + english_text, voice=voice)

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.write(chunk["data"])

            # 2. バッファの先頭に戻してpygameにロード
            mp3_buffer.seek(0)
            self.pygame_module.mixer.music.unload()
            # pygame 2系はファイルオブジェクトを直接ロード可能
            self.pygame_module.mixer.music.load(mp3_buffer, "mp3")
            self.pygame_module.mixer.music.play()

            while self.pygame_module.mixer.music.get_busy():
                self.pygame_module.time.Clock().tick(20)

            print("音声再生完了（オンメモリ）")

        except Exception as e:
            print(f"音声再生エラー: {e}")
        # finallyでのファイル削除処理も不要になります

    # --- UI作成 ---
    def create_widgets(self):

        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text="Whisper 音声入力設定", font=("Meiryo", 20, "bold")).grid(row=0, column=0, pady=20)

        # 1. モデル選択 (row 1-2)
        ctk.CTkLabel(self, text="使用モデル:").grid(row=1, column=0, sticky="w", padx=40)
        self.model_menu = ctk.CTkOptionMenu(self, values=["tiny", "base", "small", "medium","large-v3", "large-v3-turbo"],
                                            command=self.on_config_change)
        self.model_menu.set(self.config.get("model_size", "small"))
        self.model_menu.grid(row=2, column=0, padx=40, pady=(0, 10), sticky="ew")

        # 2. デバイス選択 (row 3-4)
        ctk.CTkLabel(self, text="使用デバイス (GPUならcuda):").grid(row=3, column=0, sticky="w", padx=40)
        self.device_menu = ctk.CTkOptionMenu(self, values=["cpu", "cuda"], command=self.on_config_change)
        self.device_menu.set(self.config.get("device", "cpu"))
        self.device_menu.grid(row=4, column=0, padx=40, pady=(0, 10), sticky="ew")

        # 3. 計算タイプ選択 (row 5-6)
        # ラベルのテキストを CPU/GPU 両方の推奨がわかるように変更
        ctk.CTkLabel(self, text="計算タイプ (CPU:int8 / GPU:float16推奨):").grid(row=5, column=0, sticky="w", padx=40)
        self.compute_menu = ctk.CTkOptionMenu(self, values=["int8", "float16", "int8_float16", "float32"],
                                              command=self.on_config_change)
        self.compute_menu.set(self.config.get("compute_type", "int8"))
        self.compute_menu.grid(row=6, column=0, padx=40, pady=(0, 10), sticky="ew")

        # 4. 録音キー設定 (row 7-8)
        ctk.CTkLabel(self, text="録音キー (複数可):").grid(row=7, column=0, sticky="w", padx=40)
        hotkey_text = " + ".join(self.config["hotkey"]).upper()
        self.key_button = ctk.CTkButton(self, text=hotkey_text, command=self.start_key_capture)
        self.key_button.grid(row=8, column=0, padx=40, pady=(0, 10), sticky="ew")

        # 5. プロンプト設定 (row 9-10)
        ctk.CTkLabel(self, text="初期プロンプト:").grid(row=9, column=0, sticky="w", padx=40)
        self.prompt_entry = ctk.CTkEntry(self)
        self.prompt_entry.insert(0, self.config["initial_prompt"])
        self.prompt_entry.grid(row=10, column=0, padx=40, pady=(0, 10), sticky="ew")

        # === 新規追加：入力言語 ===
        row = 13  # 既存項目の次の行から開始（実際の行番号は既存コードに合わせて調整してください）
        ctk.CTkLabel(self, text="入力言語 (音声認識言語):").grid(row=row, column=0, sticky="w", padx=40)
        self.input_lang_menu = ctk.CTkOptionMenu(
            self,
            values=["日本語優先", "英語優先", "自動検出"],
            command=self.on_config_change
        )
        current_input = self.config.get("input_language", "ja")
        self.input_lang_menu.set({
                                     "ja": "日本語優先",
                                     "en": "英語優先",
                                     "auto": "自動検出"
                                 }.get(current_input, "日本語優先"))
        self.input_lang_menu.grid(row=row + 1, column=0, padx=40, pady=(0, 10), sticky="ew")

        # === 新規追加：出力言語 ===
        row += 2
        ctk.CTkLabel(self, text="出力言語 (最終出力):").grid(row=row, column=0, sticky="w", padx=40)
        self.output_lang_menu = ctk.CTkOptionMenu(
            self,
            values=["日本語", "英語（翻訳）"],
            command=self.on_config_change
        )
        current_output = self.config.get("output_language", "ja")
        self.output_lang_menu.set("日本語" if current_output == "ja" else "英語（翻訳）")
        self.output_lang_menu.grid(row=row + 1, column=0, padx=40, pady=(0, 10), sticky="ew")





        # === 翻訳オプション（修正版）===
        row += 2

        ctk.CTkLabel(self, text="【翻訳オプション】", font=("Meiryo", 14, "bold")).grid(
            row=row, column=0, sticky="w", padx=40, pady=(25, 8))

        row += 1
        # 両方貼り付けチェック
        self.show_both_var = ctk.BooleanVar(value=self.config.get("show_both_languages", False))
        self.show_both_checkbox = ctk.CTkCheckBox(
            self,
            text="翻訳時に日本語と英語両方を貼り付ける",
            variable=self.show_both_var,
            command=self.on_config_change,
            width=320
        )
        self.show_both_checkbox.grid(row=row, column=0, padx=40, pady=6, sticky="w")

        row += 1
        # 音声出力チェック
        self.audio_output_var = ctk.BooleanVar(value=self.config.get("enable_audio_output", False))
        self.audio_output_checkbox = ctk.CTkCheckBox(
            self,
            text="翻訳時に音声出力を行う（英語のみ）",
            variable=self.audio_output_var,
            command=self.on_config_change,
            width=320
        )
        self.audio_output_checkbox.grid(row=row, column=0, padx=40, pady=6, sticky="w")

        # 保存ボタン（既存のものを下に移動）
        self.save_button = ctk.CTkButton(self, text="設定を保存して適用", fg_color="green", command=self.save_settings)
        self.save_button.grid(row=row + 3, column=0, padx=40, pady=20, sticky="ew")

        # ステータスラベル
        self.status_label = ctk.CTkLabel(self, text="準備完了", text_color="gray")
        self.status_label.grid(row=row + 4, column=0, pady=10)

    def transcribe_audio(self, audio_bytes):
        """
        ローカルモデルを使わず、OpenAI APIを使って文字起こしと翻訳を行う
        """
        try:
            # 1. 録音データをメモリ上のWAVファイル形式にする
            # APIはファイル形式（wav/mp3等）である必要があるため、メモリ内でWAVヘッダーを付けます
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(audio_bytes)

            buffer.seek(0)
            # OpenAI APIがファイル名を要求するため、仮想的な名前を付けます
            buffer.name = "temp_recording.wav"

            # 2. 日本語文字起こし（Transcribe）
            # initial_prompt は設定画面から引き継げます
            transcript_jp = client.audio.transcriptions.create(
                model="whisper-1",
                file=buffer,
                language="ja",
                prompt=self.config.get("initial_prompt", "")
            )
            japanese_text = transcript_jp.text

            english_text = None
            # 3. 翻訳が必要な場合（Translate）
            if self.config.get("output_language") == "en":
                buffer.seek(0)  # バッファを最初に戻して再利用
                translation = client.audio.translations.create(
                    model="whisper-1",
                    file=buffer
                )
                english_text = translation.text

            return {"japanese": japanese_text, "english": english_text}

        except Exception as e:
            print(f"API Error: {e}")
            return {"japanese": f"APIエラー: {e}", "english": None}

    def _process_thread(self):
        """録音バイナリをnumpyに変換して処理する"""
        try:
            # 1. 録音された全バイトデータを結合
            audio_bytes = b''.join(self.frames)

            # 2. バイナリ(int16)をWhisper用のnumpy(float32)に変換
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # 3. Whisper実行
            result = self.transcribe_audio(audio_np)
            japanese_text = result["japanese"].strip()
            english_text = result.get("english", "").strip()

            if not japanese_text:
                return

            # --- 以下、出力ロジック ---
            if english_text:
                output_text = f"{english_text}\n{japanese_text}" if self.config.get(
                    "show_both_languages") else english_text
                if self.config.get("enable_audio_output"):
                    asyncio.run(self.speak_english(english_text))
            else:
                output_text = japanese_text

            pyperclip.copy(output_text)
            self.kb_controller.press(keyboard.Key.ctrl)
            self.kb_controller.press('v')
            self.kb_controller.release('v')
            self.kb_controller.release(keyboard.Key.ctrl)

            self.after(0, lambda: self.status_label.configure(text="出力完了", text_color="green"))

        except Exception as e:
            print(f"処理エラー: {e}")
            self.after(0, lambda: self.status_label.configure(text="エラー発生", text_color="red"))
        finally:
            self.after(0, self.hide_indicator)

    # 設定変更時の共通ハンドラ
    def on_config_change(self, _=None):
        # 基本設定
        self.config["model_size"] = self.model_menu.get()
        self.config["device"] = self.device_menu.get()
        self.config["compute_type"] = self.compute_menu.get()
        self.config["initial_prompt"] = self.prompt_entry.get()

        # 新規追加した設定
        input_map = {"日本語優先": "ja", "英語優先": "en", "自動検出": "auto"}
        self.config["input_language"] = input_map.get(self.input_lang_menu.get(), "ja")

        self.config["output_language"] = "en" if self.output_lang_menu.get() == "英語（翻訳）" else "ja"

        # ★★★ チェックボックスの値を保存 ★★★
        self.config["show_both_languages"] = self.show_both_var.get()
        self.config["enable_audio_output"] = self.audio_output_var.get()

        # モデル再ロード
        self.update_model_async(
            self.config["model_size"],
            self.config["device"],
            self.config["compute_type"]
        )

    # --- キー変換ロジック ---
    def get_target_keys(self):
        target = set()
        for k in self.config["hotkey"]:
            if hasattr(keyboard.Key, k):
                target.add(getattr(keyboard.Key, k))
            else:
                try:
                    target.add(keyboard.KeyCode.from_char(k))
                except:
                    target.add(k)
        return target

    def setup_indicator(self):
        self.indicator = ctk.CTkToplevel(self)
        self.indicator.overrideredirect(True)
        self.indicator.attributes("-topmost", True)
        self.indicator.attributes("-alpha", 0.9)
        self.indicator.geometry("100x40+20+20")
        self.indicator.configure(fg_color="#222222")
        self.indicator_label = ctk.CTkLabel(self.indicator, text="● REC", text_color="#FF4444",
                                            font=("Meiryo", 14, "bold"))
        self.indicator_label.pack(expand=True)
        self.indicator.withdraw()

    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                conf = json.load(f)
                # 新しい設定項目がない場合の補完
                if "device" not in conf:
                    conf["device"] = ("cuda")
                if "compute_type" not in conf:
                    conf["compute_type"] = "int16"

                if "input_language" not in conf:
                    conf["input_language"] = "ja"
                if "output_language" not in conf:
                    conf["output_language"] = "ja"

                if "show_both_languages" not in conf:
                    conf["show_both_languages"] = False
                if "enable_audio_output" not in conf:
                    conf["enable_audio_output"] = False

                return conf
        return DEFAULT_CONFIG.copy()


    def save_settings(self):
        self.config["model_size"] = self.model_menu.get()
        self.config["device"] = self.device_menu.get()
        self.config["compute_type"] = self.compute_menu.get()
        self.config["initial_prompt"] = self.prompt_entry.get()

        # 新規追加
        input_map = {"日本語優先": "ja", "英語優先": "en", "自動検出": "auto"}
        self.config["input_language"] = input_map.get(self.input_lang_menu.get(), "ja")

        self.config["output_language"] = "en" if self.output_lang_menu.get() == "英語（翻訳）" else "ja"



        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        self.status_label.configure(text="設定を保存しました", text_color="green")

    def on_model_change(self, new_size):
        self.config["model_size"] = new_size
        self.update_model_async(new_size)

    def update_model_async(self, size, device, compute_type):
        def load():
            self.is_loading_model = True
            self.status_label.configure(text=f"ロード中 ({size}/{device}/{compute_type})...", text_color="orange")
            try:
                # 設定値を反映してロード
                self.model = WhisperModel(size, device=device, compute_type=compute_type)
                self.status_label.configure(text="準備完了", text_color="green")
            except Exception as e:
                self.status_label.configure(text=f"エラー: GPU設定を確認してください", text_color="red")
                print(f"Model load error: {e}")
            finally:
                self.is_loading_model = False

        threading.Thread(target=load, daemon=True).start()
    def start_key_capture(self):
        self.waiting_for_key = True
        self.captured_keys = []
        self.key_button.configure(text="すべて押して離すと確定...", fg_color="orange")

    # --- 録音制御メソッド ---
    def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.after(0, lambda: self.show_indicator("REC"))
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self):
        stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while self.is_recording:
            self.frames.append(stream.read(CHUNK))
        stream.stop_stream()
        stream.close()

    def stop_recording(self):
        self.is_recording = False
        self.after(0, lambda: self.show_indicator("ANALYZING"))
        threading.Thread(target=self._process_thread, daemon=True).start()

    # --- キーイベント ---
    def on_press(self, key):
        if self.waiting_for_key:
            k_name = str(key).replace("Key.", "")
            if k_name not in self.captured_keys:
                self.captured_keys.append(k_name)
            self.pressed_keys.add(key)
            return

        self.pressed_keys.add(key)
        # セット比較ですべてのターゲットが押されているか判定
        if self.target_hotkeys.issubset(self.pressed_keys):
            if not self.is_recording and not self.is_loading_model:
                self.start_recording()

    def on_release(self, key):
        if self.waiting_for_key:
            self.pressed_keys.discard(key)
            if not self.pressed_keys:  # 全キー離されたら確定
                self.config["hotkey"] = self.captured_keys
                self.target_hotkeys = self.get_target_keys()
                self.key_button.configure(text=" + ".join(self.captured_keys).upper(),
                                          fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
                self.waiting_for_key = False
            return

        # ターゲットのいずれか1つでも離されたら停止
        if key in self.target_hotkeys:
            if self.is_recording:
                self.stop_recording()

        self.pressed_keys.discard(key)


    # --- トレイ・ウィンドウ制御 ---
    def setup_tray(self):
        img = Image.new('RGB', (64, 64), (30, 30, 30))
        draw = ImageDraw.Draw(img)
        draw.ellipse((10, 10, 54, 54), fill=(0, 150, 255))
        menu = Menu(MenuItem('設定', self.show_window), MenuItem('終了', self.quit_app))
        self.tray_icon = Icon("Whisper", img, "Whisper Voice", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def show_indicator(self, mode):
        color = "#FF4444" if mode == "REC" else "#FFCC00"
        text = "● REC" if mode == "REC" else "⌛ 分析中..."
        self.indicator_label.configure(text=text, text_color=color)
        self.indicator.deiconify()

    def hide_indicator(self):
        self.indicator.withdraw()

    def show_window(self):
        self.after(0, self.deiconify)

    def hide_window(self):
        self.withdraw()

    def quit_app(self):
        self.tray_icon.stop()
        self.quit()
        sys.exit()


if __name__ == "__main__":
    app = WhisperApp()
    app.mainloop()

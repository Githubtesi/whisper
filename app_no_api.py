import io
import os
import sys
import asyncio
import edge_tts
import time
import numpy as np
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
import translators as ts  # 追加
import html  # 追加

# EXE化対応
if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS
    torch_lib_path = os.path.join(base_path, "torch", "lib")
    if os.path.exists(torch_lib_path):
        os.add_dll_directory(torch_lib_path)
    os.add_dll_directory(os.path.dirname(sys.executable))

CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "model_size": "base",
    "device": "cpu",  # デフォルトは安全なCPU
    "compute_type": "int8",
    "hotkey": ["ctrl_l", "cmd"],
    "initial_prompt": "こんにちは。日本語の文字起こしです。",
    "input_language": "ja",
    "output_language": "ja",
    "show_both_languages": False,
    "enable_audio_output": False,
}

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000


class WhisperApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()

        self.title("Whisper + DeepL Voice")
        self.geometry("400x820")
        self.protocol('WM_DELETE_WINDOW', self.hide_window)

        self.is_recording = False
        self.is_loading_model = False
        self.waiting_for_key = False
        self.frames = []
        self.model = None
        self.kb_controller = keyboard.Controller()
        self.audio = pyaudio.PyAudio()

        self.pressed_keys = set()
        self.captured_keys = []
        self.target_hotkeys = self.get_target_keys()

        self.create_widgets()
        self.setup_indicator()

        # モデルの初期ロード
        self.update_model_async(
            self.config["model_size"],
            self.config["device"],
            self.config["compute_type"]
        )

        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.setup_tray()
        self.pygame_initialized = False

    # --- 音声再生ロジック ---
    async def speak_english(self, english_text: str):
        if not english_text or not english_text.strip(): return
        voice = "en-US-AndrewNeural"
        try:
            if not self.pygame_initialized:
                pygame.mixer.pre_init(frequency=24000, size=-16, channels=1, buffer=1024)
                pygame.mixer.init()
                self.pygame_initialized = True

            mp3_buffer = io.BytesIO()
            communicate = edge_tts.Communicate(english_text, voice=voice)
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    mp3_buffer.write(chunk["data"])

            mp3_buffer.seek(0)
            pygame.mixer.music.load(mp3_buffer, "mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
        except Exception as e:
            print(f"音声再生エラー: {e}")

    # --- 文字起こし & 翻訳ロジック ---
    def transcribe_and_translate(self, audio_data: np.ndarray):
        """ローカルWhisperで文字起こしし、DeepL(Web)で翻訳する"""
        if not self.model: return {"japanese": "モデル未ロード", "english": None}

        input_lang = self.config.get("input_language", "ja")
        whisper_lang = None if input_lang == "auto" else input_lang

        try:
            # 1. 日本語文字起こし (Local Whisper)
            segments, _ = self.model.transcribe(
                audio_data,
                language=whisper_lang,
                initial_prompt=self.config.get("initial_prompt", ""),
                vad_filter=True
            )
            japanese_text = "".join(segment.text for segment in segments).strip()

            if not japanese_text:
                return {"japanese": "", "english": ""}

            english_text = ""
            # 2. DeepL翻訳 (Webシミュレート)
            if self.config.get("output_language") == "en":
                print(f"DeepL翻訳中: {japanese_text}")
                try:
                    # translatorsを使用して翻訳し、HTML特殊文字をデコード
                    translated_raw = ts.translate_text(japanese_text, from_language='ja', to_language='en',
                                                       engine='deepl')
                    english_text = html.unescape(translated_raw)
                except Exception as e:
                    print(f"DeepL翻訳エラー: {e}")
                    english_text = "[Translation Error]"

            return {"japanese": japanese_text, "english": english_text}
        except Exception as e:
            print(f"処理エラー: {e}")
            return {"japanese": f"Error: {str(e)}", "english": None}

    def _process_thread(self):
        try:
            audio_bytes = b''.join(self.frames)
            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            # 文字起こしと翻訳を実行
            result = self.transcribe_and_translate(audio_np)

            jp_text = result["japanese"]
            en_text = result.get("english", "")

            if not jp_text: return

            # 出力テキストの生成
            if en_text:
                output_text = f"{en_text}\n{jp_text}" if self.config.get("show_both_languages") else en_text
                if self.config.get("enable_audio_output"):
                    asyncio.run(self.speak_english(en_text))
            else:
                output_text = jp_text

            # クリップボード & 貼り付け
            pyperclip.copy(output_text)
            self.kb_controller.press(keyboard.Key.ctrl)
            self.kb_controller.press('v')
            self.kb_controller.release('v')
            self.kb_controller.release(keyboard.Key.ctrl)

            self.after(0, lambda: self.status_label.configure(text="完了", text_color="green"))
        except Exception as e:
            print(f"スレッドエラー: {e}")
        finally:
            self.after(0, self.hide_indicator)

    # --- UI & 設定関連 (元のコードを継承) ---
    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text="Whisper + DeepL", font=("Meiryo", 20, "bold")).grid(row=0, column=0, pady=20)

        # モデル・デバイス設定
        ctk.CTkLabel(self, text="使用モデル:").grid(row=1, column=0, sticky="w", padx=40)
        self.model_menu = ctk.CTkOptionMenu(self, values=["tiny", "base", "small", "medium","large-v3", "large-v3-turbo"],
                                            command=self.on_config_change)
        self.model_menu.set(self.config["model_size"])
        self.model_menu.grid(row=2, column=0, padx=40, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(self, text="デバイス (GPUならcuda):").grid(row=3, column=0, sticky="w", padx=40)
        self.device_menu = ctk.CTkOptionMenu(self, values=["cpu", "cuda"], command=self.on_config_change)
        self.device_menu.set(self.config["device"])
        self.device_menu.grid(row=4, column=0, padx=40, pady=(0, 10), sticky="ew")

        ctk.CTkLabel(self, text="計算タイプ (CPU:int8 / GPU:float16):").grid(row=5, column=0, sticky="w", padx=40)
        self.compute_menu = ctk.CTkOptionMenu(self, values=["int8", "float16", "float32"],
                                              command=self.on_config_change)
        self.compute_menu.set(self.config["compute_type"])
        self.compute_menu.grid(row=6, column=0, padx=40, pady=(0, 10), sticky="ew")

        # 録音キー
        ctk.CTkLabel(self, text="録音キー:").grid(row=7, column=0, sticky="w", padx=40)
        self.key_button = ctk.CTkButton(self, text=" + ".join(self.config["hotkey"]).upper(),
                                        command=self.start_key_capture)
        self.key_button.grid(row=8, column=0, padx=40, pady=(0, 10), sticky="ew")

        # 入出力言語
        ctk.CTkLabel(self, text="出力モード:").grid(row=9, column=0, sticky="w", padx=40)
        self.output_lang_menu = ctk.CTkOptionMenu(self, values=["日本語のみ", "英語（DeepL翻訳）"],
                                                  command=self.on_config_change)
        self.output_lang_menu.set("英語（DeepL翻訳）" if self.config["output_language"] == "en" else "日本語のみ")
        self.output_lang_menu.grid(row=10, column=0, padx=40, pady=(0, 10), sticky="ew")

        # オプション
        self.show_both_var = ctk.BooleanVar(value=self.config["show_both_languages"])
        ctk.CTkCheckBox(self, text="日・英両方を貼り付け", variable=self.show_both_var,
                        command=self.on_config_change).grid(row=11, column=0, padx=40, pady=5, sticky="w")

        self.audio_output_var = ctk.BooleanVar(value=self.config["enable_audio_output"])
        ctk.CTkCheckBox(self, text="翻訳時に音声で読み上げ", variable=self.audio_output_var,
                        command=self.on_config_change).grid(row=12, column=0, padx=40, pady=5, sticky="w")

        self.save_button = ctk.CTkButton(self, text="設定を保存", fg_color="green", command=self.save_settings)
        self.save_button.grid(row=13, column=0, padx=40, pady=20, sticky="ew")

        self.status_label = ctk.CTkLabel(self, text="準備完了", text_color="gray")
        self.status_label.grid(row=14, column=0, pady=10)

    # (以下、録音・キー判定・トレイアイコン等のロジックは元のコードと同一のため省略可能ですが、動作に必要です)
    def load_config(self):
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        return DEFAULT_CONFIG.copy()

    def save_settings(self):
        self.config["show_both_languages"] = self.show_both_var.get()
        self.config["enable_audio_output"] = self.audio_output_var.get()
        self.config["output_language"] = "en" if "英語" in self.output_lang_menu.get() else "ja"
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        self.status_label.configure(text="保存完了", text_color="green")

    def update_model_async(self, size, device, compute_type):
        def load():
            self.is_loading_model = True
            self.after(0, lambda: self.status_label.configure(text=f"モデルロード中...", text_color="orange"))
            try:
                self.model = WhisperModel(size, device=device, compute_type=compute_type)
                self.after(0, lambda: self.status_label.configure(text="準備完了", text_color="green"))
            except Exception as e:
                self.after(0, lambda: self.status_label.configure(text="ロード失敗", text_color="red"))
            finally:
                self.is_loading_model = False

        threading.Thread(target=load, daemon=True).start()

    def on_config_change(self, _=None):
        self.config["model_size"] = self.model_menu.get()
        self.config["device"] = self.device_menu.get()
        self.config["compute_type"] = self.compute_menu.get()
        self.update_model_async(self.config["model_size"], self.config["device"], self.config["compute_type"])

    def get_target_keys(self):
        target = set()
        for k in self.config["hotkey"]:
            if hasattr(keyboard.Key, k):
                target.add(getattr(keyboard.Key, k))
            else:
                target.add(keyboard.KeyCode.from_char(k))
        return target

    def on_press(self, key):
        if self.waiting_for_key:
            k_name = str(key).replace("Key.", "")
            if k_name not in self.captured_keys: self.captured_keys.append(k_name)
            self.pressed_keys.add(key)
            return
        self.pressed_keys.add(key)
        if self.target_hotkeys.issubset(self.pressed_keys):
            if not self.is_recording and not self.is_loading_model: self.start_recording()

    def on_release(self, key):
        if self.waiting_for_key:
            self.pressed_keys.discard(key)
            if not self.pressed_keys:
                self.config["hotkey"] = self.captured_keys
                self.target_hotkeys = self.get_target_keys()
                self.key_button.configure(text=" + ".join(self.captured_keys).upper())
                self.waiting_for_key = False
            return
        if key in self.target_hotkeys and self.is_recording: self.stop_recording()
        self.pressed_keys.discard(key)

    def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.show_indicator("REC")
        threading.Thread(target=self._record_thread, daemon=True).start()

    def _record_thread(self):
        stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        while self.is_recording: self.frames.append(stream.read(CHUNK))
        stream.stop_stream();
        stream.close()

    def stop_recording(self):
        self.is_recording = False
        self.show_indicator("ANALYZING")
        threading.Thread(target=self._process_thread, daemon=True).start()

    def setup_indicator(self):
        self.indicator = ctk.CTkToplevel(self);
        self.indicator.overrideredirect(True)
        self.indicator.attributes("-topmost", True);
        self.indicator.geometry("120x40+20+20")
        self.indicator_label = ctk.CTkLabel(self.indicator, text="● REC", text_color="red", font=("Arial", 14, "bold"))
        self.indicator_label.pack(expand=True);
        self.indicator.withdraw()

    def show_indicator(self, mode):
        text = "● REC" if mode == "REC" else "⌛ 翻訳中..."
        color = "red" if mode == "REC" else "orange"
        self.after(0, lambda: (self.indicator_label.configure(text=text, text_color=color), self.indicator.deiconify()))

    def hide_indicator(self):
        self.after(0, self.indicator.withdraw)

    def setup_tray(self):
        img = Image.new('RGB', (64, 64), (30, 30, 30))
        draw = ImageDraw.Draw(img);
        draw.ellipse((10, 10, 54, 54), fill=(0, 150, 255))
        menu = Menu(MenuItem('設定', self.show_window), MenuItem('終了', self.quit_app))
        self.tray_icon = Icon("Whisper", img, "Whisper Voice", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def show_window(self):
        self.after(0, self.deiconify)

    def hide_window(self):
        self.withdraw()

    def quit_app(self):
        self.tray_icon.stop(); self.quit(); sys.exit()

    def start_key_capture(self):
        self.waiting_for_key = True;
        self.captured_keys = []
        self.key_button.configure(text="キーを押してください...", fg_color="orange")


if __name__ == "__main__":
    app = WhisperApp()
    app.mainloop()

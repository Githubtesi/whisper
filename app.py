import os
import json
import threading
import wave
import sys
import pyaudio
import pyperclip
import customtkinter as ctk
from pynput import keyboard
from pystray import Icon, Menu, MenuItem
from PIL import Image, ImageDraw
from faster_whisper import WhisperModel

# --- 定数・デフォルト設定 ---
CONFIG_FILE = "config.json"
DEFAULT_CONFIG = {
    "model_size": "small",
    "device": "cpu",          # 追加: cpu または cuda
    "compute_type": "int8",   # 追加: int8, float16, int8_float16 など
    "hotkey": ["f8"],
    "initial_prompt": "こんにちは。日本語の文字起こしです。句読点をつけて自然に変換してください。"
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
        self.geometry("400x520")  # 470から520へ変更
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
        self.update_model_async(
            self.config["model_size"],
            self.config["device"],
            self.config["compute_type"]
        )

        # 7. 裏方の起動
        self.listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        self.listener.start()
        self.setup_tray()

    # --- UI作成 ---
    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self, text="Whisper 音声入力設定", font=("Meiryo", 20, "bold")).grid(row=0, column=0, pady=20)

        # 1. モデル選択 (row 1-2)
        ctk.CTkLabel(self, text="使用モデル:").grid(row=1, column=0, sticky="w", padx=40)
        self.model_menu = ctk.CTkOptionMenu(self, values=["tiny", "base", "small", "medium", "large-v3-turbo"],
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

        # 6. 保存ボタン (row 11)
        self.save_button = ctk.CTkButton(self, text="設定を保存して適用", fg_color="green", command=self.save_settings)
        self.save_button.grid(row=11, column=0, padx=40, pady=20, sticky="ew")

        # 7. ステータス (row 12)
        self.status_label = ctk.CTkLabel(self, text="準備完了", text_color="gray")
        self.status_label.grid(row=12, column=0, pady=10)
    # 設定変更時の共通ハンドラ
    def on_config_change(self, _=None):
        self.config["model_size"] = self.model_menu.get()
        self.config["device"] = self.device_menu.get()
        self.config["compute_type"] = self.compute_menu.get()

        # モデルの再ロード
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
                    conf["device"] = "cpu"
                if "compute_type" not in conf:
                    conf["compute_type"] = "int8"
                return conf
        return DEFAULT_CONFIG.copy()


    def save_settings(self):
        self.config["model_size"] = self.model_menu.get()
        self.config["device"] = self.device_menu.get()
        self.config["compute_type"] = self.compute_menu.get()
        self.config["initial_prompt"] = self.prompt_entry.get()
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

    def _process_thread(self):
        wf = wave.open(TEMP_FILE, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.audio.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(self.frames))
        wf.close()

        try:
            segments, _ = self.model.transcribe(TEMP_FILE, language="ja", beam_size=5,
                                                initial_prompt=self.config["initial_prompt"])
            text = "".join([s.text for s in segments])
            if text.strip():
                pyperclip.copy(text)
                self.kb_controller.press(keyboard.Key.ctrl)
                self.kb_controller.press('v')
                self.kb_controller.release('v')
                self.kb_controller.release(keyboard.Key.ctrl)
        finally:
            self.after(0, self.hide_indicator)
            if os.path.exists(TEMP_FILE): os.remove(TEMP_FILE)

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
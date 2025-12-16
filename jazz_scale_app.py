import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import numpy as np
from collections import Counter
import threading
import os
import winsound
import wave
import tempfile # 一時ファイル作成用に追加

# ==========================================
# 1. 分析ロジック (Backend)
# ==========================================
SCALE_PATTERNS = {
    'Ionian (Major)':     [0, 2, 4, 5, 7, 9, 11],
    'Dorian':             [0, 2, 3, 5, 7, 9, 10],
    'Phrygian':           [0, 1, 3, 5, 7, 8, 10],
    'Lydian':             [0, 2, 4, 6, 7, 9, 11],
    'Mixo-lydian':        [0, 2, 4, 5, 7, 9, 10],
    'Aeolian (Minor)':    [0, 2, 3, 5, 7, 8, 10],
    'Locrian':            [0, 1, 3, 5, 6, 8, 10],
    'Altered':            [0, 1, 3, 4, 6, 8, 10],
    'Diminished (H-W)':   [0, 1, 3, 4, 6, 7, 9, 10],
    'Diminished (W-H)':   [0, 2, 3, 5, 6, 8, 9, 11],
    'Wholetone':          [0, 2, 4, 6, 8, 10],
    'Phrygian Dominant':  [0, 1, 4, 5, 7, 8, 10],
    'Lydian Dominant':    [0, 2, 4, 6, 7, 9, 10],
    'Major Pentatonic':   [0, 2, 4, 7, 9],
    'Minor Pentatonic':   [0, 3, 5, 7, 10],
    'Blues Scale':        [0, 3, 5, 6, 7, 10],
    'Bebop Dominant':     [0, 2, 4, 5, 7, 9, 10, 11]
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def generate_all_scales():
    all_scales = {}
    for root_midi in range(12):
        root_name = NOTE_NAMES[root_midi]
        for scale_name, pattern in SCALE_PATTERNS.items():
            scale_notes = set([(root_midi + interval) % 12 for interval in pattern])
            full_scale_name = f"{root_name} {scale_name}"
            all_scales[full_scale_name] = scale_notes
    return all_scales

def analyze_audio(wav_path, progress_callback):
    try:
        progress_callback("音声データを読み込み中...")
        y, sr = librosa.load(wav_path, sr=None)
        
        progress_callback("ピッチ(音程)を抽出中...")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C6')
        )
        
        confident_f0 = f0[voiced_probs > 0.8]
        confident_f0 = confident_f0[~np.isnan(confident_f0)]

        if len(confident_f0) == 0:
            return None, "有効な音程が検出できませんでした。", None

        midi_notes = np.round(librosa.hz_to_midi(confident_f0)).astype(int)
        pitch_classes = [note % 12 for note in midi_notes]
        
        note_counts = Counter(pitch_classes)
        total_notes = sum(note_counts.values())
        min_count = total_notes * 0.05 
        melody_pitch_classes = set(
            [note for note, count in note_counts.items() if count >= min_count]
        )

        detected_notes = sorted([NOTE_NAMES[pc] for pc in melody_pitch_classes])
        
        progress_callback("スケール理論と照合中...")
        all_scales = generate_all_scales()
        
        scores = {}
        for scale_name, scale_notes in all_scales.items():
            match_count = len(melody_pitch_classes.intersection(scale_notes))
            if len(melody_pitch_classes) > 0:
                score = match_count / len(melody_pitch_classes)
            else:
                score = 0
            scores[scale_name] = score

        sorted_scales = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return sorted_scales, detected_notes, melody_pitch_classes

    except Exception as e:
        return None, str(e), None

# ==========================================
# 2. GUI用部品 (Virtual Keyboard with Sound Fix)
# ==========================================
class VirtualKeyboard(tk.Canvas):
    def __init__(self, master, width=700, height=120, **kwargs):
        super().__init__(master, width=width, height=height, bg="#f0f0f0", highlightthickness=0, **kwargs)
        self.key_width = width // 14
        self.white_keys = [0, 2, 4, 5, 7, 9, 11] # C, D, E, F, G, A, B
        self.black_keys = [1, 3, 6, 8, 10]       # C#, D#, F#, G#, A#
        self.key_ids = {}
        
        # 音源データのキャッシュ（ファイルパス）
        self.sound_files = {}
        # 一時ディレクトリの作成（アプリ終了時に自動削除されるように参照を保持）
        self.temp_dir = tempfile.TemporaryDirectory()
        self.preload_sounds()

        self.draw_keyboard()

    def preload_sounds(self):
        """12音階分のサイン波を生成し、一時ファイルとして保存する"""
        sr = 44100 # サンプリングレート
        duration = 0.4 # 音の長さ(秒)
        
        for i in range(12):
            midi_note = 60 + i 
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            
            # 波形生成
            t = np.linspace(0, duration, int(sr * duration), False)
            tone = np.sin(freq * t * 2 * np.pi)
            decay = np.exp(-4 * t)
            audio_data = (tone * decay * 32767).astype(np.int16)
            
            # 一時ファイルへのパスを作成
            file_path = os.path.join(self.temp_dir.name, f"note_{i}.wav")
            
            # WAVファイルとして書き出し
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sr)
                wav_file.writeframes(audio_data.tobytes())
            
            # パスを保存
            self.sound_files[i] = file_path

    def play_note(self, note_index):
        """指定されたインデックスの音をファイルから再生する"""
        if note_index in self.sound_files:
            # SND_FILENAME: ファイルから再生, SND_ASYNC: 非同期
            winsound.PlaySound(self.sound_files[note_index], winsound.SND_FILENAME | winsound.SND_ASYNC)

    def draw_keyboard(self):
        # 白鍵
        wk_index = 0
        for i in range(12):
            if i in self.white_keys:
                x = wk_index * self.key_width
                rect = self.create_rectangle(x, 0, x + self.key_width, 120, 
                                             fill="white", outline="black", tags=f"key_{i}")
                self.create_text(x + self.key_width/2, 100, text=NOTE_NAMES[i], fill="#aaa", tags=f"label_{i}")
                self.key_ids[i] = rect
                
                self.tag_bind(f"key_{i}", "<Button-1>", lambda e, n=i: self.play_note(n))
                self.tag_bind(f"label_{i}", "<Button-1>", lambda e, n=i: self.play_note(n))
                
                wk_index += 1

        # 黒鍵
        wk_index = 0
        for i in range(12):
            if i in self.white_keys:
                wk_index += 1
            elif i in self.black_keys:
                x = (wk_index * self.key_width) - (self.key_width * 0.3)
                rect = self.create_rectangle(x, 0, x + (self.key_width * 0.6), 75, 
                                             fill="black", outline="black", tags=f"key_{i}")
                self.key_ids[i] = rect
                
                self.tag_bind(f"key_{i}", "<Button-1>", lambda e, n=i: self.play_note(n))

    def highlight_keys(self, input_notes_set, scale_notes_set=None):
        scale_notes_set = scale_notes_set or set()
        for i in range(12):
            item_id = self.key_ids.get(i)
            if not item_id: continue

            default_color = "black" if i in self.black_keys else "white"
            
            is_input = i in input_notes_set
            is_scale = i in scale_notes_set

            if is_input and is_scale:
                self.itemconfig(item_id, fill="#32CD32") # LimeGreen
            elif is_input and not is_scale:
                self.itemconfig(item_id, fill="#FF6347") # Tomato
            elif not is_input and is_scale:
                self.itemconfig(item_id, fill="#87CEFA") # LightSkyBlue
            else:
                self.itemconfig(item_id, fill=default_color)

# ==========================================
# 3. メインアプリ
# ==========================================

class JazzScaleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jazz Scale Analyzer v2.3")
        self.root.geometry("800x650")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", font=("Meiryo UI", 10), rowheight=25)
        style.configure("Treeview.Heading", font=("Meiryo UI", 10, "bold"))

        self.all_scales_dict = generate_all_scales()
        self.current_input_notes = set()
        self.file_path = None

        # --- UI ---
        top_frame = ttk.Frame(root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="🎷 Jazz Phrasing Analyzer", font=("Meiryo UI", 14, "bold")).pack(side=tk.LEFT)
        
        btn_frame = ttk.Frame(top_frame)
        btn_frame.pack(side=tk.RIGHT)
        self.btn_select = ttk.Button(btn_frame, text="📂 ファイル選択", command=self.select_file)
        self.btn_select.pack(side=tk.LEFT, padx=5)
        self.btn_play = ttk.Button(btn_frame, text="▶ 再生", command=self.play_audio, state='disabled')
        self.btn_play.pack(side=tk.LEFT)

        # Keyboard Frame
        kbd_frame = ttk.LabelFrame(root, text="🎹 Visualizer (緑:一致 / 赤:スケール外 / 青:スケール音) ※クリックで試聴可", padding=10)
        kbd_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.keyboard = VirtualKeyboard(kbd_frame, width=760, height=120)
        self.keyboard.pack()

        # Result Frame
        result_frame = ttk.LabelFrame(root, text="📊 分析結果 (クリックしてスケールを確認)", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        columns = ("Rank", "Scale", "Match")
        self.tree = ttk.Treeview(result_frame, columns=columns, show="headings", selectmode="browse")
        
        self.tree.heading("Rank", text="順位")
        self.tree.heading("Scale", text="スケール名")
        self.tree.heading("Match", text="適合率")
        
        self.tree.column("Rank", width=50, anchor="center")
        self.tree.column("Scale", width=400, anchor="w")
        self.tree.column("Match", width=100, anchor="center")
        
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.tree.bind("<<TreeviewSelect>>", self.on_scale_selected)

        self.status_var = tk.StringVar(value="準備完了")
        self.lbl_status = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.file_path = file_path
            self.status_var.set(f"選択中: {os.path.basename(file_path)}")
            self.btn_play.config(state='normal')
            self.run_analysis()

    def play_audio(self):
        if self.file_path:
            winsound.PlaySound(self.file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def run_analysis(self):
        self.tree.delete(*self.tree.get_children())
        self.keyboard.highlight_keys(set())
        
        thread = threading.Thread(target=self._process_analysis)
        thread.start()

    def _process_analysis(self):
        self.status_var.set("分析中...")
        scales, note_names, note_indices = analyze_audio(self.file_path, lambda msg: self.status_var.set(msg))

        if scales is None:
            self.status_var.set(f"エラー: {note_names}")
            return

        self.current_input_notes = note_indices
        self.keyboard.highlight_keys(self.current_input_notes, self.current_input_notes)

        for i, (name, score) in enumerate(scales):
            if i >= 15 or score < 0.5: break
            self.tree.insert("", "end", values=(i+1, name, f"{score:.0%}"))

        self.status_var.set("分析完了。鍵盤をクリックすると音が鳴ります。")

    def on_scale_selected(self, event):
        selected_items = self.tree.selection()
        if not selected_items: return

        item = selected_items[0]
        scale_name = self.tree.item(item, "values")[1]
        scale_notes = self.all_scales_dict.get(scale_name, set())
        
        self.keyboard.highlight_keys(self.current_input_notes, scale_notes)

if __name__ == "__main__":
    root = tk.Tk()
    app = JazzScaleApp(root)
    root.mainloop()
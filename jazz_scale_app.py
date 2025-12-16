import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import librosa
import numpy as np
from collections import Counter
import threading
import os
import winsound
import wave
import tempfile

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
# 2. GUI用部品 (2 Octave Virtual Keyboard)
# ==========================================
class VirtualKeyboard(tk.Canvas):
    def __init__(self, master, width=760, height=120, **kwargs):
        super().__init__(master, width=width, height=height, bg="#f0f0f0", highlightthickness=0, **kwargs)
        
        self.num_octaves = 2
        self.total_keys = 12 * self.num_octaves # 24鍵盤
        
        # 白鍵の数を計算 (14個)
        num_white_keys = 7 * self.num_octaves
        self.key_width = width // num_white_keys
        
        self.white_key_indices = {0, 2, 4, 5, 7, 9, 11} 
        
        self.key_ids = {}
        self.sound_files = {}
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.preload_sounds()
        self.draw_keyboard()

    def preload_sounds(self):
        """2オクターブ分 (C3〜B4) の音声を生成"""
        sr = 44100
        duration = 0.4
        
        start_note = 48 # C3
        
        for i in range(self.total_keys):
            midi_note = start_note + i
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            
            t = np.linspace(0, duration, int(sr * duration), False)
            tone = np.sin(freq * t * 2 * np.pi)
            decay = np.exp(-4 * t)
            audio_data = (tone * decay * 32767).astype(np.int16)
            
            file_path = os.path.join(self.temp_dir.name, f"note_{i}.wav")
            with wave.open(file_path, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sr)
                wav_file.writeframes(audio_data.tobytes())
            
            self.sound_files[i] = file_path

    def play_note(self, note_index):
        if note_index in self.sound_files:
            winsound.PlaySound(self.sound_files[note_index], winsound.SND_FILENAME | winsound.SND_ASYNC)

    def draw_keyboard(self):
        wk_count = 0
        for i in range(self.total_keys):
            pitch_class = i % 12
            if pitch_class in self.white_key_indices:
                x = wk_count * self.key_width
                rect = self.create_rectangle(x, 0, x + self.key_width, 120, 
                                             fill="white", outline="black", tags=f"key_{i}")
                
                octave = 3 + (i // 12)
                note_name = NOTE_NAMES[pitch_class] + str(octave)
                
                self.create_text(x + self.key_width/2, 100, text=note_name, fill="#aaa", font=("Arial", 8), tags=f"label_{i}")
                self.key_ids[i] = rect
                
                self.tag_bind(f"key_{i}", "<Button-1>", lambda e, n=i: self.play_note(n))
                self.tag_bind(f"label_{i}", "<Button-1>", lambda e, n=i: self.play_note(n))
                wk_count += 1

        wk_count = 0
        for i in range(self.total_keys):
            pitch_class = i % 12
            if pitch_class in self.white_key_indices:
                wk_count += 1
            else: # 黒鍵
                x = (wk_count * self.key_width) - (self.key_width * 0.3)
                rect = self.create_rectangle(x, 0, x + (self.key_width * 0.6), 75, 
                                             fill="black", outline="black", tags=f"key_{i}")
                self.key_ids[i] = rect
                self.tag_bind(f"key_{i}", "<Button-1>", lambda e, n=i: self.play_note(n))

    def highlight_keys(self, input_notes_set, scale_notes_set=None):
        scale_notes_set = scale_notes_set or set()
        
        for i in range(self.total_keys):
            item_id = self.key_ids.get(i)
            if not item_id: continue

            pitch_class = i % 12
            default_color = "black" if pitch_class not in self.white_key_indices else "white"
            
            is_input = pitch_class in input_notes_set
            is_scale = pitch_class in scale_notes_set

            if is_input and is_scale:
                self.itemconfig(item_id, fill="#32CD32") # Green
            elif is_input and not is_scale:
                self.itemconfig(item_id, fill="#FF6347") # Red
            elif not is_input and is_scale:
                self.itemconfig(item_id, fill="#87CEFA") # Blue
            else:
                self.itemconfig(item_id, fill=default_color)

# ==========================================
# 3. メインアプリ
# ==========================================

class JazzScaleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jazz Scale Analyzer v2.5 (Root Filter)")
        self.root.geometry("820x700")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", font=("Meiryo UI", 10), rowheight=25)
        style.configure("Treeview.Heading", font=("Meiryo UI", 10, "bold"))

        self.all_scales_dict = generate_all_scales()
        self.current_input_notes = set()
        self.file_path = None

        # --- UI Header ---
        top_frame = ttk.Frame(root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="🎷 Jazz Phrasing Analyzer", font=("Meiryo UI", 14, "bold")).pack(side=tk.LEFT)
        
        # コントロールエリア（右側）
        ctrl_frame = ttk.Frame(top_frame)
        ctrl_frame.pack(side=tk.RIGHT)

        # 1. ルート音選択コンボボックス (NEW!)
        ttk.Label(ctrl_frame, text="ルート音指定:").pack(side=tk.LEFT, padx=(0, 5))
        
        self.root_var = tk.StringVar()
        self.cmb_root = ttk.Combobox(ctrl_frame, textvariable=self.root_var, state="readonly", width=8)
        self.cmb_root['values'] = ["指定なし"] + NOTE_NAMES
        self.cmb_root.current(0) # デフォルトは「指定なし」
        self.cmb_root.pack(side=tk.LEFT, padx=(0, 15))
        self.cmb_root.bind("<<ComboboxSelected>>", self.on_root_changed)

        # 2. ファイル選択 & 再生
        self.btn_select = ttk.Button(ctrl_frame, text="📂 ファイル選択", command=self.select_file)
        self.btn_select.pack(side=tk.LEFT, padx=5)
        
        self.btn_play = ttk.Button(ctrl_frame, text="▶ 再生", command=self.play_audio, state='disabled')
        self.btn_play.pack(side=tk.LEFT)

        # --- Keyboard ---
        kbd_frame = ttk.LabelFrame(root, text="🎹 Visualizer (2 Octaves: C3-B4)", padding=10)
        kbd_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.keyboard = VirtualKeyboard(kbd_frame, width=780, height=120)
        self.keyboard.pack()

        # --- Result ---
        result_frame = ttk.LabelFrame(root, text="📊 分析結果", padding=10)
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

        # --- Status ---
        self.status_var = tk.StringVar(value="準備完了")
        self.lbl_status = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

        # 分析結果の一時保存用 (再フィルタリング用)
        self.last_analysis_result = None # (scales, notes, indices)

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
        # 分析開始時はUIをクリア
        self.tree.delete(*self.tree.get_children())
        self.keyboard.highlight_keys(set())
        self.last_analysis_result = None
        
        thread = threading.Thread(target=self._process_analysis)
        thread.start()

    def _process_analysis(self):
        self.status_var.set("分析中...")
        result = analyze_audio(self.file_path, lambda msg: self.status_var.set(msg))
        
        scales, note_names, note_indices = result

        if scales is None:
            self.status_var.set(f"エラー: {note_names}")
            return

        # 結果を保持 (フィルタリング操作のため)
        self.last_analysis_result = result
        self.current_input_notes = note_indices
        
        # 画面更新 (メインスレッドで行うべきだが、TkinterはThreadセーフな部分もあるため簡易実装)
        self.update_result_list()

    def update_result_list(self):
        """現在のルート音設定に基づいてリストを更新する"""
        if not self.last_analysis_result:
            return

        scales, _, _ = self.last_analysis_result
        target_root = self.root_var.get()

        self.tree.delete(*self.tree.get_children())
        self.keyboard.highlight_keys(self.current_input_notes, self.current_input_notes) # 初期状態

        display_count = 0
        rank = 0
        last_score = -1

        for name, score in scales:
            # --- フィルタリング処理 ---
            # スケール名 (例: "C Major") からルート音 ("C") を取り出して比較
            scale_root = name.split()[0]
            
            if target_root != "指定なし" and scale_root != target_root:
                continue # ルート音が一致しない場合はスキップ
            
            # --- 表示処理 ---
            if display_count >= 20 or score < 0.5: break
            
            if score != last_score:
                rank = display_count + 1
            
            self.tree.insert("", "end", values=(rank, name, f"{score:.0%}"))
            last_score = score
            display_count += 1

        if display_count == 0:
            self.status_var.set(f"分析完了: ルート {target_root} に一致するスケールは見つかりませんでした。")
        else:
            self.status_var.set(f"分析完了: {display_count} 件のスケールを表示 (ルート: {target_root})")

    def on_root_changed(self, event):
        """ドロップダウンが変更されたらリストを更新"""
        if self.last_analysis_result:
            self.update_result_list()

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
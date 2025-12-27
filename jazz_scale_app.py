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
import time
import pyaudio

# ==========================================
# 1. 分析ロジック & 定数 (Backend)
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
    'Combination of Diminished': [0, 1, 3, 4, 6, 7, 9, 10],
    'Diminished (W-H)':   [0, 2, 3, 5, 6, 8, 9, 11],
    'Wholetone':          [0, 2, 4, 6, 8, 10],
    'Phrygian Dominant':  [0, 1, 4, 5, 7, 8, 10],
    'Lydian Dominant':    [0, 2, 4, 6, 7, 9, 10],
    'Major Pentatonic':   [0, 2, 4, 7, 9],
    'Minor Pentatonic':   [0, 3, 5, 7, 10],
    'Blues Scale':        [0, 3, 5, 6, 7, 10],
    'Bebop Dominant':     [0, 2, 4, 5, 7, 9, 10, 11],
    'Harmonic Minor':     [0, 2, 3, 5, 7, 8, 11],
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

INTERVAL_MAP = {
    0: "R", 1: "b9", 2: "9", 3: "b3", 4: "3", 5: "11",
    6: "#11/b5", 7: "5", 8: "b13", 9: "13", 10: "b7", 11: "7"
}

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
            y, fmin=librosa.note_to_hz('A1'), fmax=librosa.note_to_hz('C6')
        )
        
        confident_f0 = f0[voiced_probs > 0.5]
        confident_f0 = confident_f0[~np.isnan(confident_f0)]

        if len(confident_f0) == 0:
            return None, "有効な音程が検出できませんでした。", None

        # ここで MIDIノート番号（絶対値）を取得
        midi_notes = np.round(librosa.hz_to_midi(confident_f0)).astype(int)
        
        # 1. 絶対音高（MIDI番号）でカウントして、入力音を特定する
        midi_counts = Counter(midi_notes)
        total_notes = sum(midi_counts.values())
        min_count = total_notes * 0.02
        
        # 実際に検出されたMIDI番号のセット (例: {48, 52, 60})
        melody_midi_notes = set(
            [note for note, count in midi_counts.items() if count >= min_count]
        )
        
        # 2. スケール判定用に「音名(0-11)」のセットも作る
        melody_pitch_classes = set([n % 12 for n in melody_midi_notes])

        # 保険: 何も残らなかった場合
        if not melody_pitch_classes and total_notes > 0:
            top_common = midi_counts.most_common(5) # 上位5つを見る
            melody_midi_notes = set([n[0] for n in top_common])
            melody_pitch_classes = set([n % 12 for n in melody_midi_notes])

        detected_notes = sorted([NOTE_NAMES[n % 12] for n in melody_midi_notes])
        # 重複排除して表示用にする
        detected_notes = sorted(list(set(detected_notes)), key=lambda x: NOTE_NAMES.index(x) if x in NOTE_NAMES else 0)
        
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
        
        # 戻り値の3つ目を「MIDI番号のセット」に変更
        return sorted_scales, detected_notes, melody_midi_notes

    except Exception as e:
        return None, str(e), None

# ==========================================
# 2. GUI用部品 (2 Octave Virtual Keyboard)
# ==========================================
class VirtualKeyboard(tk.Canvas):
    def __init__(self, master, width=760, height=120, **kwargs):
        super().__init__(master, width=width, height=height, bg="#f0f0f0", highlightthickness=0, **kwargs)
        
        self.num_octaves = 2
        self.total_keys = 12 * self.num_octaves 
        
        num_white_keys = 7 * self.num_octaves
        self.key_width = width // num_white_keys
        
        self.white_key_indices = {0, 2, 4, 5, 7, 9, 11} 
        
        self.key_ids = {}
        self.sound_files = {}
        self.temp_dir = tempfile.TemporaryDirectory()
        
        self.preload_sounds()
        self.draw_keyboard()

    def preload_sounds(self):
        sr = 44100
        duration = 0.5 
        start_note = 48 # C3
        
        for i in range(self.total_keys):
            midi_note = start_note + i
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            
            t = np.linspace(0, duration, int(sr * duration), False)
            tone = np.sin(freq * t * 2 * np.pi)
            decay = np.exp(-5 * t)
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

    def play_sequence(self, indices):
        def _run():
            for idx in indices:
                if 0 <= idx < self.total_keys:
                    self.play_note(idx)
                    time.sleep(0.3) 
        threading.Thread(target=_run, daemon=True).start()

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
            else: 
                x = (wk_count * self.key_width) - (self.key_width * 0.3)
                rect = self.create_rectangle(x, 0, x + (self.key_width * 0.6), 75, 
                                             fill="black", outline="black", tags=f"key_{i}")
                self.key_ids[i] = rect
                self.tag_bind(f"key_{i}", "<Button-1>", lambda e, n=i: self.play_note(n))

    def highlight_keys(self, input_midi_set, scale_pc_set=None):
        """
        input_midi_set: 検出されたMIDI番号のセット (例: {48, 55}) -> 絶対的な高さ
        scale_pc_set: スケールの構成音 (0-11) -> 相対的な音名
        """
        scale_pc_set = scale_pc_set or set()
        start_note = 48 # C3
        
        for i in range(self.total_keys):
            item_id = self.key_ids.get(i)
            if not item_id: continue

            # この鍵盤の絶対MIDI番号と、音名クラス(0-11)を計算
            current_midi = start_note + i
            current_pc = current_midi % 12
            
            default_color = "black" if current_pc not in self.white_key_indices else "white"
            
            # 判定ロジックの変更点:
            # 入力音は「絶対値」で判定、スケール音は「音名」で判定
            is_input = current_midi in input_midi_set
            is_scale = current_pc in scale_pc_set

            if is_input and is_scale:
                self.itemconfig(item_id, fill="#32CD32") # Green (正解かつ弾いた音)
            elif is_input and not is_scale:
                self.itemconfig(item_id, fill="#FF6347") # Red (外した音)
            elif not is_input and is_scale:
                self.itemconfig(item_id, fill="#87CEFA") # Blue (スケールガイド)
            else:
                self.itemconfig(item_id, fill=default_color)

# ==========================================
# 3. メインアプリ
# ==========================================

class JazzScaleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jazz Scale Analyzer v2.11 (Octave Sensitive)")
        self.root.geometry("820x780")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Treeview", font=("Meiryo UI", 10), rowheight=25)
        style.configure("Treeview.Heading", font=("Meiryo UI", 10, "bold"))
        style.configure("Rec.TButton", foreground="red")

        self.all_scales_dict = generate_all_scales()
        
        # MIDI番号のセットを保持するように変更
        self.current_input_midi = set()
        self.file_path = None
        
        self.is_recording = False
        self.frames = []
        self.mic_device_index = 1 

        # --- Header ---
        top_frame = ttk.Frame(root, padding=10)
        top_frame.pack(fill=tk.X)
        
        ttk.Label(top_frame, text="🎷 Jazz Phrasing Analyzer", font=("Meiryo UI", 14, "bold")).pack(side=tk.LEFT)
        
        ctrl_frame = ttk.Frame(top_frame)
        ctrl_frame.pack(side=tk.RIGHT)

        ttk.Label(ctrl_frame, text="ルート:").pack(side=tk.LEFT, padx=(0, 2))
        self.root_var = tk.StringVar()
        self.cmb_root = ttk.Combobox(ctrl_frame, textvariable=self.root_var, state="readonly", width=5)
        self.cmb_root['values'] = ["指定なし"] + NOTE_NAMES
        self.cmb_root.current(0)
        self.cmb_root.pack(side=tk.LEFT, padx=(0, 10))
        self.cmb_root.bind("<<ComboboxSelected>>", self.on_root_changed)

        self.btn_rec_start = ttk.Button(ctrl_frame, text="🔴 録音開始", command=self.start_recording, style="Rec.TButton")
        self.btn_rec_start.pack(side=tk.LEFT, padx=2)
        
        self.btn_rec_stop = ttk.Button(ctrl_frame, text="⬛ 停止", command=self.stop_recording, state='disabled')
        self.btn_rec_stop.pack(side=tk.LEFT, padx=2)

        ttk.Separator(ctrl_frame, orient='vertical').pack(side=tk.LEFT, padx=10, fill='y')

        self.btn_select = ttk.Button(ctrl_frame, text="📂 開く", command=self.select_file, width=8)
        self.btn_select.pack(side=tk.LEFT, padx=2)
        
        self.btn_play_wav = ttk.Button(ctrl_frame, text="▶ 再生", command=self.play_audio, state='disabled', width=8)
        self.btn_play_wav.pack(side=tk.LEFT)

        # --- Keyboard ---
        kbd_frame = ttk.LabelFrame(root, text="🎹 Visualizer (C3-B4)", padding=10)
        kbd_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.keyboard = VirtualKeyboard(kbd_frame, width=780, height=120)
        self.keyboard.pack()

        # --- Degree Info Area ---
        degree_frame = ttk.LabelFrame(root, text="🎓 Degree Analysis", padding=10)
        degree_frame.pack(fill=tk.X, padx=10, pady=5)

        self.lbl_degree_info = ttk.Label(degree_frame, text="スケールを選択すると度数情報が表示されます", 
                                         font=("Meiryo UI", 11), foreground="#333")
        self.lbl_degree_info.pack(anchor="center")

        # --- Result ---
        result_frame = ttk.LabelFrame(root, text="📊 分析結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        btn_area = ttk.Frame(result_frame)
        btn_area.pack(fill=tk.X, pady=(0, 5))
        
        self.btn_preview_scale = ttk.Button(btn_area, text="🔊 スケール試聴", command=self.play_selected_scale, state='disabled')
        self.btn_preview_scale.pack(side=tk.RIGHT)
        
        ttk.Label(btn_area, text="リスト選択で詳細を表示").pack(side=tk.LEFT)

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

        self.last_analysis_result = None

    # --- Recording ---
    def start_recording(self):
        self.is_recording = True
        self.frames = []
        self.btn_rec_start.config(state='disabled')
        self.btn_rec_stop.config(state='normal')
        self.btn_select.config(state='disabled') 
        self.status_var.set("🔴 録音中... (マイクに向かって演奏してください)")
        threading.Thread(target=self._record_thread).start()

    def stop_recording(self):
        self.is_recording = False
        self.status_var.set("録音停止。保存中...")

    def _record_thread(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK, input_device_index=self.mic_device_index)
            
            while self.is_recording:
                data = stream.read(CHUNK)
                self.frames.append(data)
                
            stream.stop_stream()
            stream.close()
            p.terminate()

            filename = f"rec_{int(time.time())}.wav"
            save_path = os.path.abspath(filename)
            
            wf = wave.open(save_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            
            self.file_path = save_path
            self.btn_rec_start.config(state='normal')
            self.btn_rec_stop.config(state='disabled')
            self.btn_select.config(state='normal')
            self.btn_play_wav.config(state='normal')
            self.status_var.set(f"録音完了: {filename} を分析中...")
            self.run_analysis()
            
        except Exception as e:
            self.status_var.set(f"録音エラー: {e}")
            self.is_recording = False
            self.btn_rec_start.config(state='normal')
            self.btn_rec_stop.config(state='disabled')

    # --- Analysis & UI ---
    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.file_path = file_path
            self.status_var.set(f"選択中: {os.path.basename(file_path)}")
            self.btn_play_wav.config(state='normal')
            self.run_analysis()

    def play_audio(self):
        if self.file_path:
            winsound.PlaySound(self.file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def run_analysis(self):
        self.tree.delete(*self.tree.get_children())
        self.keyboard.highlight_keys(set())
        self.last_analysis_result = None
        self.btn_preview_scale.config(state='disabled')
        self.lbl_degree_info.config(text="スケールを選択すると度数情報が表示されます", foreground="#333")
        
        thread = threading.Thread(target=self._process_analysis)
        thread.start()

    def _process_analysis(self):
        result = analyze_audio(self.file_path, lambda msg: self.status_var.set(msg))
        
        # result: (scales, detected_note_names, melody_midi_notes)
        scales, note_names, midi_notes = result
        if scales is None:
            self.status_var.set(f"エラー: {note_names}")
            return

        self.last_analysis_result = result
        self.current_input_midi = midi_notes # ここがMIDI番号のセットになる
        self.update_result_list()

    def update_result_list(self):
        if not self.last_analysis_result: return
        scales, _, _ = self.last_analysis_result
        target_root = self.root_var.get()

        self.tree.delete(*self.tree.get_children())
        # MIDI番号を渡す
        self.keyboard.highlight_keys(self.current_input_midi, set()) 

        display_count = 0
        rank = 0
        last_score = -1

        for name, score in scales:
            scale_root = name.split()[0]
            if target_root != "指定なし" and scale_root != target_root:
                continue
            
            if display_count >= 20 or score < 0.5: break
            
            if score != last_score:
                rank = display_count + 1
            
            self.tree.insert("", "end", values=(rank, name, f"{score:.0%}"))
            last_score = score
            display_count += 1

        self.status_var.set(f"分析完了: {display_count} 件表示")

    def on_root_changed(self, event):
        if self.last_analysis_result:
            self.update_result_list()

    def on_scale_selected(self, event):
        selected_items = self.tree.selection()
        if not selected_items:
            self.btn_preview_scale.config(state='disabled')
            return

        self.btn_preview_scale.config(state='normal')
        
        item = selected_items[0]
        full_scale_name = self.tree.item(item, "values")[1] 
        scale_notes = self.all_scales_dict.get(full_scale_name, set())
        
        # MIDI番号のセット(入力) と スケール音名のセット(0-11) を渡す
        self.keyboard.highlight_keys(self.current_input_midi, scale_notes)
        self.update_degree_display(full_scale_name)

    def update_degree_display(self, full_scale_name):
        try:
            root_str = full_scale_name.split(' ')[0]
            root_idx = NOTE_NAMES.index(root_str)
            
            display_parts = []
            # MIDI番号をソートして処理
            sorted_midi_notes = sorted(list(self.current_input_midi))
            
            for midi_note in sorted_midi_notes:
                pitch_class = midi_note % 12
                note_name = NOTE_NAMES[pitch_class]
                # オクターブ表記を追加 (例: C3, D4)
                octave = (midi_note // 12) - 1 
                
                interval = (pitch_class - root_idx) % 12
                degree_name = INTERVAL_MAP.get(interval, "?")
                display_parts.append(f"{note_name}{octave}({degree_name})")
            
            result_text = f"【 {full_scale_name} 】上の度数:   " + "  -  ".join(display_parts)
            self.lbl_degree_info.config(text=result_text, foreground="#0055AA", font=("Meiryo UI", 12, "bold"))
            
        except Exception as e:
            print(f"Degree Calc Error: {e}")
            self.lbl_degree_info.config(text="度数情報の計算に失敗しました")

    def play_selected_scale(self):
        selected_items = self.tree.selection()
        if not selected_items: return

        item = selected_items[0]
        full_scale_name = self.tree.item(item, "values")[1]
        
        try:
            split_name = full_scale_name.split(' ', 1)
            root_str = split_name[0]
            pattern_name = split_name[1]
            
            pattern = SCALE_PATTERNS.get(pattern_name)
            if not pattern: return 

            root_midi = NOTE_NAMES.index(root_str)
            start_key_index = root_midi 
            
            sequence = []
            for interval in pattern:
                sequence.append(start_key_index + interval)
            sequence.append(start_key_index + 12)
            
            self.keyboard.play_sequence(sequence)

        except Exception as e:
            print(f"Play Error: {e}")

if __name__ == "__main__":
    root = tk.Tk()
    app = JazzScaleApp(root)
    root.mainloop()
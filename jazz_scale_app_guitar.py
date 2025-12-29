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
GUITAR_OPEN_STRINGS = [40, 45, 50, 55, 59, 64]

def generate_all_scales():
    all_scales = {}
    for root_midi in range(12):
        root_name = NOTE_NAMES[root_midi]
        for scale_name, pattern in SCALE_PATTERNS.items():
            scale_notes = set([(root_midi + interval) % 12 for interval in pattern])
            all_scales[f"{root_name} {scale_name}"] = scale_notes
    return all_scales

def analyze_audio(wav_path, progress_callback):
    try:
        progress_callback("音声読み込み中...")
        y, sr = librosa.load(wav_path, sr=None)
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('A1'), fmax=librosa.note_to_hz('C6'))
        confident_f0 = f0[voiced_probs > 0.5]
        confident_f0 = confident_f0[~np.isnan(confident_f0)]
        if len(confident_f0) == 0: return None, "検出エラー", None
        midi_notes = np.round(librosa.hz_to_midi(confident_f0)).astype(int)
        midi_counts = Counter(midi_notes)
        total_notes = sum(midi_counts.values())
        melody_midi_notes = set([note for note, count in midi_counts.items() if count >= total_notes * 0.02])
        if not melody_midi_notes and total_notes > 0:
            melody_midi_notes = set([n[0] for n in midi_counts.most_common(5)])
        melody_pitch_classes = set([n % 12 for n in melody_midi_notes])
        detected_notes = sorted(list(set([NOTE_NAMES[n%12] for n in melody_midi_notes])), key=lambda x: NOTE_NAMES.index(x))
        all_scales = generate_all_scales()
        scores = {}
        for name, notes in all_scales.items():
            match_count = len(melody_pitch_classes.intersection(notes))
            scores[name] = match_count / len(melody_pitch_classes) if len(melody_pitch_classes) > 0 else 0
        return sorted(scores.items(), key=lambda item: item[1], reverse=True), detected_notes, melody_midi_notes
    except Exception as e: return None, str(e), None

# ==========================================
# 2. GUI部品 (指板)
# ==========================================
class GuitarFretboard(tk.Canvas):
    def __init__(self, master, width=1050, height=220, **kwargs):
        super().__init__(master, width=width, height=height, bg="#2a2a2a", highlightthickness=0, **kwargs)
        self.num_frets = 12
        self.num_strings = 6
        self.open_strings = GUITAR_OPEN_STRINGS
        self.margin_left, self.margin_right, self.margin_top, self.margin_bottom = 60, 30, 40, 30
        self.fret_width = (width - self.margin_left - self.margin_right) / (self.num_frets + 1)
        self.string_height = (height - self.margin_top - self.margin_bottom) / (self.num_strings - 1)
        self.sound_files = {}
        self.temp_dir = tempfile.TemporaryDirectory()
        self.preload_sounds()
        self.drawn_items = []
        self.draw_board()

    def preload_sounds(self):
        sr = 44100
        for midi_note in range(40, 100):
            freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
            t = np.linspace(0, 0.5, int(sr * 0.5), False)
            tone = (np.sin(freq * t * 2 * np.pi) + 0.5 * np.sin(freq * 2 * t * 2 * np.pi)) * np.exp(-5 * t)
            audio_data = (tone * 32767 * 0.4).astype(np.int16)
            path = os.path.join(self.temp_dir.name, f"n_{midi_note}.wav")
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(audio_data.tobytes())
            self.sound_files[midi_note] = path

    def play_note(self, n):
        if n in self.sound_files: winsound.PlaySound(self.sound_files[n], winsound.SND_FILENAME | winsound.SND_ASYNC)

    def play_sequence(self, indices):
        def _run():
            for idx in indices:
                m = 48 + idx
                if m in self.sound_files: winsound.PlaySound(self.sound_files[m], winsound.SND_FILENAME | winsound.SND_ASYNC)
                time.sleep(0.3)
        threading.Thread(target=_run, daemon=True).start()

    def draw_board(self):
        self.delete("all")
        nut_x = self.margin_left
        self.create_rectangle(nut_x - 5, self.margin_top, nut_x, self.height() - self.margin_bottom, fill="#DDD")
        inlays = [3, 5, 7, 9, 12]
        for f in range(self.num_frets + 1):
            x = self.margin_left + (f * self.fret_width)
            if f > 0:
                self.create_line(x, self.margin_top, x, self.height() - self.margin_bottom, fill="#777", width=2)
                self.create_text(x - (self.fret_width/2), self.height() - 10, text=str(f), fill="#999")
            if f in inlays:
                cx, cy, r = self.margin_left + (f * self.fret_width) - (self.fret_width / 2), self.height() / 2, 6
                self.create_oval(cx-r, cy-r, cx+r, cy+r, fill="#444", outline="")
                if f == 12:
                    self.create_oval(cx-r, cy-r-20, cx+r, cy+r-20, fill="#444", outline="")
                    self.create_oval(cx-r, cy-r+20, cx+r, cy+r+20, fill="#444", outline="")
        for s in range(self.num_strings):
            y = self.margin_top + (s * self.string_height)
            self.create_line(self.margin_left, y, self.width() - self.margin_right, y, fill="#888", width=1+(s*0.5))
            self.create_text(20, y, text=f"{s+1}st", fill="#CCC", font=("Arial", 8))
            self.create_text(45, y, text=NOTE_NAMES[self.open_strings[5-s]%12], fill="#EEE", font=("Arial", 9, "bold"))

    def highlight_notes(self, input_midi_set, scale_pc_set=None, min_fret=0, max_fret=12):
        scale_pc_set = scale_pc_set or set()
        for item in self.drawn_items: self.delete(item)
        self.drawn_items = []
        for s_idx in range(self.num_strings): 
            open_midi = self.open_strings[5 - s_idx] 
            y = self.margin_top + (s_idx * self.string_height)
            for f in range(self.num_frets + 1):
                cur_midi, cur_pc = open_midi + f, (open_midi + f) % 12
                is_in, is_sc = (cur_midi in input_midi_set) and (min_fret <= f <= max_fret), cur_pc in scale_pc_set
                color = "#32CD32" if is_in and is_sc else "#FF6347" if is_in else "#87CEFA" if is_sc else None
                if color:
                    x = self.margin_left - 15 if f == 0 else self.margin_left + (f * self.fret_width) - (self.fret_width / 2)
                    m = self.create_oval(x-11, y-11, x+11, y+11, fill=color, outline="white")
                    t = self.create_text(x, y, text=NOTE_NAMES[cur_pc], fill="black", font=("Arial", 8, "bold"))
                    self.drawn_items.extend([m, t])
                    self.tag_bind(m, "<Button-1>", lambda e, n=cur_midi: self.play_note(n))
                    self.tag_bind(t, "<Button-1>", lambda e, n=cur_midi: self.play_note(n))
    def width(self): return int(self['width'])
    def height(self): return int(self['height'])

# ==========================================
# 3. メインアプリ
# ==========================================
class JazzGuitarApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jazz Guitar Analyzer (v2.20 - Output Folder Support)")
        self.root.geometry("1120x860")
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#f5f5f5")
        style.configure("Treeview", font=("Meiryo UI", 10), rowheight=28)
        style.configure("Header.TLabel", font=("Meiryo UI", 16, "bold"))
        style.configure("Record.TButton", font=("Meiryo UI", 10, "bold"), foreground="red")
        style.configure("Action.TButton", font=("Meiryo UI", 10))

        self.all_scales_dict = generate_all_scales()
        self.current_input_midi = set(); self.file_path = None; self.is_recording = False
        self.frames = []; self.mic_device_index = 1 

        # --- Top Toolbar ---
        toolbar = ttk.Frame(root, padding=10, relief="raised")
        toolbar.pack(fill=tk.X)
        
        ttk.Label(toolbar, text="🎸 Jazz Guitar Analyzer", style="Header.TLabel").pack(side=tk.LEFT, padx=(0, 20))

        # グループ1: 音声入力
        input_grp = ttk.LabelFrame(toolbar, text=" 1. 音声入力 ", padding=5)
        input_grp.pack(side=tk.LEFT, padx=5)
        ttk.Button(input_grp, text="📂 ファイルを開く", command=self.select_file, style="Action.TButton").pack(side=tk.LEFT, padx=2)
        self.btn_rec_start = ttk.Button(input_grp, text="🔴 録音開始", command=self.start_recording, style="Record.TButton")
        self.btn_rec_start.pack(side=tk.LEFT, padx=2)
        self.btn_rec_stop = ttk.Button(input_grp, text="⬛ 停止", command=self.stop_recording, state='disabled')
        self.btn_rec_stop.pack(side=tk.LEFT, padx=2)

        # グループ2: 設定
        setting_grp = ttk.LabelFrame(toolbar, text=" 2. 設定 ", padding=5)
        setting_grp.pack(side=tk.LEFT, padx=5)
        ttk.Label(setting_grp, text="Key:").pack(side=tk.LEFT)
        self.root_var = tk.StringVar()
        self.cmb_root = ttk.Combobox(setting_grp, textvariable=self.root_var, state="readonly", width=6, values=["指定なし"] + NOTE_NAMES)
        self.cmb_root.current(0); self.cmb_root.pack(side=tk.LEFT, padx=5)
        self.cmb_root.bind("<<ComboboxSelected>>", self.on_root_changed)

        # グループ3: 再生
        play_grp = ttk.LabelFrame(toolbar, text=" 3. 確認 ", padding=5)
        play_grp.pack(side=tk.LEFT, padx=5)
        self.btn_play_wav = ttk.Button(play_grp, text="▶ 録音を再生", command=self.play_audio, state='disabled')
        self.btn_play_wav.pack(side=tk.LEFT, padx=2)

        # --- Fretboard Setting Area ---
        fret_ctrl = ttk.Frame(root, padding=(10, 10, 10, 0))
        fret_ctrl.pack(fill=tk.X)
        inner_fret_ctrl = ttk.LabelFrame(fret_ctrl, text=" 🎸 指板の表示設定 ", padding=10)
        inner_fret_ctrl.pack(fill=tk.X)
        ttk.Label(inner_fret_ctrl, text="入力メロディを表示する範囲を指定:   Start").pack(side=tk.LEFT)
        self.min_fret_var = tk.IntVar(value=0); self.max_fret_var = tk.IntVar(value=12)
        c1 = ttk.Combobox(inner_fret_ctrl, textvariable=self.min_fret_var, values=list(range(13)), width=4, state="readonly")
        c1.pack(side=tk.LEFT, padx=5); c1.bind("<<ComboboxSelected>>", self.on_range_changed)
        ttk.Label(inner_fret_ctrl, text="～ End").pack(side=tk.LEFT)
        c2 = ttk.Combobox(inner_fret_ctrl, textvariable=self.max_fret_var, values=list(range(13)), width=4, state="readonly")
        c2.pack(side=tk.LEFT, padx=5); c2.bind("<<ComboboxSelected>>", self.on_range_changed)
        ttk.Label(inner_fret_ctrl, text="Fret", foreground="#777").pack(side=tk.LEFT)

        # --- Fretboard Canvas ---
        kbd_frame = ttk.Frame(root, padding=10)
        kbd_frame.pack(fill=tk.X)
        self.fretboard = GuitarFretboard(kbd_frame, width=1080, height=220)
        self.fretboard.pack()

        # --- Degree Info ---
        deg_frame = ttk.Frame(root, padding=(10, 0))
        deg_frame.pack(fill=tk.X)
        self.lbl_degree_info = ttk.Label(deg_frame, text="リストからスケールを選択すると度数分析が表示されます", font=("Meiryo UI", 11), foreground="#555", relief="groove", padding=10, anchor="center")
        self.lbl_degree_info.pack(fill=tk.X)

        # --- Result Table ---
        res_frame = ttk.LabelFrame(root, text=" 📊 分析結果（適合率の高い順） ", padding=10)
        res_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        btn_area = ttk.Frame(res_frame)
        btn_area.pack(fill=tk.X, pady=(0, 5))
        self.btn_preview_scale = ttk.Button(btn_area, text="🔊 選択したスケールを試聴する", command=self.play_selected_scale, state='disabled')
        self.btn_preview_scale.pack(side=tk.RIGHT)
        self.tree = ttk.Treeview(res_frame, columns=("Rank", "Scale", "Match"), show="headings", selectmode="browse")
        self.tree.heading("Rank", text="順位"); self.tree.heading("Scale", text="推奨スケール名"); self.tree.heading("Match", text="適合率")
        self.tree.column("Rank", width=60, anchor="center"); self.tree.column("Scale", width=450, anchor="w"); self.tree.column("Match", width=120, anchor="center")
        sb = ttk.Scrollbar(res_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=sb.set)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True); sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self.on_scale_selected)

        self.status_var = tk.StringVar(value="準備完了：WAVファイルを開くか、録音ボタンを押してください。")
        ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5).pack(side=tk.BOTTOM, fill=tk.X)
        self.last_analysis_result = None

    # --- Functions ---
    def start_recording(self):
        self.is_recording, self.frames = True, []
        self.btn_rec_start.config(state='disabled'); self.btn_rec_stop.config(state='normal')
        self.status_var.set("🔴 録音中... マイクに向かってフレーズを弾いてください。")
        threading.Thread(target=self._record_thread).start()

    def stop_recording(self): self.is_recording = False

    def _record_thread(self):
        try:
            # ★ outputフォルダの作成
            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024, input_device_index=self.mic_device_index)
            while self.is_recording: self.frames.append(stream.read(1024))
            stream.stop_stream(); stream.close(); p.terminate()
            
            # ★ パスをoutputフォルダ内に設定
            filename = f"rec_{int(time.time())}.wav"
            path = os.path.abspath(os.path.join(output_dir, filename))
            
            with wave.open(path, 'wb') as wf:
                wf.setnchannels(1); wf.setsampwidth(p.get_sample_size(pyaudio.paInt16)); wf.setframerate(44100); wf.writeframes(b''.join(self.frames))
            
            self.file_path = path; self.btn_rec_start.config(state='normal'); self.btn_rec_stop.config(state='disabled')
            self.btn_play_wav.config(state='normal'); self.run_analysis()
        except Exception as e: self.status_var.set(f"録音エラー: {e}")

    def select_file(self):
        p = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if p: self.file_path = p; self.btn_play_wav.config(state='normal'); self.run_analysis()

    def play_audio(self): winsound.PlaySound(self.file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def run_analysis(self):
        self.tree.delete(*self.tree.get_children()); self.fretboard.highlight_notes(set())
        threading.Thread(target=self._process_analysis).start()

    def _process_analysis(self):
        self.status_var.set("⏳ 分析中...")
        res = analyze_audio(self.file_path, lambda m: self.status_var.set(m))
        if res[0]:
            self.last_analysis_result, self.current_input_midi = res, res[2]; self.update_result_list()
        else: self.status_var.set(f"❌ エラー: {res[1]}")

    def update_result_list(self):
        if not self.last_analysis_result: return
        scales, target_root = self.last_analysis_result[0], self.root_var.get()
        self.tree.delete(*self.tree.get_children())
        self.fretboard.highlight_notes(self.current_input_midi, set(), self.min_fret_var.get(), self.max_fret_var.get())
        display_count = 0
        for name, score in scales:
            if target_root != "指定なし" and not name.startswith(target_root): continue
            if display_count >= 20 or score < 0.5: break
            self.tree.insert("", "end", values=(display_count+1, name, f"{score:.0%}"))
            display_count += 1
        self.status_var.set(f"✅ 分析完了: {display_count} 件のスケールが候補に挙がりました。")

    def on_root_changed(self, event): self.update_result_list()
    def on_range_changed(self, event):
        if not self.last_analysis_result: return
        scale_notes = set()
        sel = self.tree.selection()
        if sel: scale_notes = self.all_scales_dict.get(self.tree.item(sel[0], "values")[1], set())
        self.fretboard.highlight_notes(self.current_input_midi, scale_notes, self.min_fret_var.get(), self.max_fret_var.get())

    def on_scale_selected(self, event):
        sel = self.tree.selection()
        if not sel: self.btn_preview_scale.config(state='disabled'); return
        self.btn_preview_scale.config(state='normal')
        name = self.tree.item(sel[0], "values")[1]
        self.fretboard.highlight_notes(self.current_input_midi, self.all_scales_dict.get(name, set()), self.min_fret_var.get(), self.max_fret_var.get())
        self.update_degree_display(name)

    def update_degree_display(self, name):
        try:
            root_idx = NOTE_NAMES.index(name.split(' ')[0])
            parts = [f"{NOTE_NAMES[m%12]}{(m//12)-1}({INTERVAL_MAP.get((m%12-root_idx)%12,'?')})" for m in sorted(list(self.current_input_midi))]
            self.lbl_degree_info.config(text=f"【 {name} 】上の役割: " + " - ".join(parts), foreground="#0055AA", font=("Meiryo UI", 12, "bold"))
        except: pass

    def play_selected_scale(self):
        sel = self.tree.selection()
        if not sel: return
        pat = SCALE_PATTERNS.get(self.tree.item(sel[0], "values")[1].split(' ', 1)[1])
        if pat: self.fretboard.play_sequence(pat + [12])

if __name__ == "__main__":
    app_root = tk.Tk(); app = JazzGuitarApp(app_root); app_root.mainloop()
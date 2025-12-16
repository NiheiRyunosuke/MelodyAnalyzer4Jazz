import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox
from tkinter import ttk  # モダンなウィジェット用
import librosa
import numpy as np
from collections import Counter
import threading
import os
import winsound  # Windows標準の音声再生用

# ==========================================
# 1. 分析ロジック (Backend) - 変更なし
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
            return None, "有効な音程が検出できませんでした。"

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
        return sorted_scales, detected_notes

    except Exception as e:
        return None, str(e)

# ==========================================
# 2. アプリ画面 (Frontend / GUI) - 大幅改良
# ==========================================

class JazzScaleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Jazz Scale Detector")
        self.root.geometry("700x600")
        
        # スタイル設定
        style = ttk.Style()
        style.theme_use('clam') # Windowsっぽいきれいなテーマ
        style.configure("TButton", font=("Meiryo UI", 10), padding=6)
        style.configure("TLabel", font=("Meiryo UI", 10))
        style.configure("Header.TLabel", font=("Meiryo UI", 14, "bold"), foreground="#333333")

        # --- メインコンテナ ---
        main_frame = ttk.Frame(root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 1. ヘッダー
        header = ttk.Label(main_frame, text="🎷 Jazz Phrasing Analyzer", style="Header.TLabel")
        header.pack(pady=(0, 20))

        # 2. コントロールエリア（ファイル選択・再生）
        control_frame = ttk.LabelFrame(main_frame, text="入力ファイル", padding=15)
        control_frame.pack(fill=tk.X, pady=(0, 15))

        # ボタン配置用のサブフレーム
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        self.btn_select = ttk.Button(btn_frame, text="📂 WAVファイルを開く", command=self.select_file)
        self.btn_select.pack(side=tk.LEFT, padx=(0, 10))

        self.btn_play = ttk.Button(btn_frame, text="▶ 再生", command=self.play_audio, state='disabled')
        self.btn_play.pack(side=tk.LEFT)

        self.lbl_filename = ttk.Label(control_frame, text="ファイルが選択されていません", foreground="#666666")
        self.lbl_filename.pack(anchor=tk.W, pady=(5, 0))

        # 3. 結果表示エリア
        result_frame = ttk.LabelFrame(main_frame, text="分析結果", padding=10)
        result_frame.pack(fill=tk.BOTH, expand=True)

        self.txt_result = scrolledtext.ScrolledText(result_frame, font=("Consolas", 11), state='disabled', height=15)
        self.txt_result.pack(fill=tk.BOTH, expand=True)

        # テキストの装飾タグ設定
        self.txt_result.tag_config("header", font=("Meiryo UI", 11, "bold"), foreground="#000080")
        self.txt_result.tag_config("notes", font=("Meiryo UI", 12), foreground="#006400")
        self.txt_result.tag_config("top_rank", font=("Meiryo UI", 14, "bold"), foreground="#D00000", background="#FFF0F0")
        self.txt_result.tag_config("normal_rank", font=("Meiryo UI", 11))

        # 4. ステータスバー
        self.status_var = tk.StringVar()
        self.status_var.set("準備完了")
        self.lbl_status = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, padding=5)
        self.lbl_status.pack(side=tk.BOTTOM, fill=tk.X)

        self.file_path = None

    def select_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("WAV files", "*.wav")])
        if file_path:
            self.file_path = file_path
            self.lbl_filename.config(text=f"選択中: {os.path.basename(file_path)}")
            self.btn_play.config(state='normal') # 再生ボタンを有効化
            self.run_analysis()

    def play_audio(self):
        if self.file_path:
            # Windows標準機能で非同期再生（SND_ASYNC）
            winsound.PlaySound(self.file_path, winsound.SND_FILENAME | winsound.SND_ASYNC)

    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()

    def run_analysis(self):
        thread = threading.Thread(target=self._process_analysis)
        thread.start()

    def _process_analysis(self):
        self.txt_result.config(state='normal')
        self.txt_result.delete(1.0, tk.END)
        self.txt_result.insert(tk.END, "分析中...\n")
        self.txt_result.config(state='disabled')

        scales, notes = analyze_audio(self.file_path, self.update_status)

        self.txt_result.config(state='normal')
        self.txt_result.delete(1.0, tk.END) # クリア

        if scales is None:
            self.txt_result.insert(tk.END, f"\n⚠ エラーが発生しました:\n{notes}\n")
        else:
            # 構成音の表示
            self.txt_result.insert(tk.END, "🎹 検出されたメロディーの構成音\n", "header")
            self.txt_result.insert(tk.END, f"   {', '.join(notes)}\n\n", "notes")
            
            self.txt_result.insert(tk.END, "📊 推定されるスケール (可能性の高い順)\n", "header")
            self.txt_result.insert(tk.END, "-"*40 + "\n")
            
            last_score = -1
            rank = 0
            
            for i, (name, score) in enumerate(scales):
                if i >= 10 and score < last_score: break
                if score < 0.5: break

                if score != last_score:
                    rank = i + 1
                
                # 表示テキストの作成
                text = f"{rank}. {name:<30} | 適合率: {score:.0%}\n"
                
                # 1位だけ大きく赤字で表示、それ以外は普通に表示
                if rank == 1:
                    self.txt_result.insert(tk.END, text, "top_rank")
                else:
                    self.txt_result.insert(tk.END, text, "normal_rank")
                
                last_score = score
            
        self.txt_result.config(state='disabled')
        self.update_status("分析完了")

if __name__ == "__main__":
    root = tk.Tk()
    app = JazzScaleApp(root)
    root.mainloop()
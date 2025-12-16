import librosa
import numpy as np
from collections import Counter

wav_path="nazo.wav"

# --- 1. スケールの定義 (新しいスケールを追加) ---
# 各スケールをルート音からの半音の数で定義します。
SCALE_PATTERNS = {
    # --- 基本的なスケール ---
    'Major': [0, 2, 4, 5, 7, 9, 11],
    'Natural Minor': [0, 2, 3, 5, 7, 8, 10],
    'Harmonic Minor': [0, 2, 3, 5, 7, 8, 11],
    'Melodic Minor': [0, 2, 3, 5, 7, 9, 11],
    'Major Pentatonic': [0, 2, 4, 7, 9],
    'Minor Pentatonic': [0, 3, 5, 7, 10],

    # --- 教会旋法 (Church Modes) ---
    'Ionian (Major)':     [0, 2, 4, 5, 7, 9, 11],
    'Dorian':             [0, 2, 3, 5, 7, 9, 10],
    'Phrygian':           [0, 1, 3, 5, 7, 8, 10],
    'Lydian':             [0, 2, 4, 6, 7, 9, 11],
    'Mixo-lydian':        [0, 2, 4, 5, 7, 9, 10],
    'Aeolian (Nat.Minor)':[0, 2, 3, 5, 7, 8, 10],
    'Locrian':            [0, 1, 3, 5, 6, 8, 10],

    # --- モダン/ジャズ/その他 --- 
    'Chromatic':          [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Altered (Super Locrian)': [0, 1, 3, 4, 6, 8, 10],
    'Diminished (H-W)':   [0, 1, 3, 4, 6, 7, 9, 10], # Half-Whole
    'Diminished (W-H)':   [0, 2, 3, 5, 6, 8, 9, 11], # Whole-Half
    'Wholetone':          [0, 2, 4, 6, 8, 10],
    'Phrygian Dominant (HMP5b)': [0, 1, 4, 5, 7, 8, 10], # Harmonic Minor P5 Below
}

# MIDIノート番号に対応するノート名
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def generate_all_scales():
    """12個のキー（C, C#, ...）それぞれについて、全てのスケールを生成します。"""
    all_scales = {}
    for root_midi in range(12):
        root_name = NOTE_NAMES[root_midi]
        for scale_name, pattern in SCALE_PATTERNS.items():
            scale_notes = set([(root_midi + interval) % 12 for interval in pattern])
            full_scale_name = f"{root_name} {scale_name}"
            all_scales[full_scale_name] = scale_notes
    return all_scales

# --- 2. 音声処理 ---
def extract_pitch_classes_from_wav(wav_path, confidence_threshold=0.5):
    """WAVファイルからメロディーを構成する主要な音（ピッチクラス）を抽出します。"""
    try:
        y, sr = librosa.load(wav_path, sr=None)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C6')
        )
        confident_f0 = f0[voiced_probs > confidence_threshold]
        confident_f0 = confident_f0[~np.isnan(confident_f0)]

        if len(confident_f0) == 0:
            print("⚠️ 音声から有効なピッチを抽出できませんでした。")
            return set()

        midi_notes = np.round(librosa.hz_to_midi(confident_f0)).astype(int)
        pitch_classes = [note % 12 for note in midi_notes]

        if not pitch_classes:
            return set()
        
        note_counts = Counter(pitch_classes)
        min_count = note_counts.most_common(1)[0][1] * 0.2
        melody_pitch_classes = set(
            [note for note, count in note_counts.items() if count >= min_count]
        )
        return melody_pitch_classes

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return set()

# --- 3. スケール判定 ---
def find_matching_scales(melody_pitch_classes, all_scales):
    """メロディーの音セットに最も適合するスケールを見つけます。"""
    if not melody_pitch_classes:
        return []

    scores = {}
    for scale_name, scale_notes in all_scales.items():
        match_count = len(melody_pitch_classes.intersection(scale_notes))
        
        # 適合率を「(一致した音の数) / (メロディーの全音数)」で計算
        if len(melody_pitch_classes) > 0:
            score = match_count / len(melody_pitch_classes)
        else:
            score = 0
        
        # クロマチックスケールは常に100%になるので、少しだけスコアを下げる
        if 'Chromatic' in scale_name:
            score *= 0.99 

        scores[scale_name] = score

    sorted_scales = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return sorted_scales

# --- 4. メイン実行関数 ---
def analyze_melody_scale(wav_path, top_n=5):
    """
    WAVファイルを分析し、メロディーが含まれる可能性の高いスケールTop Nを表示します。
    """
    print(f"🎵 '{wav_path}' のメロディーを分析中...")
    all_scales = generate_all_scales()
    melody_notes = extract_pitch_classes_from_wav(wav_path)
    
    if not melody_notes:
        print("分析を終了します。")
        return

    melody_note_names = sorted([NOTE_NAMES[pc] for pc in melody_notes])
    print(f"🎶 抽出されたメロディーの構成音: {', '.join(melody_note_names)}")
    print("-" * 40)

    matching_scales = find_matching_scales(melody_notes, all_scales)

    print("【スケール判定結果】")
    if not matching_scales:
        print("適合するスケールが見つかりませんでした。")
        return
        
    print(f"📈 可能性の高いスケール Top {top_n}:")
    last_score = -1
    rank = 0
    displayed_count = 0
    for i, (scale_name, score) in enumerate(matching_scales):
        # top_n位以降で、かつスコアが前の順位より低い場合は表示を打ち切る
        if displayed_count >= top_n and score < last_score:
            break
        # 適合率が0%のものは表示しない
        if score <= 0:
            break

        # スコアが変わった時だけ順位を更新する
        if score != last_score:
            rank = displayed_count + 1
        
        print(f"{rank}. {scale_name:<35} | 適合率: {score:.0%}")
        last_score = score
        displayed_count +=1

analyze_melody_scale(wav_path)

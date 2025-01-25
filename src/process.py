import os
import json
import librosa
import numpy as np
import chardet
import subprocess
import pathlib
import argparse

FFMPEG_PATH = "/usr/bin/ffmpeg"  # 确保此路径正确

def find_tja_and_audio(root_folder):
    """Find all TJA files and corresponding audio files"""
    dataset = []
    file_index = 1  # 用于重命名问题文件

    for subdir, _, files in os.walk(root_folder):
        tja_file = None
        audio_file = None

        for file in files:
            file_path = os.path.join(subdir, file)
            if file.endswith(".tja"):
                tja_file = file_path
            elif file.endswith((".mp3", ".ogg", ".wav")):
                audio_file = file_path

        if tja_file and audio_file:
            valid_tja_file = rename_if_needed(tja_file, file_index)
            valid_audio_file = rename_if_needed(audio_file, file_index)
            dataset.append({"tja": valid_tja_file, "audio": valid_audio_file})
            file_index += 1
    return dataset

def rename_if_needed(file_path, index):
    """如果文件名包含乱码，则重命名"""
    try:
        file_path.encode('utf-8')  # 检查是否为有效 UTF-8 文件名
        return file_path
    except UnicodeEncodeError:
        new_name = f"renamed_file_{index}{os.path.splitext(file_path)[1]}"
        new_path = os.path.join(os.path.dirname(file_path), new_name)
        os.rename(file_path, new_path)
        print(f"Renamed {file_path} to {new_path}")
        return new_path

def detect_encoding(file_path):
    """检测文件编码"""
    with open(file_path, 'rb') as f:
        raw_data = f.read()
    return chardet.detect(raw_data)['encoding']

def parse_tja(file_path):
    """解析 TJA 文件"""
    encoding = detect_encoding(file_path)
    try:
        with open(file_path, 'r', encoding=encoding) as f:
            content = f.readlines()
    except UnicodeDecodeError:
        print(f"Encoding issue with {file_path}, trying UTF-8 fallback.")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.readlines()

    metadata = {
        "title": None,
        "bpm": None,
        "offset": None,
        "audio": None,
        "courses": {}
    }

    current_course = None
    notes = []
    in_chart = False

    for line in content:
        line = line.strip()

        if line.startswith("TITLE:"):
            metadata["title"] = line.split(":", 1)[1].strip()
        elif line.startswith("BPM:"):
            metadata["bpm"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("OFFSET:"):
            metadata["offset"] = float(line.split(":", 1)[1].strip())
        elif line.startswith("WAVE:"):
            metadata["audio"] = line.split(":", 1)[1].strip()
        elif line.startswith("COURSE:"):
            current_course = line.split(":", 1)[1].strip()
            if current_course.lower() == "edit":
                print(f"Ignoring Edit mode: {file_path}")
                current_course = None
            else:
                metadata["courses"][current_course] = {"level": None, "balloon": [], "notes": []}
        elif line.startswith("LEVEL:") and current_course:
            try:
                metadata["courses"][current_course]["level"] = int(line.split(":", 1)[1].strip())
            except ValueError:
                print(f"Invalid LEVEL value: {line}")
        elif line.startswith("BALLOON:") and current_course:
            balloon_values = line.split(":", 1)[1].strip()
            metadata["courses"][current_course]["balloon"] = [
                int(x) for x in balloon_values.split(",") if x.strip().isdigit()
            ] if balloon_values else []
        elif line.startswith("#START"):
            in_chart = True
            notes = []
        elif line.startswith("#END"):
            in_chart = False
            if current_course:
                metadata["courses"][current_course]["notes"] = notes
        elif in_chart and line:
            notes.append(line.replace(",", "").strip())

    return metadata

def normalize_path(file_path):
    """规范化文件路径"""
    return os.path.abspath(str(pathlib.Path(file_path).resolve()))

def convert_audio_to_wav(audio_path):
    """使用 FFmpeg 将音频转换为 WAV"""
    audio_path = normalize_path(audio_path)
    wav_path = audio_path.rsplit(".", 1)[0] + ".wav"

    if os.path.exists(wav_path):
        print(f"WAV file already exists: {wav_path}")
        return wav_path

    try:
        subprocess.run([FFMPEG_PATH, "-i", audio_path, "-acodec", "pcm_s16le", "-ar", "22050", wav_path], check=True)
        return wav_path
    except subprocess.CalledProcessError:
        print(f"FFmpeg conversion failed: {audio_path}")
        return None

def extract_audio_features(audio_path, offset):
    """提取 Mel 频谱特征"""
    audio_path = normalize_path(audio_path)

    if audio_path.endswith((".mp3", ".ogg")):
        audio_path = convert_audio_to_wav(audio_path)
        if audio_path is None:
            return None

    try:
        y, sr = librosa.load(audio_path, sr=22050, offset=offset)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        return mel.mean(axis=1).tolist()
    except Exception as e:
        print(f"Audio processing error: {audio_path}, Error: {e}")
        return None

def prepare_dataset(input_folder, output_file):
    """处理所有 TJA 文件并保存为 JSONL"""
    dataset = []
    file_pairs = find_tja_and_audio(input_folder)

    for pair in file_pairs:
        tja_file, audio_file = pair["tja"], pair["audio"]
        metadata = parse_tja(tja_file)
        audio_features = extract_audio_features(audio_file, metadata.get("offset", 0))

        for difficulty, data in metadata["courses"].items():
            dataset.append({
                "title": metadata["title"],
                "bpm": metadata["bpm"],
                "offset": metadata.get("offset", 0),
                "level": data["level"],
                "course": difficulty,
                "audio_features": audio_features,
                "notes": " ".join(data["notes"])
            })

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input dataset path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    print(f"Processing dataset in {args.input}...")
    prepare_dataset(args.input, args.output)
    print(f"Dataset saved to {args.output}")

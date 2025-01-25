import torch
import librosa
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import os

# ✅ 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 加载 LLaMA2-13B
MODEL_PATH = "model/llama2_13B_taiko"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16, device_map="auto").to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ✅ 设置 Pad Token
tokenizer.pad_token = tokenizer.eos_token

def extract_bpm(audio_path):
    """ 使用 librosa 自动检测 BPM """
    y, sr = librosa.load(audio_path, sr=22050)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    return int(tempo) if tempo else 143  # 默认 BPM 143

def clean_tja_output(output_text):
    """ 过滤 AI 生成的无关内容，确保只保留 TJA 格式数据 """
    lines = output_text.split("\n")
    clean_lines = []

    for line in lines:
        line = line.strip()

        if "Notes:" in line or "0: Empty beat" in line:
            continue  # 跳过 AI 可能写的说明

        if line.startswith("Generate a Taiko no Tatsujin"):
            continue  # 跳过 prompt 复述

        clean_lines.append(line)

    return "\n".join(clean_lines)

def generate_tja(audio_path, output_dir="output", offset=0):
    """ 🎵 生成完整的 TJA 谱面 """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    bpm = extract_bpm(audio_path)
    song_name = os.path.basename(audio_path).replace(".mp3", "").replace(".wav", "")
    tja_filename = os.path.join(output_dir, f"{song_name}.tja")

    difficulties = {"Easy": 3, "Normal": 5, "Hard": 7, "Oni": 9}
    tja_content = ""

    for course, level in tqdm(difficulties.items(), desc="🎵 生成谱面中..."):
        print(f"🎵 生成 {course} ({level}) 谱面中...")

        prompt = f"""
Generate a Taiko no Tatsujin drum chart in TJA format.
Output ONLY the TJA data, do not explain anything.

TITLE: {song_name}
BPM: {bpm}
OFFSET: {offset}
WAVE: {song_name}.mp3

COURSE: {course}
LEVEL: {level}
#START
"""

        # ✅ 发送到 LLM
        input_data = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, padding="max_length").to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids=input_data["input_ids"],
                attention_mask=input_data["attention_mask"],
                max_new_tokens=4000,  # ✅ 增加最大 Token 限制
                temperature=0.7,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id
            )

        # ✅ 解析 LLM 生成的内容
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        clean_text = clean_tja_output(generated_text)

        # ✅ 确保谱面不为空
        if len(clean_text.strip()) == 0:
            print(f"⚠️ 生成的 {course} 谱面为空，使用占位符")
            clean_text = "100000000,010000000,001000000,000100000,000010000,000001000,"

        # ✅ 添加到最终 TJA
        tja_content += f"COURSE:{course}\nLEVEL:{level}\n#START\n{clean_text}\n#END\n\n"

    # ✅ 保存 TJA 文件
    with open(tja_filename, "w", encoding="utf-8") as f:
        f.write(tja_content)

    print(f"✅ TJA 谱面已生成：{tja_filename}")
    return tja_filename

# **测试**
if __name__ == "__main__":
    audio_file = "audio/ReadyNow.mp3"
    generate_tja(audio_file)

import os
from pydub import AudioSegment

# 定义路径
input_folder = r"D:\PyCharm\complex_STFT_SST\watkins"
output_folder = r"D:\PyCharm\complex_STFT_SST\watkins_0.5s"
species_folders = [
    "Spinner_Dolphin", "Frasers_Dolphin", "Striped_Dolphin",
    "Sperm_Whale", "Long-Finned_Pilot_Whale", "Grampus_Rissos_Dolphin"
]
window_length_ms = 500  # 500ms窗长
overlap_ms = 250  # 50%重叠 (500ms窗长的50%)

# 创建输出目录（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# 切分音频函数
def split_audio(file_path, species_name):
    # 读取音频文件
    audio = AudioSegment.from_wav(file_path)

    # 截取前1秒的音频
    if len(audio) > 1000:  # 如果音频长度超过1秒，截取前1秒
        audio = audio[:1000]

    # 获取文件名
    file_name = os.path.basename(file_path)

    # 创建物种对应的输出文件夹
    species_output_folder = os.path.join(output_folder, species_name)
    if not os.path.exists(species_output_folder):
        os.makedirs(species_output_folder)

    # 计算切分后的样本数量
    audio_length = len(audio)
    step = window_length_ms - overlap_ms
    num_chunks = (audio_length - window_length_ms) // step + 1

    # 如果音频少于窗长时，进行调整
    if audio_length < window_length_ms:
        num_chunks = 1  # 只会得到一个完整的片段

    for i in range(num_chunks):
        start_ms = i * step
        end_ms = start_ms + window_length_ms
        if end_ms > audio_length:  # 如果超出音频有效长度，结束切分
            break
        chunk = audio[start_ms:end_ms]

        # 构建输出文件路径
        output_file = os.path.join(species_output_folder, f"{file_name[:-4]}_chunk_{i + 1}.wav")

        # 保存切分后的音频片段
        chunk.export(output_file, format="wav")
        print(f"Saved chunk {i + 1} for {file_name} to {output_file}")


# 遍历每个物种文件夹进行处理
for species_name in species_folders:
    species_folder = os.path.join(input_folder, species_name)

    # 检查物种文件夹是否存在
    if os.path.exists(species_folder):
        for file_name in os.listdir(species_folder):
            file_path = os.path.join(species_folder, file_name)

            # 检查是否为WAV文件
            if file_name.endswith(".wav"):
                print(f"Processing {file_name} in {species_name}...")
                split_audio(file_path, species_name)
    else:
        print(f"Folder for {species_name} not found.")

print("Audio splitting completed.")
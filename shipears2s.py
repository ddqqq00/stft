import os
from pydub import AudioSegment

# 定义路径
input_folder = r"D:\PyCharm\complex_STFT_SST\shipsEar"
output_folder = r"D:\PyCharm\complex_STFT_SST\shipsEar2s"

# 定义类别文件夹
class_folders = ["classA", "classB", "classC", "classD", "classE"]

# 切分长度（2秒）
segment_length_ms = 2000  # 2秒

# 确保输出目录存在
for class_name in class_folders:
    class_output_folder = os.path.join(output_folder, class_name)
    if not os.path.exists(class_output_folder):
        os.makedirs(class_output_folder)


# 切分音频函数
def split_audio(file_path, class_name):
    try:
        # 读取音频文件
        audio = AudioSegment.from_wav(file_path)

        # 获取音频长度（毫秒）
        audio_length = len(audio)

        # 计算可以切分的段数（每段2秒）
        num_segments = audio_length // segment_length_ms

        # 获取文件名（不含扩展名）
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_name)[0]

        # 切分音频
        for i in range(num_segments):
            start_ms = i * segment_length_ms
            end_ms = start_ms + segment_length_ms

            # 提取2秒片段
            segment = audio[start_ms:end_ms]

            # 构建输出文件路径
            output_file = os.path.join(output_folder, class_name, f"{base_name}_segment_{i + 1}.wav")

            # 保存切分后的音频片段
            segment.export(output_file, format="wav")
            print(f"Saved segment {i + 1} for {file_name} to {output_file}")

    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")


# 处理每个类别文件夹
for class_name in class_folders:
    class_input_folder = os.path.join(input_folder, class_name)

    # 检查类别文件夹是否存在
    if not os.path.exists(class_input_folder):
        print(f"Warning: Input folder {class_input_folder} does not exist. Skipping.")
        continue

    print(f"Processing class: {class_name}")

    # 遍历类别文件夹中的所有WAV文件
    for file_name in os.listdir(class_input_folder):
        if file_name.endswith(".wav"):
            file_path = os.path.join(class_input_folder, file_name)
            print(f"Processing {file_name}...")
            split_audio(file_path, class_name)

print("Audio splitting completed.")
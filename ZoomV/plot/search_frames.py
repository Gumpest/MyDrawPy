#%%
from decord import VideoReader, cpu
import numpy as np
import os
from PIL import Image, ImageOps, ImageDraw, ImageFont
import numpy as np

def load_video(video_path, max_frames_num, fps=1, force_sample=False, timestamp_format="float", timestamp_number_length=3):
    if max_frames_num == 0:
        return np.zeros((1, 336, 336, 3))
    vr = VideoReader(video_path, ctx=cpu(0),num_threads=1)
    total_frame_num = len(vr)
    video_time = total_frame_num / vr.get_avg_fps()
    fps = round(vr.get_avg_fps()/fps)
    print(f"Total frames: {total_frame_num}, Video time: {video_time}, fps: {fps}, vr.get_avg_fps(): {vr.get_avg_fps()}")
    frame_idx = [i for i in range(0, len(vr), fps)]
    frame_time = [i/fps for i in frame_idx]
    if len(frame_idx) > max_frames_num or force_sample:
        sample_fps = max_frames_num
        uniform_sampled_frames = np.linspace(0, total_frame_num - 1, sample_fps, dtype=int)
        frame_idx = uniform_sampled_frames.tolist()
        frame_time = [i/vr.get_avg_fps() for i in frame_idx]
    if timestamp_format == "float":
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
    elif timestamp_format == "int":
        frame_time = [round(i) for i in frame_time]
        frame_time = ",".join([f"{i}s" for i in frame_time])
    elif timestamp_format == "zfill":
        frame_time = [str(round(i)) for i in frame_time]
        zfill_length = timestamp_number_length
        frame_time = [i.zfill(zfill_length) for i in frame_time]
        frame_time = ",".join([f"{i}s" for i in frame_time])
    else:
        raise ValueError(f"Unsupported timestamp format: {timestamp_format}")
    spare_frames = vr.get_batch(frame_idx).asnumpy()

    return spare_frames, frame_time, video_time

def extract_frames(video_path, output_folder, fps=0.1):
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    spare_frames, frame_time, video_time = load_video(video_path, 20)
    print(f"Extracting frames from {video_path} to {output_folder} at {fps} fps")
    print(f"Total frames: {len(spare_frames)}, Frame time: {frame_time}, Video time: {video_time}")
    # 遍历并保存帧
    times = frame_time.split(",")
    for i, frame_array in enumerate(spare_frames):
        frame_filename = os.path.join(output_folder, f"frame_{times[i]}.jpg")
        Image.fromarray(frame_array).save(frame_filename)


def create_timeline_with_timestamps(video_path, output_dir, max_frames=4, fps=1, start_idx=None, end_idx=None, frame_width=200, frame_height=200, padding=3, border_size=8, checker_size=10, font_size=20, num_splits=1, add_timestamp=True):
    # 使用load_video函数获取视频帧
    frames, frame_time, video_time = load_video(video_path, max_frames, fps=fps, timestamp_format="zfill", force_sample=True)
    frame_time_list = frame_time.split(",")  # 分割时间戳字符串为列表
    if start_idx is not None and end_idx is not None:
        frames = frames[start_idx:end_idx]
        frame_time_list = frame_time_list[start_idx:end_idx]

    print(f"Creating timeline with {len(frames)} frames, fps: {fps}, max_frames: {max_frames}, start_idx: {start_idx}, end_idx: {end_idx}")
    # 设置时间轴的总宽度和高度
    total_width = len(frames) * (frame_width + padding) + padding
    timeline_height = frame_height + 2 * border_size + font_size + 5  # 加上时间戳的高度
    if not add_timestamp:
        timeline_height = frame_height + 2 * border_size

    # 创建空白的时间轴图像
    timeline_image = Image.new("RGB", (total_width, timeline_height), "black")
    # 逐帧添加到时间轴图像上
    x_offset = padding
    for i, frame_array in enumerate(frames):
        frame = Image.fromarray(frame_array)
        frame = frame.resize((frame_width, frame_height))
        
        # 创建一个带黑白相间格子的边框
        bordered_frame = Image.new("RGB", (frame_width, frame_height + 2 * border_size), "black")
        bordered_frame.paste(frame, (0, border_size))

        # 绘制黑白相间的格子在上边框和下边框
        draw = ImageDraw.Draw(bordered_frame)
        for x in range(0, frame_width, checker_size):
            # 上边框
            color = "white" if (x // checker_size) % 2 == 0 else "black"
            draw.rectangle([x, 0, x + checker_size, border_size], fill=color)
            # 下边框
            draw.rectangle([x, frame_height + border_size, x + checker_size, frame_height + 2 * border_size], fill=color)

        # 将带有边框的帧粘贴到时间轴图像
        timeline_image.paste(bordered_frame, (x_offset, 0))

        # 添加时间戳文本
        timestamp = frame_time_list[i]
        draw = ImageDraw.Draw(timeline_image)
        text_x = x_offset + frame_width // 2
        text_y = frame_height + 2 * border_size + 10
        if add_timestamp:
            draw.text((text_x, text_y), timestamp, fill="white", anchor="mm", font=ImageFont.load_default(size=font_size))  # 居中绘制时间戳

        x_offset += frame_width + padding

    # 保存时间轴图像
    # 根据指定的分割份数切割并保存
    split_width = total_width // num_splits
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_splits):
        left = i * split_width
        right = (i + 1) * split_width if i < num_splits - 1 else total_width
        split_image = timeline_image.crop((left, 0, right, timeline_height))
        split_path = os.path.join(output_dir, f"timeline_split_{i + 1}.jpg")
        split_image.save(split_path)
        print(f"Saved split {i + 1} to {split_path}")

# 使用示例
# video_path = "/mnt/bn/bes-generalaudit-shared7/home/zhangrui/dataset/evaluation/llava_next/LongVideoBench/videos/66dwcQ1Y048.mp4"
video_path = '/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT/visualize/search/xb-GHGr3s6k_concat.mp4'
# create_timeline_with_timestamps(video_path=video_path, output_dir="visualize/supp_1_coarse", max_frames=80, num_splits=4)
duration = 450
start = 380
end = 420
base=10
for num_frames in [10]:
    # num_frames = 10
    fps = num_frames/(end-start)

    max_frames = round(duration * fps)
    start_idx = round(start * fps)
    end_idx = round(end * fps)

    num_splits = num_frames//base

    create_timeline_with_timestamps(video_path=video_path, output_dir=f"visualize/search_frames_{num_frames}f_grounded", fps=fps, max_frames=max_frames, start_idx=start_idx, end_idx=end_idx, num_splits=num_splits, add_timestamp=True)


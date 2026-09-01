#%%
import os
import glob
from decord import VideoReader, cpu
import matplotlib.pyplot as plt

def get_video_duration(file_path):
    """使用decord获取视频时长（秒）"""
    try:
        # 创建视频读取器
        vr = VideoReader(file_path)
        
        # 获取关键信息
        fps = vr.get_avg_fps()
        total_frames = len(vr)
        
        if fps <= 0:
            raise ValueError(f"无效的帧率: {fps}")
        
        return total_frames / fps
    except Exception as e:
        print(f"错误处理 {os.path.basename(file_path)}: {str(e)}")
        return None
    finally:
        # 显式释放资源
        del vr

def plot_duration_distribution(durations):
    """绘制视频时长分布图"""
    plt.figure(figsize=(12, 6))
    
    # 移除无效数据
    valid_durations = [d for d in durations if d is not None]
    
    # 绘制直方图
    plt.hist(valid_durations, bins=20, color='skyblue', edgecolor='black')
    
    # 设置图表属性
    plt.title('视频时长分布', fontsize=14)
    plt.xlabel('时长（秒）', fontsize=12)
    plt.ylabel('视频数量', fontsize=12)
    plt.grid(axis='y', alpha=0.4)
    
    # 显示统计信息
    avg = sum(valid_durations)/len(valid_durations)
    plt.axvline(avg, color='red', linestyle='dashed', linewidth=1.5, 
                label=f'平均时长: {avg:.1f}秒')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

folder_path = "/mnt/bn/bes-generalaudit-shared7/home/zhangrui/dataset/LVBench/all_videos"

video_files = glob.glob(os.path.join(folder_path, '**/*.mp4'), recursive=True)

if not video_files:
    print("未找到任何MP4文件")

print(f"找到 {len(video_files)} 个视频文件")

# 获取所有视频时长
durations = []
for i, file in enumerate(video_files, 1):
    duration = get_video_duration(file)
    if duration is not None:
        durations.append(duration)
    print(f"处理进度: {i}/{len(video_files)}", end='\r')

# 绘制分布图
if durations:
    plot_duration_distribution(durations)
else:
    print("未成功获取任何视频时长数据")

#%%
min(durations)
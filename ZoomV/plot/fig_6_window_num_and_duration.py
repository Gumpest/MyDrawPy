#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load custom font
from matplotlib import font_manager 
font_path = "/mnt/bn/bes-mllm-shared/zhangyuan.kevin/SparseVLM+/ProductSans-Regular.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Product Sans'

plt.rcParams['grid.color'] = 'lightgray'

# 使用其中一种蓝色（比如steelblue）来绘制直方图
main_color = "#244C8B"
main_color = "#3274BE"
# main_color = "#91B3D7"

df = pd.read_csv('window_num_and_duration.csv')
# 去掉count为0的行
df = df[df['count'] > 0]
count = df['count']
duration = df['duration']


plt.style.use('seaborn-v0_8-paper')
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.3))

"""num_frames,0,8,16,24,32,40,48,56
LongVideoBench,0.58414,0.59387,0.5991,0.59312,0.5991,0.59835,0.59312,0.59536
LongVideoBench(900s-3600s),0.49468,0.51773,0.52482,0.51596,0.52482,0.52482,0.51773,0.52305"""
num_frames = [0,8,16,24,32,40,48,56]
accuracy1 = [0.58414,0.59387,0.5991,0.59312,0.5991,0.59835,0.59312,0.59536]
accuracy2 = [0.49468,0.51773,0.52482,0.51596,0.52482,0.52482,0.51773,0.52305]

frames_accuracy1 = pd.DataFrame({'num_frames': num_frames, 'accuracy': accuracy1})
frames_accuracy2 = pd.DataFrame({'num_frames': num_frames, 'accuracy': accuracy2})


sns.lineplot(x='num_frames', y='accuracy', data=frames_accuracy1, marker='o', color=main_color, ax=ax1, label='Overall')
sns.lineplot(x='num_frames', y='accuracy', data=frames_accuracy2, marker='o', color='green', ax=ax1, label='900s-3600s')
ax1.set_title('Spotlight Frames', pad=10, fontsize=15)
ax1.set_xlabel('Spotlight Frames (Total 64)', fontsize=14)
ax1.set_ylabel('Accuracy', fontsize=14)
ax1.legend(fontsize=12)
ax1.set_axisbelow(True)
ax1.grid(axis='y', alpha=0.3, linestyle='--')  # 只保留横向网格线
ax1.set_xticks(num_frames)  # 设置X轴刻度为num_frames的具体值

# 使用steelblue绘制直方图
sns.histplot(data=count, ax=ax2, bins=10, color=main_color, edgecolor=None, alpha=0.7, stat='percent')
ax2.set_title('Spotlight Windows', pad=10, fontsize=15)
ax2.set_xlabel('Windows', fontsize=14)
ax2.set_ylabel('Frequency', fontsize=14)
ax2.set_axisbelow(True)
ax2.grid(axis='y', alpha=0.3, linestyle='--')  # 只保留横向网格线

sns.histplot(data=duration, ax=ax3, bins=20, color=main_color, edgecolor=None, alpha=0.7, stat='percent')
ax3.set_title('Duration of Windows', pad=10, fontsize=15)
ax3.set_xlabel('Duration (s)', fontsize=14)
ax3.set_ylabel('Frequency', fontsize=14)
ax3.set_axisbelow(True)
ax3.grid(axis='y', alpha=0.3, linestyle='--')  # 只保留横向网格线

plt.tight_layout()
# plt.show()
plt.savefig('spotlight_window_num_and_duration.pdf', dpi=300, bbox_inches="tight")
plt.savefig('spotlight_window_num_and_duration.png', dpi=300, bbox_inches="tight")
# #%%
# accuracy2
# #%%
# df_num_frames = pd.read_csv('num_spotlight_frames.csv')
# df_num_frames['num_frames'] = df_num_frames['num_frames'].astype(int)
# df_num_frames['accuracy'] = df_num_frames['accuracy'].astype(float)

# # 使用seaborn绘制折线图
# sns.lineplot(x='num_frames', y='accuracy', data=df_num_frames, marker='o', color='steelblue')
# plt.title('Accuracy of Spotlight Windows', pad=10, fontsize=14)
# plt.xlabel('Number of Frames', fontsize=12)
# plt.ylabel('Accuracy', fontsize=12)
# plt.tight_layout()
# plt.savefig('num_spotlight_frames.pdf', dpi=300, bbox_inches="tight")
# plt.savefig('num_spotlight_frames.png', dpi=300, bbox_inches="tight")
# plt.show()

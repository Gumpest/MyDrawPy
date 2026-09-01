#%%
import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# 1. 准备示例数据
# ----------------------
tasks = [
    "MVBench", 
    "Perception Test", 
    "MViT", 
    "MLVU", 
    "VideoMME", 
    "LVBench", 
    "Charades-STA"
]

# 每个任务下的三种方法/模型的数值
method1 = [60.9, 66.7, 64.7, 74.7, 65.3, 48.2, 74.5]
method2 = [74.0, 76.2, 85.2, 71.9, 71.5, 42.9, 48.2]
method3 = [69.5, 67.6, 72.7, 65.3, 33.1, 33.1, 31.1]

# 为了保证风格一致，这里选取与示例中相近的配色
# 可以根据需要自行微调
colors = [
    "#AEC6E8",  # 浅蓝
    "#7FBF7B",  # 较深的绿色
    "#B4DEB3"   # 较浅的绿色
]

# ----------------------
# 2. 创建画布和坐标轴
# ----------------------
plt.figure(figsize=(8, 4), dpi=120)
ax = plt.gca()

# x 轴上每个分组（任务）的位置
x = np.arange(len(tasks))
# 每组柱子的宽度
bar_width = 0.25

# ----------------------
# 3. 绘制分组柱状图
# ----------------------
# 为了让柱子并列放置，分别在 x - bar_width、x、x + bar_width 处绘制
rects1 = ax.bar(x - bar_width, method1, width=bar_width, 
                color=colors[0], edgecolor='black', label='Method 1')
rects2 = ax.bar(x, method2, width=bar_width, 
                color=colors[1], edgecolor='black', label='Method 2')
rects3 = ax.bar(x + bar_width, method3, width=bar_width, 
                color=colors[2], edgecolor='black', label='Method 3')

# ----------------------
# 4. 设置坐标轴与标签
# ----------------------
ax.set_xticks(x)
ax.set_xticklabels(tasks, rotation=0, fontsize=10)  # 如果任务名较长，可适当旋转
ax.set_ylabel("Performance Metric", fontsize=11)
# ax.set_title("Comparison on Various Video-Linguistic Tasks", fontsize=12)  # 若需要标题可打开

# 只保留左、下坐标轴边框，使风格更接近示例
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 若需要微调 y 轴范围，可以手动设置
# ax.set_ylim([0, 90])

# ----------------------
# 5. 在柱顶显示数值
# ----------------------
def autolabel(rects):
    """在每个柱形条上方标注对应数值"""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),  # 在柱顶上方 3 像素处
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# ----------------------
# 6. 添加分组大标签（可选）
# ----------------------
# 如果需要和原图相似，把 7 个任务分成三类，比如：
# 前两组是 “Short Video QA”，中间三组是 “Long Video QA”，最后两组是 “Temporal Grounding”
# 可以用文字注释的方式标注在图的上方
ax.text( (0 + 1)/2.0 - 0.5, 90, "Short Video QA", ha='center', va='bottom', fontsize=10)
ax.text( (2 + 4)/2.0,       90, "Long Video QA",  ha='center', va='bottom', fontsize=10)
ax.text( (5 + 6)/2.0 + 0.5, 90, "Temporal Grounding", ha='center', va='bottom', fontsize=10)

# ----------------------
# 7. 添加图例并紧凑布局
# ----------------------
ax.legend(frameon=False, fontsize=10, loc='upper left')
plt.tight_layout()
plt.show()
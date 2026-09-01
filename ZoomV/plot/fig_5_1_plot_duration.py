#%%
import numpy as np
import matplotlib.pyplot as plt

# Load custom font
from matplotlib import font_manager 
font_path = "/mnt/bn/bes-mllm-shared/zhangyuan.kevin/SparseVLM+/ProductSans-Regular.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Product Sans'

plt.rcParams['grid.color'] = 'lightgray'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

baseline_color = "#244C8B"
tsg_color = "#3274BE"
rts_color = "#91B3D7"

# ============ 子图1：LongVideoBench ============
labels_1 = ['8s-60s', '180s-600s', '900s-3600s']
baseline_1 = [71.7, 55.83, 51.95]
tsg_1 = [72.30, 60.19, 51.77]
rts_1 = [72.30, 59.95, 54.97]

x1 = np.arange(len(labels_1))
width = 0.25

rects1_1 = ax1.bar(x1 - width, baseline_1, width, label='Baseline', color=baseline_color)
rects1_2 = ax1.bar(x1, tsg_1, width, label='w/ TemporalLink', color=tsg_color)
rects1_3 = ax1.bar(x1 + width, rts_1, width, label='w/ TemporalLight', color=rts_color)
ax1.set_axisbelow(True)
ax1.grid(linestyle='--', axis='y')
ax1.set_xticks(x1)
ax1.set_xticklabels(labels_1, fontsize=12)
ax1.set_ylim([50, 75])  # 从50开始
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('LongVideoBench', fontsize=12)
ax1.legend()

def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1_1, ax1)
autolabel(rects1_2, ax1)
autolabel(rects1_3, ax1)

# ============ 子图2：MLVU ============

labels_2 = ['180s-600s', '900s-32550s']
baseline_2 = [66.51, 55.91]
tsg_2 = [67.11, 57.57]
rts_2 = [68.63, 58.06]

x2 = np.arange(len(labels_2))
ax2.set_axisbelow(True)
ax2.grid(linestyle='--', axis='y')
rects2_1 = ax2.bar(x2 - width, baseline_2, width, label='Baseline', color=baseline_color)
rects2_2 = ax2.bar(x2, tsg_2, width, label='w/ TemporalLink', color=tsg_color)
rects2_3 = ax2.bar(x2 + width, rts_2, width, label='w/ TemporalLight', color=rts_color)
# ax2.grid() 
# 虚线，仅横向
ax2.set_xticks(x2)
ax2.set_xticklabels(labels_2, fontsize=12)
ax2.set_ylim([50, 75])  # 同样从50开始
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('MLVU', fontsize=12)
ax2.legend()
autolabel(rects2_1, ax2)
autolabel(rects2_2, ax2)
autolabel(rects2_3, ax2)

plt.tight_layout()
# plt.show()
# save pdf
plt.savefig('duration_plot.pdf', dpi=300, bbox_inches="tight")
plt.savefig('duration_plot.png', dpi=300, bbox_inches="tight")
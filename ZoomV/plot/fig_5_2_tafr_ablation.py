#%%
import numpy as np
import matplotlib.pyplot as plt

# Load custom font
from matplotlib import font_manager 
font_path = "/mnt/bn/bes-mllm-shared/zhangyuan.kevin/SparseVLM+/ProductSans-Regular.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Product Sans'

plt.rcParams['grid.color'] = 'lightgray'

# Define figure size
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

# Define colors
color_baseline = "#244C8B" # Dark gray
color_tafr = "#3274BE" # Orange

# Data for QVHighlight
labels_1 = ["R@0.5", "R@0.7", "mIoU"]
baseline_1 = [71.10, 49.42, 62.96]
tafr_1 = [77.61, 63.16, 70.40]

x1 = np.arange(len(labels_1))
width = 0.3

rects1_1 = ax1.bar(x1 - width/2, baseline_1, width, label="w/o TemporalLink", color=color_baseline)
rects1_2 = ax1.bar(x1 + width/2, tafr_1, width, label="w/ TemporalLink", color=color_tafr)
ax1.set_axisbelow(True)
ax1.grid(linestyle="--", axis="y")
ax1.set_xticks(x1)
ax1.set_xticklabels(labels_1, fontsize=12)
ax1.set_ylim([40, 90])
ax1.set_ylabel("Score", fontsize=12)
ax1.set_title("QVHighlight", fontsize=12)
ax1.legend(fontsize=12)

# Data for ReXTime
labels_2 = ["R@0.5", "R@0.7", "mIoU"]
baseline_2 = [30.18, 19.54, 34.18]
tafr_2 = [38.65, 27.25, 38.98]

x2 = np.arange(len(labels_2))

rects2_1 = ax2.bar(x2 - width/2, baseline_2, width, label="w/o TemporalLink", color=color_baseline)
rects2_2 = ax2.bar(x2 + width/2, tafr_2, width, label="w/ TemporalLink", color=color_tafr)
ax2.set_axisbelow(True)
ax2.grid(linestyle="--", axis="y")
ax2.set_xticks(x2)
ax2.set_xticklabels(labels_2, fontsize=12)
ax2.set_ylim([10, 50])
ax2.set_ylabel("Score", fontsize=12)
ax2.set_title("ReXTime", fontsize=12)
ax2.legend(fontsize=12)

# Function to annotate values on bars
def autolabel(rects, ax):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f"{height:.1f}",
                    xy=(rect.get_x() + rect.get_width()/2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

autolabel(rects1_1, ax1)
autolabel(rects1_2, ax1)
autolabel(rects2_1, ax2)
autolabel(rects2_2, ax2)

plt.tight_layout()
# plt.show()
plt.savefig("tafr_ablation.pdf", dpi=300, bbox_inches="tight")
plt.savefig("tafr_ablation.png", dpi=300, bbox_inches="tight")
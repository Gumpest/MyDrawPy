#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Load custom font
from matplotlib import font_manager 
font_path = "/mnt/bn/bes-mllm-shared/zhangyuan.kevin/SparseVLM+/ProductSans-Regular.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Product Sans'

plt.rcParams['grid.color'] = 'lightgray'

# Updated data
durations = [180, 600, 1200, 1800, 2400]

data = {
    "ZoomV":       [5.4, 5.4, 12.1, 18.9, 25.7],
    "VideoTree":   [7.8, np.nan, np.nan, np.nan, np.nan],
    "LongVU": [np.nan, np.nan, 32.96, np.nan, np.nan],
    "LLaMA-VID": [np.nan, np.nan, 55.3, np.nan, np.nan],
    "VideoLLaMA2": [np.nan, np.nan, 58.6, np.nan, np.nan],
    "LLaVA-OneVision": [np.nan, np.nan, 25.84, np.nan, np.nan],
}

df = pd.DataFrame(data, index=durations)

plt.figure(figsize=(6, 4))

# Plot ZoomV points (highlighted)
plt.scatter(df.index, df["ZoomV"], s=70, marker="s", color="#d62728")
# plt.scatter(df.index, df["ZoomV"], s=120, marker="o", color="#d62728", 
#             edgecolors="black", linewidths=1.0)
plt.plot(df.index, df["ZoomV"], color="#d62728", linewidth=1.5, zorder=4)
# Plot other models + annotate name near each point
for model in df.columns:
    if model != "ZoomV":
        series = df[model].dropna()
        for x, y in series.items():
            plt.scatter(x, y, s=80, marker="x")
            plt.text(x + 60, y, model, fontsize=10, va="center")

plt.xlabel("Video Length (s)")
plt.ylabel("Inference Time (s)")
plt.title("Runtime Comparison of Different Models with ZoomV")
plt.gca().set_axisbelow(True)
plt.grid(True, linestyle=":")
# 标注，实心点是zoomv，X是其他模型
plt.legend(["ZoomV"])

plt.tight_layout()
# plt.show()
plt.savefig("plot_cost_inference_time.pdf", dpi=300, bbox_inches="tight")
plt.savefig("plot_cost_inference_time.png", dpi=300, bbox_inches="tight")
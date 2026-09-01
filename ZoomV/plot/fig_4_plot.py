#%%
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager 

prob_list = [0.304165951436264, 0.5718036347641089, 0.7233146977673899, 0.8256359704900668, 0.8863455157186216, 0.925020290780372, 0.9511821778035582, 0.96805010891147, 0.9805456549710925, 0.9905207767754776]
miou_list = [0.20532343591108387, 0.24664108061166432, 0.27375979195227257, 0.381419460452166, 0.4133396458002415, 0.4934914509384848, 0.5601958690530837, 0.5033757444770682, 0.5987076229180766, 0.6130803777092418]
fraction_of_positives = [0.35294117647058826, 0.2692307692307692, 0.42105263157894735, 0.4716981132075472, 0.5846153846153846, 0.49230769230769234, 0.6060606060606061, 0.6666666666666666, 0.8152173913043478, 0.9449760765550239]
mean_predicted_value = [0.05689794702076864, 0.16138248963858765, 0.2513623584071849, 0.35330938828745057, 0.4501428902104477, 0.5515144501830318, 0.6508183258833184, 0.7488564078705443, 0.8537384254783545, 0.9722834679677657]

# 设置论文风格
sns.set_theme(style="ticks", font_scale=0.95)

# Load custom font
font_path = "/mnt/bn/bes-mllm-shared/zhangyuan.kevin/SparseVLM+/ProductSans-Regular.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Product Sans'
plt.rcParams['font.size'] = 18

# 创建一个 1x2 并列子图，figsize 加宽
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# 第一个图：Yes/No Reflection vs mIoU
axes[0].plot(prob_list, miou_list, marker="s", markersize=5, linestyle="-", color="#1f77b4", label="mIoU")
axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7, label="y=x")
axes[0].set_xlabel("Probability", labelpad=4, fontsize=13)
axes[0].set_ylabel("mIoU", labelpad=4, fontsize=13)
axes[0].set_title("Yes/No Reflection vs mIoU", fontsize=15, pad=6)
axes[0].grid(True, linestyle="--", alpha=0.5)
axes[0].tick_params(axis='both', labelsize=12)

# 第二个图：MC Reflection Calibration Curve
axes[1].plot(mean_predicted_value, fraction_of_positives, marker="s", markersize=5, linestyle="-", color="#d62728")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.7)
axes[1].set_xlabel("Probability", labelpad=4, fontsize=13)
axes[1].set_ylabel("Accuracy", labelpad=4, fontsize=13)
axes[1].set_title("Multiple Choice Reflection vs Accuracy", fontsize=15, pad=6)
axes[1].grid(True, linestyle="--", alpha=0.5)
axes[1].tick_params(axis='both', labelsize=12)

# 收紧整体布局
plt.tight_layout()
plt.show()

#%% save pdf
fig.savefig("yes_no_reflection_vs_mIoU_and_mc_reflection_vs_accuracy.png", dpi=300, bbox_inches="tight")
fig.savefig("yes_no_reflection_vs_mIoU_and_mc_reflection_vs_accuracy.pdf", dpi=300, bbox_inches="tight")
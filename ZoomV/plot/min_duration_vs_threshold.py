#%%
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 设置全局字体
plt.rcParams['font.size'] = 16  # 设置全局默认字体大小
plt.rcParams['font.family'] = 'Times New Roman'
# 读取CSV文件
df_longvideo_bench = pd.read_csv('threshold_longvideo_bench.csv')
df_lvbench = pd.read_csv('threshold_lvbench.csv')

# 筛选阈值范围 0.4-1.0
df_filtered_longvideo = df_longvideo_bench[df_longvideo_bench['threshold'] >= 0.5]
df_filtered_lvbench = df_lvbench[df_lvbench['threshold'] >= 0.5]

# 将数据重塑为热力图所需的格式
df_heatmap_longvideo = df_filtered_longvideo.set_index('threshold')
df_heatmap_lvbench = df_filtered_lvbench.set_index('threshold')

# 将数值转换为百分比（乘以100）
df_heatmap_longvideo = df_heatmap_longvideo * 100
df_heatmap_longvideo = df_heatmap_longvideo[['2400', '1200', '600', '300']]
df_heatmap_longvideo = df_heatmap_longvideo.T

df_heatmap_lvbench = df_heatmap_lvbench * 100
df_heatmap_lvbench = df_heatmap_lvbench[['2400', '1200', '600', '300']]
df_heatmap_lvbench = df_heatmap_lvbench.T

# 创建热力图，左右两列分别显示longvideo和lvbench
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 绘制第一个热力图（longvideo）
sns.heatmap(df_heatmap_longvideo, 
            cmap='GnBu',  # 使用绿色到蓝色的渐变色
            annot=True,   # 显示数值
            fmt='.1f',    # 数值格式化为1位小数
            linewidths=0,  # 添加网格线宽度
            linecolor='white',  # 设置网格线颜色
            annot_kws={'size': 16, 'color': 'black'},  # 设置数值的字体
            cbar=False,  # 移除colorbar
            ax=ax1)

# 绘制第二个热力图（lvbench）
sns.heatmap(df_heatmap_lvbench, 
            cmap='GnBu',  # 使用绿色到蓝色的渐变色
            annot=True,   # 显示数值
            fmt='.1f',    # 数值格式化为1位小数
            linewidths=0,  # 添加网格线宽度
            linecolor='white',  # 设置网格线颜色
            annot_kws={'size': 18, 'color': 'black'},  # 设置数值的字体
            cbar=False,  # 移除colorbar
            ax=ax2)

titles = ['LongVideoBench', 'LVBench']
# 为两个子图分别设置边框和标签
for ax, title in zip([ax1, ax2], titles):
    # 添加边框
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_linewidth(1)
        spine.set_color('black')
    
    # 将X轴刻度移到顶部
    # ax.xaxis.set_ticks_position('top')
    # ax.xaxis.set_label_position('top')
    # 设置标题
    ax.set_title(title, fontsize=20, pad=10)
    
    # 设置标签
    ax.set_xlabel('Threshold $\epsilon$', fontsize=18, color='black')
    ax.set_ylabel('Min. Duration $\Delta$(s)', fontsize=18, color='black')
    
    # 设置刻度标签
    ax.tick_params(colors='black', labelsize=18)

# 调整布局
plt.tight_layout()
# 显示图表
# plt.show()
plt.savefig('min_duration_vs_threshold.pdf')

#%% 按照threshold排序
# df_heatmap_longvideo = df_heatmap_longvideo.sort_index(ascending=False)
df_lvbench = pd.read_csv('/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT/scripts/analyze/threshold_lvbench.csv')
df_filtered_lvbench = df_lvbench[df_lvbench['threshold'] >= 0.5]

# 将数据重塑为热力图所需的格式
df_heatmap_lvbench = df_filtered_lvbench.set_index('threshold')
print(df_heatmap_lvbench)
df_heatmap_lvbench = df_heatmap_lvbench * 100
print(df_heatmap_lvbench)
df_heatmap_lvbench = df_heatmap_lvbench[['2400', '1200', '600', '300']]
print(df_heatmap_lvbench)
df_heatmap_lvbench = df_heatmap_lvbench.T
print(df_heatmap_lvbench)

#%%
import matplotlib.font_manager as fm
print(fm.findfont('Times New Roman'))

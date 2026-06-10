import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager 

font_path = "/mnt/bn/bes-mllm-shared/zhangyuan.kevin/SparseVLM+/Palatino.ttf"
font_manager.fontManager.addfont(font_path)
plt.rcParams['font.family'] = 'Palatino'

# Define the range for x (representing the error values)
x = np.linspace(0, 2, 100)

# =======================
# Loss definitions
# =======================

# MSE
mse_loss = x ** 2
mse_gradient = 2 * x

# LogMSE
logmse_loss = np.log(1 + x ** 2)
logmse_gradient = (2 * x) / (1 + x ** 2)

# KL divergence (Gaussian: N(0,1) || N(x,1))
kl_loss = 0.5 * x ** 2
kl_gradient = x

# DIST
dist_loss = 1 - 1 / np.sqrt(1 + x ** 2)
dist_gradient = x / (1 + x ** 2) ** (3 / 2)

# =======================
# Plot
# =======================

plt.figure(figsize=(8, 4.5))
plt.rcParams.update({'font.size': 14}) 

# ---- Loss comparison ----
plt.subplot(2, 1, 1)
plt.plot(x, mse_loss, label='MSE Loss', color='#3374BE')
plt.plot(x, logmse_loss, label='LogMSE Loss', color='#60AA66')
# plt.plot(x, kl_loss, label='KL Loss', color='#E1812C')
# plt.plot(x, dist_loss, label='DIST Loss', color='#B279A2')
plt.title('Loss Comparison', fontsize=19)
plt.ylabel('Loss', fontsize=16)
plt.legend(fontsize=12)

# ---- Gradient comparison ----
plt.subplot(2, 1, 2)
plt.plot(x, mse_gradient, label='MSE Gradient', color='#3374BE', linestyle='dashed')
plt.plot(x, logmse_gradient, label='LogMSE Gradient', color='#60AA66', linestyle='dashed')
# plt.plot(x, kl_gradient, label='KL Gradient', color='#E1812C', linestyle='dashed')
# plt.plot(x, dist_gradient, label='DIST Gradient', color='#B279A2', linestyle='dashed')
plt.title('Gradient Comparison', fontsize=19)
plt.xlabel('Error Value', fontsize=16)
plt.ylabel('Gradient', fontsize=16)
plt.legend(fontsize=12)

plt.tight_layout()
plt.savefig('logmse_kl.pdf')
plt.savefig('logmse_kl.png')

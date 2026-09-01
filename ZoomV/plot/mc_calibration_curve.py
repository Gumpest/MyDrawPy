#%%
import json
import re

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l) for l in f]

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')

#%%
def extract_answer_letter(text):
    match = re.search(r"Answer:\s*(\w)", text)
    if match:
        return match.group(1)
    else:
        return text[0:1]

def calculate_accuracy(input_data):
    # 计算准确率
    total_num = len(input_data)
    correct_num = 0
    for x in input_data:
        if extract_answer_letter(x['pred']) == extract_answer_letter(x['target']):
            correct_num += 1
    return correct_num / total_num

#%%
answer_path = "/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT/scripts/analyze/mc_reflection/rextime_val_answering.jsonl"
answer_data = load_jsonl(answer_path)

input_data = sorted(answer_data, key=lambda x: x['token_scores'], reverse=False)
calculate_accuracy(input_data)
# 划分为10个bins，分别计算每个bin的指标
# 归一化token_scores
scores = [x['token_scores'] for x in input_data]
min_score = min(scores)
max_score = max(scores)
normalized_scores = [(x - min_score) / (max_score - min_score) for x in scores]
for x in input_data:
    x['normalized_scores'] = (x['token_scores'] - min_score) / (max_score - min_score)
    x['true_label'] = extract_answer_letter(x['pred']) == extract_answer_letter(x['target'])

#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

y_pred_prob = [x['normalized_scores'] for x in input_data]
y_true = [x['true_label'] for x in input_data]
fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_prob, n_bins=10,
                                                                # strategy='quantile',
                                                                )

# 画校准曲线
plt.figure(figsize=(4, 4))
plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model Calibration")
plt.plot([0, 1], [0, 1], "o--", label="Perfect Calibration")  # y=x 参考线
plt.xlabel("Mean Predicted Confidence")
plt.ylabel("Empirical Accuracy")
plt.title("MC Reflection Calibration Curve")
plt.legend()
plt.grid()
plt.show()

#%%
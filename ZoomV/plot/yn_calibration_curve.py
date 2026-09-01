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

def extract_time_range(x):
    """
    example: '<image>\nQuestion: What happens after they make a circle around the yard several times?\nOptions:\n(A) they sit down on the grass to catch their breath.\n(B) a sudden rain forces them to run indoors.\n(C) the one hugs the other as he holds a drink.\n(D) one starts to set up a game for them to play next.\nDuration: 150.7\nProposed time range: [[72, 134]]\nIs the proposed time range relevant to the question and options?'},
   {'from': 'gpt', 'value': '[[133, 140]]'}],
    """
    try:
        time_range = re.findall(r"\[\[\d+, \d+\]\]", x)
        if len(time_range) == 1:
            return time_range[0]
    except Exception as e:
        print(e)
        return []

    
def convert_one(x):
    """
    {'id': 'anet_val1459',
    'conversations': [{'from': 'human',
        'value': '<image>\nQuestion: Why does the first man grip the bridge railing and prepare to jump off the bridge?\nOptions:\n(A) to retrieve a valuable item that accidentally fell into the water below.\n(B) to jump off the bridge while the second man throws a rope down alongside.\n(C) to demonstrate a safety drill for a water rescue training session.\n(D) to perform a bungee jump as part of an extreme sports event.\nDuration: 61.38\nProposed time range: [[0, 36]]\nIs the proposed time range relevant to the question and options?'},
    {'from': 'gpt', 'value': '[[36, 40]]'}],
    'data_source': 'activitynet',
    'video': 'activitynet/videos/v_zSeLjjo3KF0.mp4',
    'target': '[[36, 40]]',
    'pred': '[[0, 36]]',
    'token_scores': 0.8262702822685242,
    'pred_verify': 'Yes.'},
    """
    pred_time = extract_time_range(x['conversations'][0]['value'])
    # print(pred_time)
    x['pred_verify'] = x['pred']
    if not x['pred_verify'].startswith("Yes"):
        x["token_scores"] = 1 - x["token_scores"]
    x['pred'] = f"{pred_time}"
    return x

def convert_all(data):
    return [convert_one(x) for x in data]


#%%
input_path = "/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT/scripts/analyze/yes_no_reflection/rextime_val_grounding_verify2.jsonl"
input_data = load_jsonl(input_path)
input_data = convert_all(input_data)
input_data

input_data = [x for x in input_data if x['token_scores'] != -1]
input_data = sorted(input_data, key=lambda x: x['token_scores'], reverse=False)

token_scores = [x['token_scores'] for x in input_data]
min_score = min(token_scores)
max_score = max(token_scores)
# min_score_idx = token_scores.index(min_score)
# max_score_idx = token_scores.index(max_score)
print(min_score, max_score)

for x in input_data:
    x['normalized_scores'] = (x['token_scores'] - min_score) / (max_score - min_score)
#%%
import sys
sys.path.insert(0, "/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT")
from llava.eval.evaluate_moment_retrieval import *

def eval_llm_moment_retrieval(results, output_file=None, verbose=True):

    total_num = len(results)
    print(f"Total number of queries: {total_num}")
    
    results_interpreted = [
        {
            "qid": r["id"],
            "pred_relevant_windows": moment_str_to_list(r["pred"]), # tricky(moment_str_to_list(r["pred"]), moment_str_to_list(r["target"])),
            "relevant_windows": moment_str_to_list(r["target"]),
        }
        for r in results
    ]

    all_metrics = eval_submission(results_interpreted, results_interpreted)
    r1_avg = all_metrics["brief"]["MR-full-R1-avg"]
    mIoU = all_metrics["brief"]["MR-full-mIoU"]
    invalid_pred_num = all_metrics["brief"]["MR-full-invalid_pred_num"]

    # log metrics
    metrics = {
        "agg_metrics": r1_avg,
        "r1": all_metrics["full"]["MR-R1"],
        "mAP": all_metrics["full"]["MR-mAP"],
        "mIoU": mIoU,
        "invalid_predictions": invalid_pred_num / total_num,
        "total": total_num,
    }
    if verbose:
        print(all_metrics)
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(metrics, f)
        return None
    return metrics


# 划分为10个bins，分别计算每个bin的指标
# eval_llm_moment_retrieval(input_data)
num_samples = len(input_data)
num_bins = 10
bin_size = num_samples // num_bins
miou_list = []
prob_list = []
for i in range(num_bins):
    bin_data = input_data[i * bin_size: (i + 1) * bin_size]
    metrics = eval_llm_moment_retrieval(bin_data)
    miou = metrics['mIoU']
    miou_list.append(miou)
    prob = sum([x['normalized_scores'] for x in bin_data]) / len(bin_data)
    # prob = min([x['normalized_scores'] for x in bin_data])
    # prob = i / num_bins
    prob_list.append(prob)
    
# %%
prob_list = [0.304165951436264, 0.5718036347641089, 0.7233146977673899, 0.8256359704900668, 0.8863455157186216, 0.925020290780372, 0.9511821778035582, 0.96805010891147, 0.9805456549710925, 0.9905207767754776]
miou_list = [0.20532343591108387, 0.24664108061166432, 0.27375979195227257, 0.381419460452166, 0.4133396458002415, 0.4934914509384848, 0.5601958690530837, 0.5033757444770682, 0.5987076229180766, 0.6130803777092418]


import matplotlib.pyplot as plt
# plot
# 横轴prob，纵轴mIoU
plt.figure(figsize=(4, 4))
plt.plot(prob_list, miou_list, "s-")
plt.plot([0, 1], [0, 1], "--")  # y=x 参考线
plt.xlabel("Probability")
plt.ylabel("mIoU")
plt.title("Yes/No Reflection vs mIoU")
plt.grid()
plt.show()
# %%

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
# ! pip3 install scikit-learn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

y_pred_prob = [x['normalized_scores'] for x in input_data]
y_true = [x['true_label'] for x in input_data]
fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_pred_prob, n_bins=10,
                                                                # strategy='quantile',
                                                                )

fraction_of_positives = [0.35294117647058826, 0.2692307692307692, 0.42105263157894735, 0.4716981132075472, 0.5846153846153846, 0.49230769230769234, 0.6060606060606061, 0.6666666666666666, 0.8152173913043478, 0.9449760765550239]
mean_predicted_value = [0.05689794702076864, 0.16138248963858765, 0.2513623584071849, 0.35330938828745057, 0.4501428902104477, 0.5515144501830318, 0.6508183258833184, 0.7488564078705443, 0.8537384254783545, 0.9722834679677657]
# 画校准曲线
# plt.figure(figsize=(4, 4))
plt.plot(mean_predicted_value, fraction_of_positives, "s-")
plt.plot([0, 1], [0, 1], "o--")  # y=x 参考线
plt.xlabel("Mean Predicted Confidence")
plt.ylabel("Empirical Accuracy")
plt.title("MC Reflection Calibration Curve")
plt.legend()
plt.grid()
plt.show()
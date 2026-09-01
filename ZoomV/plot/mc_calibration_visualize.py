#%%
import sys
sys.path.insert(0, "/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT")
from llava.eval.evaluate_moment_retrieval import *
import json
import re

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l) for l in f]

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')


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
        print(metrics)
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(metrics, f)
        return None
    return metrics

#%%
#%%
answer_path = "/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT/scripts/analyze/mc_reflection/rextime_val_answering.jsonl"
answer_data = load_jsonl(answer_path)
ground_path = "/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT/scripts/analyze/mc_reflection/rextime_val_grounding.jsonl"
ground_data = load_jsonl(ground_path)

# join两个数据集，根据id匹配
answer_data_dict = {x['id']: x for x in answer_data}
for x in ground_data:
    x['reflection_score'] = answer_data_dict[x['id']]['token_scores']

input_data = sorted(ground_data, key=lambda x: x['token_scores'], reverse=False)
eval_llm_moment_retrieval(input_data)
input_data
#%%
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
    prob_list.append(i/num_bins)

# plot
# 横轴prob，纵轴mIoU
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
plt.plot(prob_list, miou_list, marker='o')
# 画一条对角线，虚线，作为参照
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Reflection probability')
plt.ylabel('mIoU GT')
# save pdf
# plt.savefig('calibration.pdf')

#%%
input_data_true = [x for x in input_data if x['pred_verify'].startswith("Yes")]
input_data_false = [x for x in input_data if not x['pred_verify'].startswith("Yes")]
eval_llm_moment_retrieval(input_data_true)
eval_llm_moment_retrieval(input_data_false)
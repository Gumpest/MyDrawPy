#%%
import sys
sys.path.insert(0, "/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT")
import json
import re

def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(l) for l in f]

def save_jsonl(data, path):
    with open(path, 'w') as f:
        for d in data:
            f.write(json.dumps(d) + '\n')
    
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
    x['pred_verify'] = x['pred']
    if not x['pred_verify'].startswith("Yes"):
        x["token_scores"] = 1 - x["token_scores"]
    x['pred'] = f"{pred_time}"
    return x

def convert_all(data):
    return [convert_one(x) for x in data]


#%%
input_path = "/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT/work_dirs/experiments/rextime-ptrue-quick-verify-lv-ft_llm-60f-1ep-bs8-interlv/checkpoint-500/rextime_val_grounding_verify2.jsonl"
verify_path = "/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT/work_dirs/experiments/gvllm-lv-ft_llm-60f-2ep-bs8-interlv--qwen/2024-10-29-00-49-59/rextime_val_grounding_verify.jsonl"
ground_path = "/mnt/bn/besaudit-pjw-lq/LLaVA-NeXT/work_dirs/experiments/gvllm-lv-ft_llm-60f-2ep-bs8-interlv--qwen/2024-10-29-00-49-59/rextime_val_grounding.jsonl"
input_data = load_jsonl(input_path)
input_data = convert_all(input_data)

input_data = sorted(input_data, key=lambda x: x['token_scores'], reverse=False)
input_data

#%%
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
        print(metrics)
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(metrics, f)
        return None
    return metrics


# 划分为10个bins，分别计算每个bin的指标
# eval_llm_moment_retrieval(input_data)
num_samples = len(input_data)
num_bins = 5
bin_size = num_samples // num_bins
miou_list = []
prob_list = []
for i in range(num_bins):
    bin_data = input_data[i * bin_size: (i + 1) * bin_size]
    print(len(bin_data))
    metrics = eval_llm_moment_retrieval(bin_data)
    miou = metrics['mIoU']
    miou_list.append(miou)
    prob_list.append(i/num_bins)

print(miou_list)
print(prob_list)

# plot
# 横轴prob，纵轴mIoU
import matplotlib.pyplot as plt
plt.figure(figsize=(4, 4))
plt.plot(prob_list, miou_list, marker='o')
# 画一条对角线，虚线，作为参照
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('Reflection Confidence')
plt.ylabel('mIoU')
# save pdf
plt.savefig('calibration.pdf')

#%%
input_data_true = [x for x in input_data if x['pred_verify'].startswith("Yes")]
print(len(input_data_true))
eval_llm_moment_retrieval(input_data_true)
# input_data_true
#%%
input_data_false = [x for x in input_data if not x['pred_verify'].startswith("Yes")]
print(len(input_data_false))
eval_llm_moment_retrieval(input_data_false)
# input_data_false

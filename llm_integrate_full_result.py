import os
import json
import torch
import tqdm

from litgpt import LLM


def load_data_from_pt(dataset):
    data_path = f'datasets/hypotrans/test_{dataset}_en_zh_st_largev2.pt'
    data = torch.load(data_path)

    nbest, gt = [[],[],[],[],[],], []
    for datapoint in data:
        five_best = datapoint['input']
        ground_truth = datapoint['ground_truth'].strip()
        for i in range(5):
            nbest[i].append(five_best[i].strip())
        gt.append(ground_truth)

    # fix a 'None' gt bug in hypotrans-covost en-zh st dataset
    if dataset == 'covost2':
        gt[458] = 'HTML 级别等于n.'
    
    return nbest, gt


output_folder = "datasets/hypotrans_integrated"
if not os.path.exists(output_folder):
        os.makedirs(output_folder)


llm = LLM.load("meta-llama/Meta-Llama-3-8B-Instruct")
integrated_results = []
for dataset in ['fleurs', 'covost2', 'mustc']:
    nbest, gt = load_data_from_pt(dataset)
    nbest_integrated = {}
    for i in tqdm.trange(len(gt)):
        prompt = [
            {"role": "system", "content": "You are a professional, authentic translation engine. 请整合不同版本的翻译，以生成更高质量的中文结果. Only return the result without any explanations."},
            {"role": "user", "content": "\n".join(nbest[j][i] for j in range(5))},
        ]
        text = llm.generate(prompt, 
                            max_new_tokens=100,
                            temperature=0.2,
                            top_k=1,)
        nbest_integrated[str(i)] = [text.split("\n"), nbest[0][i]] # save full_result and 1-best for comparison and later choice

    current_result = {"dataset": dataset, "nbest_integrated": nbest_integrated, "gt": gt}
    integrated_results.append(current_result)

    with open(output_folder + f'/integrated_{dataset}.json', 'w', encoding='utf-8') as f:
        json.dump([current_result], f, ensure_ascii=False, indent=4)


with open(output_folder + '/integrated_full_results.json', 'w', encoding='utf-8') as f:
    json.dump(integrated_results, f, ensure_ascii=False, indent=4)

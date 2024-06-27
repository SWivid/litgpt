import torch
from pprint import pprint

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


llm = LLM.load("meta-llama/Meta-Llama-3-8B-Instruct")
dataset = 'covost2'

nbest, gt = load_data_from_pt(dataset)
sys_prompt_choice = 1
sys_prompts = [
    "You are a professional, authentic translation engine. Your function is to integrate the diverse versions of translations to generate a higher-quality result, maintaining the original language, and only return the integrated result without any explanations.", # still suffer from output english result
    
    "You are a professional, authentic translation engine. 请整合不同版本的翻译，以生成更高质量的中文结果. Only return the result without any explanations.",
    
    "请整合不同版本的翻译，以生成更高质量的中文结果：", # generated jumbled-form result
    ]

while True:
    inputs_idx = input(f"Enter index of input you want to test in {dataset} 0~{len(gt)-1}: ")
    inputs = "\n".join(nbest[j][int(inputs_idx)] for j in range(5))
    print("### INPUT ####:")
    print(inputs, "\n")
    print("### OUTPUT ###:")

    # suit for llama3, otherwise define your own prompt with corresponding style
    prompt = [
        {"role": "system", "content": sys_prompts[sys_prompt_choice]},
        {"role": "user", "content": inputs},
    ]

    text = llm.generate(prompt, 
                        max_new_tokens=100,
                        temperature=0.2,
                        top_k=1,)
    print(text.split("\n"))

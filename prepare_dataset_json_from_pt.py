## read all .pt files in current folder and merge all to litgpt dataset json files
## Usage: python prepare_dataset_json_from_pt.py

import os
import json
import torch
import argparse
from tqdm import tqdm
from pprint import pprint


def parse_data_to_json(data):
    zh_instuct = '请整合不同版本的翻译，以原语言生成更高质量的结果：'
    en_instuct = 'Please integrate the diverse versions of translations to generate a higher-quality result:'
    json_data = []
    for datapoint in data:
        five_best = datapoint['input']
        ground_truth = datapoint['ground_truth'].strip()
        json_data.append({'instruction': zh_instuct, 'input': '\n'.join(five_best), 'output': ground_truth})
    return json_data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, default='.')
    parser.add_argument('--output_folder', type=str, default='./hypotrans_en_zh_st')
    args = parser.parse_args()

    input_folder = args.input_folder
    output_folder = args.output_folder

    ''' input:
    # ./train_fleurs_en_zh_st_large.pt
    # ./train_covost2_en_zh_st_large.pt
    # ./train_mustc_en_zh_st_large.pt
    # ./dev_fleurs_en_zh_st_large.pt
    # ./dev_covost2_en_zh_st_large.pt
    # ./dev_mustc_en_zh_st_large.pt

        output:
    # ./hypotrans_en_zh_st/train.json
    # ./hypotrans_en_zh_st/val.json
    # ./hypotrans_en_zh_st/test.json '''

    train_data = []
    val_data = []
    test_data = []
    for file in tqdm(os.listdir(input_folder)):
        if file.endswith('.pt'):
            if file.startswith('train_'):
                train_data.extend(parse_data_to_json(torch.load(os.path.join(input_folder, file))))
            elif file.startswith('dev_'):
                val_data.extend(parse_data_to_json(torch.load(os.path.join(input_folder, file))))
            elif file.startswith('test_'):
                test_data.extend(parse_data_to_json(torch.load(os.path.join(input_folder, file))))

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(output_folder + '/train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)
    with open(output_folder + '/val.json', 'w', encoding='utf-8') as f: 
        json.dump(val_data, f, ensure_ascii=False, indent=4)
    with open(output_folder + '/test.json', 'w', encoding='utf-8') as f: 
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    pprint(test_data[:5])


if __name__ == '__main__':
    main()

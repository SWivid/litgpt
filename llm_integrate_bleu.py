import json

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str) # fleurs / covost2 / mustc / full_results
args = parser.parse_args()

import sacrebleu
bleu_metric = sacrebleu.BLEU(tokenize='zh')
chrf_metric = sacrebleu.CHRF()


dataset = args.dataset
data_path = f'datasets/hypotrans_integrated/integrated_{dataset}.json'
with open(data_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
''' ---------DATA FORMAT---------
    [
     {"dataset":          "fleurs", 
      "nbest_integrated": {
                           "0": [["The result of integration if llm do it right"],                                      "The 1-best result from hypotrans"],
                           "1": [["A too long result indicating a bad result of llm, which will be replace by 1-best"], "The 1-best result from hypotrans"],
                           "2": [["jumbled-form makes U guess where the result locates", "jumbled-form result 2", ...], "The 1-best result from hypotrans"],
                           ...: ...,
                          }, 
      "gt":               ["Groundtruth for the 0-idx data point",
                           "Groundtruth for the 1-idx data point",  
                           "Groundtruth for the 1-idx data point", 
                           ...,
                          ]},
      
      {"dataset":         "covost2", 
      "nbest_integrated": <nbest_integrated>, 
      "gt":               <gt>},

      {"dataset":         "mustc", 
      "nbest_integrated": <nbest_integrated>, 
      "gt":               <gt>},
    ]
    ------------------------------
'''


def bleu(input_dict):
    dataset_name = input_dict["dataset"]
    nbest_integrated = input_dict["nbest_integrated"]
    gt = input_dict["gt"]
    pr = []
    for i in range(len(gt)):
        inte_result, one_best = nbest_integrated[str(i)]
        if len(inte_result) > 1: # 3rd mentioned above
            pr.append(one_best)
        else:
            if len(inte_result[0]) > 1.5*len(one_best): # 2nd mentioned above
                pr.append(one_best)
            else:
                pr.append(inte_result[0]) # 1st mentioned above

    # BLEU score
    print('\n', dataset_name)
    print(bleu_metric.corpus_score(pr, [gt]))
    print(chrf_metric.corpus_score(pr, [gt]))

def main():
    for dataset in data:
        bleu(dataset)

if __name__ == '__main__':
    main()

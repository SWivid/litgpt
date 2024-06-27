<div align="center">

# 😢 LLaMA3 for ST 
![PyPI - Python Version](https://img.shields.io/badge/python-3.10-Green)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/Lightning-AI/lit-stablelm/blob/master/LICENSE)

</div>


# Install LitGPT⚡
```bash
# release version 0.4.2.dev0
git clone https://github.com/Lightning-AI/litgpt
cd litgpt
pip install -e '.[all]'
```

&nbsp;
# Download model
```bash
litgpt download meta-llama/Meta-Llama-3-8B-Instruct --access_token <ACCESS-TOKEN>
# 对应下载到 litgpt/checkpoints/meta-llama/Meta-Llama-3-8B-Instruct
```
Visit the respective Model Hub website, e.g., [Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct). 
&nbsp;The access token can be created under your Model Hub in the `Profile > Access` Tokens menu.

&nbsp;
# Integrate with finetuned Llama3
## Do Finetune
Download .pt [HypoTranslate](vscode-local:/d%3A/Workaholic/Postgraduate/2024%E6%98%A5/HypoTranslate/sacrebleu_hypotranslate.py) dataset, run prepare_dataset_json_from_pt.py to generate LitGPT needed data.
```bash
python prepare_dataset_json_from_pt.py
```
### litgpt finetune_adapter_v2
```bash
# work for 8*3090 GPUs
litgpt finetune_adapter_v2 meta-llama/Meta-Llama-3-8B-Instruct \
  --data JSON \
  --data.json_path datasets/hypotrans_en_zh_st \
  --data.seed 42 \
  --out_dir out/finetune/adapter-v2 \
  --devices 8 \
  --train.save_interval 200 \
  --train.global_batch_size 32 \
  --train.micro_batch_size 2 \
  --train.lr_warmup_steps 100 \
  --train.epochs 2 \
  --train.max_seq_length 512 \
  --logger_name wandb
```

<details>
  <summary>p.s. litgpt/litgpt/finetune/adapter_v2.py</summary>

```python
def setup(
    checkpoint_dir: Path,
    out_dir: Path = Path("out/finetune/adapter-v2"),
    precision: Optional[str] = None,
    quantize: Optional[Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8-training"]] = None,
    devices: Union[int, str] = 1,
    data: Optional[DataModule] = None,
    train: TrainArgs = TrainArgs(
        save_interval=1000,
        log_interval=1,
        global_batch_size=16,
        micro_batch_size=1,
        lr_warmup_steps=100,
        epochs=5,
        max_seq_length=None,
    ),
    eval: EvalArgs = EvalArgs(interval=100, max_new_tokens=100, max_iters=100),
    optimizer: Union[str, Dict] = "AdamW",
    logger_name: Literal["wandb", "tensorboard", "csv"] = "csv",
    seed: int = 1337,
) -> None:
    """Finetune a model using the Adapter V2 method.

    Arguments:
        checkpoint_dir: The path to the base model's checkpoint directory to load for finetuning.
        out_dir: Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
            /teamspace/jobs/<job-name>/share.
        precision: The precision to use for finetuning. Possible choices: "bf16-true", "bf16-mixed", "32-true".
        quantize: If set, quantize the model with this algorithm. See ``tutorials/quantize.md`` for more information.
        devices: How many devices/GPUs to use.
        data: Data-related arguments. If not provided, the default is ``litgpt.data.Alpaca``.
        train: Training-related arguments. See ``litgpt.args.TrainArgs`` for details.
        eval: Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details.
        optimizer: An optimizer name (such as "AdamW") or config.
        logger_name: The name of the logger to send metrics to.
        seed: The random seed to use for reproducibility.
    """
```
&nbsp;
</details>

Also, [Example: LoRA finetuning config](https://github.com/Lightning-AI/litgpt/tree/7d0430a78edecd532309ae03eac07e45b4e485ad?tab=readme-ov-file#training-recipes:~:text=Example%3A%20LoRA%20finetuning%20config)

### litgpt finetune_lora
```bash
litgpt finetune_lora meta-llama/Meta-Llama-3-8B-Instruct \
```
## litgpt generate with finetuned models
Not fully tested yet ...😔😔😔
```bash
litgpt generate_adapter_v2 meta-llama/Meta-Llama-3-8B-Instruct \
    --prompt "TODO"
```

&nbsp;
# Integrate directly using LLaMA3-8B-Instruct
Try out `llm_integrate_full_result.py` script,
or start with `llm_integrate_simple_test.py` for initial brief view.

You may craft your own prompt, and the format can be referenced at litgpt repo code
[litgpt/litgpt/prompts.py](https://github.com/Lightning-AI/litgpt/blob/c032d88867cddb5894f23aef45e435921ad0a65e/litgpt/prompts.py#L203), or directly view at [Meta Llama 3 | Model Cards & Prompt formats](https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/).

Example for integrating Chinese N-Best Translations:
```bash
python llm_integrate_full_result.py
```
And do BLEU,
```bash
# --dataset full_result / fleurs / covost2 / mustc
python llm_integrate_bleu.py --dataset full_result
```




&nbsp;

----
## License

[Apache 2.0](https://github.com/Lightning-AI/litgpt/blob/main/LICENSE) license.

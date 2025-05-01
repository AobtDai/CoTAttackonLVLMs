import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
# https://huggingface.co/docs/trl/grpo_trainer
from peft import PrefixTuningConfig, LoraConfig
from typing import Union

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# 数据、数据集处理部分
def extract_xml_answer(text: str) -> str:       # 获取llm的回答部分
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> Union[str, None]:   # 获取正确答案
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# uncomment middle messages for 1-shot prompting
def get_gsm8k_questions(split = "train") -> Dataset:    # 获得数据集并加入prompt
    data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer']) 
    }) # type: ignore
    return data # type: ignore


def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-'*50, f"\n>>>>> Question:\n{q}", f"\n>>>>> Answer: {answer[0]}", f"\n>>>>> Response:\n{responses[0]}", f"\n>>>>> Extracted: {extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


def reasoning_cal(text) -> float:
    count = 0.0
    pattern = r"^<reasoning>\n.*?\n</reasoning>$"
    match = re.match(pattern, text) 
    if not match:
        return 0.0
    else:
        end_point_score = match.end() if match.end() < 200 else 200
        reward_score = 1.0 - (200.0 - end_point_score) / 200.0
        return reward_score
    
def reasoning_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [reasoning_cal(c) for c in contents]


def int_reward_func(completions, **kwargs) -> list[float]:
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [1.0 if r.isdigit() else 0.0 for r in extracted_responses]


def strict_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

def soft_format_reward_func(completions, **kwargs) -> list[float]:
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.3 if match else 0.0 for match in matches]


def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
        # count += 0.05
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]


dataset = get_gsm8k_questions()
model_name = "../models/Qwen2.5-32B-Instruct"

# model_name = "../models/Qwen2.5-0.5B-Instruct"
# model_name = "../models/Qwen2.5-3B-Instruct"

output_dir="../Outputs/qwen2.5-finetuned_GRPO-32B"
run_name="Qwen-32B-GRPO-gsm8k"
checkpoint_path = "../Outputs/qwen2.5-finetuned_GRPO-32B/checkpoint-6120"
# logging_dir = "../Outputs/qwen2.5-finetuned_GRPO-32B/logs/"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    # fp16=True,
    bf16=True,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_generations=2,
    # num_processes = 1,
    # num_generations=16,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=10,
    save_steps=20,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda",
    # report_to="none", #I'm disabling Wandb.
    report_to="tensorboard", 
    # logging_dir=logging_dir,
    # resume_from_checkpoint=checkpoint_path
)

# prefix_config = PrefixTuningConfig(
#     peft_type="PREFIX_TUNING",
#     task_type="CAUSAL_LM",
#     num_virtual_tokens=20,      # 虚拟前缀的长度
#     token_dim=2048,             # 前缀嵌入的维度，同原模型
#     num_layers=36,              # 模型隐藏层数，同原模型
#     prefix_projection=True,     # 是否对前缀进行投影
#     encoder_hidden_size=2048    # 前缀编码器的隐藏层大小，同原模型
# )

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=64,
    bias="none",
    task_type="CAUSAL_LM",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # checkpoint_path,
    torch_dtype=torch.bfloat16,
    # torch_dtype=torch.float16,
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
    # peft_config=prefix_config,
    peft_config=peft_config,
)
trainer.train()

trainer.save_model(output_dir)
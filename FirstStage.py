import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import MllamaForConditionalGeneration, AutoProcessor
from trl import GRPOConfig, GRPOTrainer
# https://huggingface.co/docs/trl/grpo_trainer
from typing import Union

SYSTEM_PROMPT = """
Respond in the following format:
<1.choice>
...
</1.choice>
<2.choice>
...
</2.choice>
<3.choice>
...
</3.choice>
"""

# XML_COT_FORMAT = """\
# <reasoning>
# {reasoning}
# </reasoning>
# <answer>
# {answer}
# </answer>
# """

def extract_xml_answer(text: str) -> str:       
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

def extract_hash_answer(text: str) -> Union[str, None]:   
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def data_process(data) -> Dataset:   
    data = data.map(lambda x: { 
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']} # malicious prompts
        ],
        'gen_answer': extract_hash_answer(x['answer']) # generated answer from victim model
    }) # type: ignore
    return data

# # uncomment middle messages for 1-shot prompting
# def get_gsm8k_questions(split = "train") -> Dataset:   
#     data = load_dataset('openai/gsm8k', 'main')[split] # type: ignore
#     data = data.map(lambda x: { # type: ignore
#         'prompt': [
#             {'role': 'system', 'content': SYSTEM_PROMPT},
#             {'role': 'user', 'content': x['question']}
#         ],
#         'answer': extract_hash_answer(x['answer']) 
#     }) # type: ignore
#     return data # type: ignore

def max_output_reward_func(prompts, completions, gen_answer, **kwargs) -> list[float]:
    leng = [str(t).__len__() for t in gen_answer]
    score = []
    for i, leng_i in enumerate(leng):
        if leng_i <= 100:
            score[i] = 0
        elif leng_i >= 1000:
            score[i] = 2.0
        else:
            score[i] = (leng_i - 100) * 2.0 / 1000
    return score

def get_context_aware_score(system_prompt, gen_answer, eval_model):
    pass

def context_aware_reward_func(prompts, completions, gen_answer, **kwargs) -> list[float]:
    system_prompt_context_aware = '''
    Divide the answer into several part. Then summarize each part and evaluate the correlation between each adjacent parts
    ...
    '''
    # Send system_prompt & gen_answer to eval model
    return [get_context_aware_score(system_prompt_context_aware, t, eval_model) for t in gen_answer]

def semantic_reward_func(prompts, completions, gen_answer, **kwargs) -> list[float]:
    system_prompt_context_aware = '''
    Check the grammar of this text. And analyze the level of its semantic correlation.
    ...
    '''
    # Send system_prompt & gen_answer to eval model
    return [get_context_aware_score(system_prompt_context_aware, t, eval_model) for t in gen_answer]


# def correctness_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     q = prompts[0][-1]['content']
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     print('-'*50, f"\n>>>>> Question:\n{q}", f"\n>>>>> Answer: {answer[0]}", f"\n>>>>> Response:\n{responses[0]}", f"\n>>>>> Extracted: {extracted_responses[0]}")
#     return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


# def reasoning_cal(text) -> float:
#     count = 0.0
#     pattern = r"^<reasoning>\n.*?\n</reasoning>$"
#     match = re.match(pattern, text) 
#     if not match:
#         return 0.0
#     else:
#         end_point_score = match.end() if match.end() < 200 else 200
#         reward_score = 1.0 - (200.0 - end_point_score) / 200.0
#         return reward_score
    
# def reasoning_reward_func(completions, **kwargs) -> list[float]:
#     contents = [completion[0]["content"] for completion in completions]
#     return [reasoning_cal(c) for c in contents]


# def int_reward_func(completions, **kwargs) -> list[float]:
#     responses = [completion[0]['content'] for completion in completions]
#     extracted_responses = [extract_xml_answer(r) for r in responses]
#     return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


if __name__ == '__main__':
    # dataset = get_gsm8k_questions()
    redteam_model_name = "./models/Qwen2.5-0.5B-Instruct"
    # model_name = "../models/Qwen2.5-3B-Instruct"
    victim_model_name = "./models/Llama-3.2V-11B-cot"
    eval_model_name = ""

    output_dir="./Outputs/test-0.5b+11b"
    run_name="test-0.5b+11b"
    # checkpoint_path = "./Outputs/test-0.5b+11b/checkpoint-233"

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
        bf16=True,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        num_generations=2,
        # num_generations=16,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        vllm_gpu_memory_utilization=.3,
        vllm_device="cuda:0",
        report_to="none", #disabling Wandb.
        # resume_from_checkpoint=checkpoint_path
    )

    redteam_model = AutoModelForCausalLM.from_pretrained(
        redteam_model_name,
        # checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map=None
    ).to("cuda")

    redteam_tokenizer = AutoTokenizer.from_pretrained(redteam_model_name)
    redteam_tokenizer.pad_token = redteam_tokenizer.eos_token

    victim_model = MllamaForConditionalGeneration.from_pretrained(
        victim_model_name,
        # checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    victim_processor = AutoProcessor.from_pretrained(victim_model_name)
    
    while(eval()):
        # In each epoch, the dataset will be revised with new malicious perturbation
        dataset = ds_update()
        trainer = GRPOTrainer(
            model=redteam_model,
            processing_class=redteam_tokenizer,
            reward_funcs=[
                # reward func with attacks
                max_output_reward_func,
                context_aware_reward_func,
                semantic_reward_func,

                ],
            args=training_args,
            train_dataset=dataset,
            #peft_config=peft_config
        )
        trainer.train()
        
        # expand raw prompt repository (dataset)
        # remove original one
        # 

        trainer.save_model(output_dir)
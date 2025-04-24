import re
import torch
from datasets import load_dataset, Dataset
from typing import Union

class MyReward:
    def __init__(self):
        pass
    
    # def extract_xml_answer(self, text: str) -> str:       
    #     answer = text.split("<answer>")[-1]
    #     answer = answer.split("</answer>")[0]
    #     return answer.strip()

    # def extract_hash_answer(self, text: str) -> Union[str, None]:   
    #     if "####" not in text:
    #         return None
    #     return text.split("####")[1].strip()

    def max_output_reward_func(self, prompts, completions, gen_answer, **kwargs) -> list[float]:
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

    def get_context_aware_score(self, system_prompt, gen_answer):
        return 0

    def context_aware_reward_func(self, prompts, completions, gen_answer, **kwargs) -> list[float]:
        system_prompt_context_aware = '''
        Divide the answer into several part. Then summarize each part and evaluate the correlation between each adjacent parts
        ...
        '''
        # Send system_prompt & gen_answer to eval model
        return [self.get_context_aware_score(system_prompt_context_aware, t) for t in gen_answer]

    def semantic_reward_func(self, prompts, completions, gen_answer, **kwargs) -> list[float]:
        system_prompt_context_aware = '''
        Check the grammar of this text. And analyze the level of its semantic correlation.
        ...
        '''
        # Send system_prompt & gen_answer to eval model
        return [self.get_context_aware_score(system_prompt_context_aware, t) for t in gen_answer]


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


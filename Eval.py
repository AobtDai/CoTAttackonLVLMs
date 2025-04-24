import os
from openai import AzureOpenAI, OpenAIError

import yaml
from typing import Dict, Any

def load_config(file_path: str) -> Dict[str, Any]:
    with open(file_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        return config
    
config = load_config("apikey.yml")

os.environ["AZURE_OPENAI_API_KEY"] = config["api"]["key"]
os.environ["AZURE_OPENAI_ENDPOINT"] = config["api"]["url"]

def get_llm_response(msg, system_prompt:str=None):
     client = AzureOpenAI(
         azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
         api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
         api_version=config["api"]["api_version"]
     )
     if not system_prompt:
        system_prompt = '''
            You are a Text Analysis Assistant. Your tasks are:

            Text Segmentation & Summarization:
            Divide the input text into logically coherent sections
            Provide concise summaries for each section (2-3 sentences)
            Use clear numbering for segments (Section 1, Section 2, etc.)
            Logical Relationship Analysis:
            Analyze the relationships between consecutive sentences within each section
            Identify relationship types (cause-effect, contrast, elaboration, sequence, contradiction, etc.)
            Assign a relevance score from 1-5 for each sentence pair:
            1 - No clear connection
            2 - Weak/possible connection
            3 - Moderate connection
            4 - Strong connection
            5 - Essential/defining connection
            Explain your scoring rationale briefly
            Output Format:

            Segmented Summary:
            Section 1: [Summary]
            Section 2: [Summary]
            ...
            Logical Analysis:
            Section 1 Analysis:
            Sentence 1 → Sentence 2: [Relationship Type] (Score: X) - [Explanation]
            Sentence 2 → Sentence 3: [Relationship Type] (Score: Y) - [Explanation]
            Section 2 Analysis:
            ...
            """
            Example Input:
            "While renewable energy adoption grows, fossil fuels still dominate global energy production. Solar and wind installations increased by 30% last year. Government subsidies for renewables remain inconsistent. Petroleum companies invest more in carbon capture technologies. Transitioning to clean energy requires both technological innovation and policy support."

            Example Output:

            Segmented Summary:
            Section 1: Current state of energy production with contrasting trends between renewables growth and fossil fuel dominance.
            Section 2: Specific data on renewable growth and inconsistent government support.
            Section 3: Fossil fuel industry response and the requirements for energy transition.
            Logical Analysis:
            Section 1 Analysis:
            Sentence 1 → Sentence 2: Contrast (Score: 4) - Establishes opposing trends using "while"
            Sentence 2 → Sentence 3: Complication (Score: 3) - Introduces nuance about government actions
            Section 2 Analysis:
            Sentence 3 → Sentence 4: Cause-effect (Score: 5) - Direct link between subsidies and corporate investments
            Sentence 4 → Sentence 5: Solution statement (Score: 4) - Connects industry actions to transition requirements
        '''
 
     response = client.chat.completions.create(
         model="gpt-4o",
         messages=[
             { 
                 "role":"system",
                 "content":system_prompt
             },
            {
            "role": "user", 
            "content": [
	            {
	                "type": "text",
	                # "text": "Describe this picture:"
	                "text": msg
	            },
	            # {
	            #     "type": "image_url",
	            #     "image_url": {
                #         "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                #     }
                # } 
                ] 
            }
         ],
         max_tokens=1024,
        #  reasoning_effort = "medium"
    )
     return response.choices[0].message.content

msg = '''
    Reinforcement Learning from Human Feedback (RLHF), which often uses algorithms like Proximal Policy Optimization (PPO) or similar methods (e.g., modified GRPO), provides several advantages when training or fine-tuning large language models (LLMs). Here's a concise summary:

### 1. **Alignment with Human Preferences:**
   - RLHF enables models to better align with specific human preferences, values, and ethical considerations.
   - By incorporating feedback from humans, the models can generate outputs that are more helpful, less biased, and safer for real-world applications.

### 2. **Improved Output Quality:**
   - Human feedback helps the model optimize for outputs that are more coherent, informative, and relevant to user intent.
   - This is particularly valuable for conversational or generative tasks where nuance and personalization are important.

### 3. **Mitigation of Undesirable Outputs:**
   - RLHF helps reduce harmful or inappropriate responses by penalizing outputs flagged as problematic in the reward signal.
   - It allows active training against failure cases such as toxicity, misinformation, or harmful stereotypes.

### 4. **Feedback-Driven Fine-Tuning:**
   - Fine-tuning with RLHF creates a feedback loop between model outputs and human evaluations, allowing the model to learn from mistakes iteratively.
   - This avoids the limitations of static pretraining datasets that may not capture real-world complexity or updated user expectations.

### 5. **Balancing Trade-offs:**
   - RLHF provides a mechanism to balance competing objectives, such as maximizing informativeness while minimizing verbosity or irrelevant details.
   - Reward design ensures that models balance creativity with factual accuracy and safety.

### 6. **Adaptability to Specific Applications:**
   - Human feedback can target the model toward specialized domains or particular use cases, tailoring its behavior and outputs as required (e.g., customer support, content creation, or education).
   - This leads to a more versatile and robust deployment.

### 7. **Enhances User Trust and Satisfaction:**
   - By aligning the model closer to human preferences and avoiding unexpected or harmful responses, RLHF improves how users perceive the tool.
   - Trustworthy and reliable interactions promote wider acceptance and adoption.

### 8. **Efficient Training Paradigm:**
   - RLHF often utilizes computed rewards and human input efficiently, focusing on improving higher-level model behaviors without retraining on vast datasets.
   - Algorithms like PPO allow stable and scalable fine-tuning for large models, making them effective for iterative optimization.

Overall, RLHF empowers LLMs to better understand and respond to human expectations while simultaneously addressing challenges like safety, alignment, and domain-specific customization. It enhances the practicality and reliability of using LLMs in real-world scenarios.

'''
# "Summarize several advatages to use RLHF (like GRPO) to train/finetune a LLM."
output = get_llm_response(msg=msg)
print(output)
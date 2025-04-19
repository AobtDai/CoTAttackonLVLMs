from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

model_path = "../models/Qwen2.5-VL-32B-Instruct"
# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     model_path, torch_dtype="auto", device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2",
    device_map="auto",
    # load_in_8bit=True,
)

# default processer
processor = AutoProcessor.from_pretrained(model_path)

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)
for i in range(1, 2):
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "image",
                    "image": "./Overview.png",
                },
                {
                    "type": "text", 
                    "text": '''
                        You are a helpful robot assistant located in the Stage in the south. 
                        You have access to an input image that represents the layout of the house. 
                        Users will provide you with a series of instructions. 
                        Your responses should be structured in the following format, 
                        which includes both a reasoning section and a sequence of actions:
                        
                        <Reasoning>
                        Here, you should provide a detailed explanation of the steps required to complete the task, 
                        considering the layout of the house.
                        </Reasoning>

                        <Action_1>
                        {action: "[specific action]", target: "[location / object]"}
                        </Action_1>

                        <Action_2>
                        {action: "[specific action]", target: "[location / object]"}
                        </Action_2>

                        ...

                        <Action_i>
                        {action: "[specific action]", target: "[location / object]"}
                        </Action_i>

                        ...

                        <Action_n>
                        {action: "[specific action]", target: "[location / object]"}
                        </Action_n>
                        '''
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "Please help me get my black bottle, I think it's in the Open Workshop Area at LFT. Then give this bottle to Mr. Qiu, I think he is in the meeting room, next to th Relax room. Finally, please help me get my laptop which is at Mr. Qiu's place."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=10000)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)

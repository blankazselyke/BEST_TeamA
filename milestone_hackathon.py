from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
import cv2
from PIL import Image
import numpy as np

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct",
#     torch_dtype=torch.bfloat16,
#     #attn_implementation="flash_attention_2",
#     device_map="auto",
# )



def extract_video_frames(video_path, target_fps=2):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps // target_fps)

    frames = []
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            # Convert BGR (OpenCV) to RGB (PIL)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            frames.append(pil_image)
        frame_count += 1

    cap.release()
    return frames

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

video_frames = extract_video_frames("2018-03-13.17-20-14.17-21-19.school.G421.r13.avi", target_fps=10)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_frames,  # list of PIL images
            },
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
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
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)

# save the output in a json file
# Save to file
with open("output.txt", "w", encoding="utf-8") as f:
    for line in output_text:
        f.write(line + "\n")

# Optional: Also print
print("Saved output to output_7b.txt")

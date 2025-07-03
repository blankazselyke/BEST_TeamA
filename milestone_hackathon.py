from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os


VIDEO_PATH = "2018-03-13.17-20-14.17-21-19.school.G421.r13.avi"
TARGET_FPS = 0.2 # Set to 0.2 to extract one frame per 5 seconds

torch.cuda.empty_cache()

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

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()

    if success:
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image)
        cap.release()
        return pil_image
    else:
        cap.release()
        return None


def extract_video_frames(video_path, target_fps=2.0):
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(video_fps // target_fps)

    frames = []
    prev_hist = None
    count = 0

    while True:
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
    print(f"Extracted {len(frames)} unique frames.")
    return frames

# default processer
#processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
min_pixels = 256*28*28
max_pixels = 1280*28*28
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)


video_frames = extract_video_frames(VIDEO_PATH, target_fps=TARGET_FPS)
print(f"Extracted {len(video_frames)} frames from the video.")

# Plot the extracted frames
def plot_video_frames(frames, max_frames=12):
    """Plot video frames in a grid layout"""
    num_frames = min(len(frames), max_frames)
    if num_frames == 0:
        print("No frames to plot")
        return
    
    # Calculate grid dimensions
    cols = min(4, num_frames)
    rows = (num_frames + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i in range(num_frames):
        axes[i].imshow(frames[i])
        axes[i].set_title(f'Frame {i+1}')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(num_frames, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('video_frames.png', dpi=150, bbox_inches='tight')
    plt.show()

# Plot the frames
plot_video_frames(video_frames)

#picture = extract_first_frame(VIDEO_PATH)

# Prepare prompt
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_frames,  # Use the extracted video frames as a list of PIL Images
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]


# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, _ = process_vision_info(messages)
inputs = processor(
    text=[prompt_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
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
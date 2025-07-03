from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

VIDEO_PATH = "2018-03-13.17-20-14.17-21-19.school.G421.r13.avi"
TARGET_FPS = 0.2
OUTPUT_FILENAME = "video_description.txt"
PLOT_FILENAME = "extracted_frames.png"
MAX_FRAMES = 50  # Optional: limit number of extracted frames

torch.cuda.empty_cache()

# Model & processor
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

print(f"Loading and quantizing model: {MODEL_NAME}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto"
)
model.eval()
print("Model loaded and set to eval mode.")

print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)

# --- Video Frame Extraction with Quality Filtering ---
def extract_video_frames(video_path, target_fps=0.2, max_frames=None):
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        print("[WARNING] Invalid FPS detected. Defaulting to 30.")
        video_fps = 30

    frame_interval = int(round(video_fps / target_fps))
    frames = []
    prev_hist = None
    count = 0

    while True:
        success, frame = cap.read()
        if not success:
            break
        if count % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist = cv2.normalize(hist, hist).flatten()

            # Avoid saving near-duplicate frames
            if prev_hist is None or cv2.compareHist(prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA) > 0.3:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                frames.append(pil_img)
                prev_hist = hist

            if max_frames and len(frames) >= max_frames:
                break
        count += 1

    cap.release()
    print(f"Extracted {len(frames)} unique frames.")
    return frames

def plot_and_save_frames(frames, filename="plot.png"):
    if not frames:
        print("[INFO] No frames to plot.")
        return

    fig, axes = plt.subplots(len(frames), 1, figsize=(8, 5 * len(frames)))
    if len(frames) == 1:
        axes = [axes]
    for i, (frame, ax) in enumerate(zip(frames, axes)):
        ax.imshow(frame)
        ax.set_title(f"Frame {i + 1}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved plot to {filename}")

# --- Main Inference Pipeline ---
video_frames = extract_video_frames(VIDEO_PATH, target_fps=TARGET_FPS, max_frames=MAX_FRAMES)
if not video_frames:
    print("[ERROR] No frames extracted. Exiting.")
    exit()

plot_and_save_frames(video_frames, filename=PLOT_FILENAME)

# Prepare prompt
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_frames,
            },
            {
                "type": "text",
                "text": (
                    "This is a surveillance video captured in a real-world environment. "
                    "Please analyze the sequence of frames and describe any unusual, suspicious, or unexpected activities you observe. "
                    "Be detailed and chronological in your description. If everything appears normal, state that explicitly.\n\n"
                    "Focus on:\n"
                    "- Strange movements or behaviors\n"
                    "- Unauthorized presence\n"
                    "- Fast motion, running, or fighting\n"
                    "- Dropped or abandoned objects\n"
                    "- Rule violations or anomalies\n"
                    "- Time progression and scene changes"
                ),
            },
        ],
    }
]


print("Preparing inputs for the model...")
prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[prompt_text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to(model.device)

# Free memory
del video_frames, messages, image_inputs, video_inputs
torch.cuda.empty_cache()

# Generate
print("Generating description...")
with torch.no_grad():
    output_ids = model.generate(**inputs, max_new_tokens=1024)

generated_ids_trimmed = [
    out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n--- Video Description ---")
print(output_text[0])
print("-------------------------")

# Save
with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
    f.write(output_text[0])
print(f"Saved description to {OUTPUT_FILENAME}")

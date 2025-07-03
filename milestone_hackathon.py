from transformers import Qwen2_5_VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


VIDEO_PATH = "2018-03-13.17-20-14.17-21-19.school.G421.r13.avi"
# Set to 0.2 to extract one frame every 5 seconds (1 / 5 = 0.2)
TARGET_FPS = 0.2
OUTPUT_FILENAME = "video_description.txt"
PLOT_FILENAME = "extracted_frames.png"

torch.cuda.empty_cache()

# --- Use consistent model and processor names ---
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"

# Load the model on the available device(s)
print(f"Loading model: {MODEL_NAME}...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME, torch_dtype="auto", device_map="auto"
)

# Use the processor that corresponds to the loaded model
print("Loading processor...")
processor = AutoProcessor.from_pretrained(MODEL_NAME)


def extract_video_frames(video_path, target_fps=0.2):
    """Extracts frames from a video at a specified target frame rate."""
    print(f"Opening video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps == 0:
        print("Warning: Video FPS is 0. Defaulting to 30 for interval calculation.")
        video_fps = 30
        
    frame_interval = int(round(video_fps / target_fps))
    print(f"Video FPS: {video_fps:.2f}, Target FPS: {target_fps:.2f}, Frame Interval: {frame_interval}")

    frames = []
    frame_count = 0
    success = True

    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_interval == 0:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            frames.append(pil_image)
        frame_count += 1

    cap.release()
    return frames


def plot_and_save_frames(frames, filename="plot.png"):
    """
    Plots a list of video frames vertically and saves the plot to a file.
    """
    num_frames = len(frames)
    if num_frames == 0:
        print("No frames to plot.")
        return

    fig_width = 8
    height_per_frame = 5
    fig, axes = plt.subplots(
        num_frames, 1, figsize=(fig_width, num_frames * height_per_frame)
    )

    if num_frames == 1:
        axes = np.array([axes])
        
    axes = axes.flatten()

    for i, frame in enumerate(frames):
        ax = axes[i]
        ax.imshow(frame)
        ax.set_title(f"Frame {i+1}", fontsize=12)
        ax.axis('off')

    plt.tight_layout(pad=2.0)
    
    # --- FIX: Save the plot instead of trying to show it ---
    plt.savefig(filename)
    plt.close() # Close the plot to free memory
    print(f"Plot of extracted frames saved to {filename}")


# --- Main script execution ---
video_frames = extract_video_frames(VIDEO_PATH, target_fps=TARGET_FPS)
print(f"Extracted {len(video_frames)} frames from the video.")

if not video_frames:
    print("No frames were extracted. Exiting.")
    exit()

# Plot and save the frames
plot_and_save_frames(video_frames, filename=PLOT_FILENAME)

# Prepare the prompt for the model
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_frames,
            },
            {"type": "text", "text": "Describe what is happening in this video in detail."},
        ],
    }
]

# Preparation for inference
print("Preparing inputs for the model...")
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# --- THE MAIN FIX IS HERE: Unpack only two values ---
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to(model.device)

# Inference: Generation of the output
print("Generating video description...")
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n--- Generated Description ---")
print(output_text[0])
print("---------------------------\n")

# Save to file with a consistent filename
with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
    f.write(output_text[0])

print(f"Saved output to {OUTPUT_FILENAME}")
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image, ImageDraw
import random

# 1. Setup
model_dir = "Qwen/Qwen2-VL-2B-Instruct"
adapter_path = "train_VL/qwen2-vl-2b-lora-vl-ts1-1ep/checkpoint-58"

print(f"Loading base model: {model_dir}")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="cuda"
)
processor = AutoProcessor.from_pretrained(model_dir)

print(f"Loading YOUR trained weights: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)
model.to("cuda")

# 2. GENERATE A LOCAL TEST IMAGE (No Internet Needed)
# We create a 512x512 gray background with a black "crack"
print("Generating local test image...")
image = Image.new('RGB', (512, 512), color=(200, 200, 200))
draw = ImageDraw.Draw(image)
# Draw a random 'crack'
start_pos = (150, 150)
end_pos = (350, 350)
draw.line([start_pos, end_pos], fill=(0, 0, 0), width=15)
draw.line([(150, 150), (200, 250)], fill=(0, 0, 0), width=10)

# 3. Ask the Model
prompts = [
    "Describe this image.",
    "Is there an anomaly in this image?",
    "Describe the defect."
]

print("\n" + "="*20 + " ZERO-SHOT TRANSFER TEST " + "="*20)
for p in prompts:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": p}
            ],
        }
    ]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=image,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
    
    output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True
    )[0].split("assistant\n")[-1].strip()
    
    print(f"\nPROMPT: {p}")
    print(f"OUTPUT: {output_text}")

print("="*65)

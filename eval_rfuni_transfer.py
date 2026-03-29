import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
import requests
from io import BytesIO

# 1. Setup
model_dir = "Qwen/Qwen2-VL-2B-Instruct"
# POINT TO YOUR SAVED CHECKPOINT
adapter_path = "train_VL/qwen2-vl-2b-lora-vl-ts1-1ep/checkpoint-58"

print(f"Loading base model: {model_dir}")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="cuda"
)
processor = AutoProcessor.from_pretrained(model_dir)

print(f"Loading YOUR trained weights: {adapter_path}")
model = PeftModel.from_pretrained(model, adapter_path)
model.to("cuda")

# 2. Get a "General" Anomaly Image (Simulating RFUni/MVTec)
# This is a standard 'Hazelnut with crack' image URL
img_url = "https://mvtec.org/fileadmin/content/news/2021/2021-04-20_MVTec_AD_Hazelnut_Crack.png"
print(f"Downloading generic anomaly image from: {img_url}")
response = requests.get(img_url)
image = Image.open(BytesIO(response.content))

# 3. Ask the Model
# We use a generic prompt to see if it defaults to "time series" language
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
    )[0].split("assistant\n")[-1].strip() # Clean up output
    
    print(f"\nPROMPT: {p}")
    print(f"OUTPUT: {output_text}")

print("="*65)

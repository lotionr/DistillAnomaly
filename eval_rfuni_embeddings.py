import torch
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image, ImageDraw
import numpy as np

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

# 2. GENERATE TEST IMAGES
print("Generating Synthetic Test Data...")
img_normal = Image.new('RGB', (512, 512), color=(200, 200, 200))
img_anomaly = img_normal.copy()
draw = ImageDraw.Draw(img_anomaly)
draw.line([(150, 150), (350, 350)], fill=(0, 0, 0), width=15)

# 3. DEFINE FEATURE EXTRACTION
def get_visual_embedding(image, text_prompt="Describe this image"):
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text_prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=image,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Get last layer hidden states
        last_hidden_state = outputs.hidden_states[-1]
        # Mean pooling to get one vector per image
        embedding = last_hidden_state.mean(dim=1)
        
    return embedding

# 4. RUN EXTRACTION
print("Extracting features for NORMAL image...")
emb_normal = get_visual_embedding(img_normal)

print("Extracting features for ANOMALY image...")
emb_anomaly = get_visual_embedding(img_anomaly)

# 5. CALCULATE DISTANCE
# FIX: Flatten both to ensure they are 1D vectors [1536] instead of [1, 1536]
vec_normal = emb_normal.flatten()
vec_anomaly = emb_anomaly.flatten()

similarity = F.cosine_similarity(vec_normal, vec_anomaly, dim=0)
distance = 1.0 - similarity.item()

print("\n" + "="*20 + " RFUni FEATURE ANALYSIS " + "="*20)
print(f"Cosine Similarity: {similarity.item():.4f}")
print(f"Feature Distance:  {distance:.4f}")
print("-" * 30)

if distance > 0.001:
    print("RESULT: SUCCESS. The model sees a DIFFERENCE.")
    print(f"Interpretation: The internal representation shifted by {distance:.4f}.")
    print("This confirms the visual encoder IS sensitive to the defect.")
else:
    print("RESULT: FAILURE. The model sees them as identical.")
print("="*54)

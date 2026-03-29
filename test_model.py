import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from PIL import Image

# 1. Load Base Model
# We explicitly force device_map="cuda" to ensure it lands on the GPU
model_dir = "Qwen/Qwen2-VL-2B-Instruct"
print(f"Loading base model from {model_dir}...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="cuda"
)
processor = AutoProcessor.from_pretrained(model_dir)

# 2. Load Adapter & FORCE IT TO GPU
# Note: We assume the checkpoint-58 path is correct based on your logs
adapter_path = "train_VL/qwen2-vl-2b-lora-vl-ts1-1ep/checkpoint-58"
print(f"Loading LoRA adapter from {adapter_path}...")

# Use PeftModel wrapper which handles device placement better
model = PeftModel.from_pretrained(model, adapter_path)
model.to("cuda")  # <--- THIS FIXES THE RUNTIME ERROR

# 3. Load Image
image_path = "all_data/synthetic/trend/train/figs/31.png"
print(f"Testing on image: {image_path}")
image = Image.open(image_path)

# 4. Prepare Input
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": "Describe this time series image."}
        ],
    }
]

# 5. Inference
print("Generating response...")
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
).to("cuda")

# Generate
with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=128)

generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

print("\n" + "="*20 + " MODEL OUTPUT " + "="*20)
print(output_text[0])
print("="*54)

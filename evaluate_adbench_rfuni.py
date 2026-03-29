import os
import torch
import numpy as np
import torch.nn.functional as F
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

# 1. Load Model
model_dir = "Qwen/Qwen2-VL-2B-Instruct"
adapter_path = "train_VL/qwen2-vl-2b-lora-vl-ts1-1ep/checkpoint-58"

print("Loading model and weights...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_dir, torch_dtype="auto", device_map="cuda"
)
processor = AutoProcessor.from_pretrained(model_dir)
model = PeftModel.from_pretrained(model, adapter_path)
model.to("cuda")
model.eval()

# 2. Define Feature Extraction
def get_embedding(img_path):
    image = Image.open(img_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Analyze this data."}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=image, padding=True, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        # Extract last hidden state, mean pool, and flatten
        embedding = outputs.hidden_states[-1].mean(dim=1).flatten()
    return embedding

# 3. Load Paths
normal_dir = "adbench_images/normal"
anomaly_dir = "adbench_images/anomaly"

normal_files = [os.path.join(normal_dir, f) for f in os.listdir(normal_dir) if f.endswith('.png')]
anomaly_files = [os.path.join(anomaly_dir, f) for f in os.listdir(anomaly_dir) if f.endswith('.png')]

# Unsupervised Setup: Use a subset of normal data to build the "Baseline Profile"
# We use the first 40 normal images as the "training" reference, just like ADBench unsupervised protocols
ref_files = normal_files[:40]
test_normal_files = normal_files[40:]
test_anomaly_files = anomaly_files

# 4. Extract Reference Embeddings
print(f"Extracting baseline profile from {len(ref_files)} normal samples...")
ref_embeddings = []
for f in ref_files:
    ref_embeddings.append(get_embedding(f))
# Calculate the "centroid" of normal behavior
centroid = torch.stack(ref_embeddings).mean(dim=0)

# 5. Score the Test Set
print(f"Scoring {len(test_normal_files)} normal and {len(test_anomaly_files)} anomaly test samples...")
y_true = []
y_scores = []

# Score Normal Test Samples (Label = 0)
for f in test_normal_files:
    emb = get_embedding(f)
    # Distance from centroid is the anomaly score
    distance = 1.0 - F.cosine_similarity(emb, centroid, dim=0).item()
    y_true.append(0)
    y_scores.append(distance)

# Score Anomaly Test Samples (Label = 1)
for f in test_anomaly_files:
    emb = get_embedding(f)
    distance = 1.0 - F.cosine_similarity(emb, centroid, dim=0).item()
    y_true.append(1)
    y_scores.append(distance)

# 6. Calculate RFuni/ADBench Metrics
auc_roc = roc_auc_score(y_true, y_scores)
auc_pr = average_precision_score(y_true, y_scores)

print("\n" + "="*40)
print(" RFuni / ADBench REPLICATION RESULTS")
print("="*40)
print(f"ROC AUC Score: {auc_roc:.4f}")
print(f"PR AUC Score:  {auc_pr:.4f}")
print("="*40)

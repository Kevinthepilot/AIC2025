from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import numpy as np
import torch
import pickle

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

frame_folder = r"frames"
output_pickle = os.path.join(frame_folder, "frame_vectors.pkl")

def encode_image(img_path):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.squeeze().cpu().numpy()

all_vectors = []
all_names = []

for file in sorted(os.listdir(frame_folder)):
    if file.endswith(".jpg"):
        full_path = os.path.join(frame_folder, file)
        print("Encoding:", file)
        vec = encode_image(full_path)
        all_vectors.append(vec)
        all_names.append(file)


with open(output_pickle, "wb") as f:
    pickle.dump((np.stack(all_vectors), all_names), f)

print("vector frame saved:", output_pickle)

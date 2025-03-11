import os
import zipfile
import collections
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# Define dataset paths
dataset_zip = "flickr8k.zip"  # Update with the actual path
dataset_folder = "Flickr8k"
captions_file = "Flickr8k_text/Flickr8k.token.txt"

# Extract dataset if not already extracted
if not os.path.exists(dataset_folder):
    with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(dataset_folder)
    print("Dataset extracted!")

# Load captions
image_captions = collections.defaultdict(list)
with open(captions_file, 'r') as f:
    for line in f:
        image, caption = line.strip().split('\t')
        image = image.split('#')[0]
        image_captions[image].append(caption)

# Dummy model for caption generation (Replace with actual model)
def generate_caption(image):
    return "This is a sample caption. Replace with a trained model output."

# Gradio UI
def predict(image):
    caption = generate_caption(image)
    return caption

demo = gr.Interface(fn=predict, inputs="image", outputs="text")
demo.launch()

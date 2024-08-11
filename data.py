### Install requirements ###
!pip install --upgrade diffusers[torch]
!pip install transformers

### Create image generation pipeline ###

from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipeline.to("cuda")

### Generate images ###

import random
import os

import matplotlib.pyplot as plt

# Define directories for different dog emotions in various environments
dog_emotions = [
    'happy', 
    'sad', 
    'angry', 
    'surprised', 
    'relaxed', 
    'anxious'
]

environments = ['living_room', 'garden', 'park', 'beach', 'forest', 'mountain']

# Define the base directory on your local device where images will be saved
base_dir = 'C:/Users/YourUsername/Documents/dog_emotions'  # Replace with your desired path

# Create directories for each emotion and environment
for emotion in dog_emotions:
    for environment in environments:
        os.makedirs(f'{base_dir}/{emotion}/{environment}', exist_ok=True)

# Define the prompts for each type of dog emotion
dog_emotion_prompts = {
    'happy': 'a joyful dog wagging its tail, with a big smile',
    'sad': 'a sad dog with drooping ears and big teary eyes',
    'angry': 'an angry dog with bared teeth and glaring eyes',
    'surprised': 'a surprised dog with wide open eyes and raised ears',
    'relaxed': 'a relaxed dog lying down with eyes half-closed and a gentle expression',
    'anxious': 'an anxious dog with ears down and tail tucked, looking around nervously'
}

# Loop to generate images for each dog emotion in different environments
for emotion, prompt in dog_emotion_prompts.items():
    for environment in environments:
        for j in range(25):  # Number of images per dog emotion per environment
            full_prompt = '{} in a {}, realistic environment, photorealistic, hyperrealistic, high detail, digital art, 8k resolution'.format(prompt, environment)
            negative_prompt = '3d, cartoon, anime, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ' + \
                              '((grayscale)) Low Quality, Worst Quality, plastic, fake, disfigured, deformed, blurry, bad anatomy, blurred, watermark, grainy, signature'
            
            img = pipeline(full_prompt, negative_prompt=negative_prompt).images[0]

            img.save(f'{base_dir}/{emotion}/{environment}/{emotion}_dog_{str(j).zfill(4)}.png')


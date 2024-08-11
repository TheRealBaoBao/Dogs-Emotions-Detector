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

# Define directories for different categories of pet emotions in various environments
pet_categories = [
    'happy', 
    'sad', 
    'angry', 
    'surprised', 
    'relaxed', 
    'anxious'
]

environments = ['living_room', 'garden', 'park', 'beach', 'forest', 'mountain']

# Define the base directory on your local device where images will be saved
base_dir = 'C:/Users/YourUsername/Documents/pet_emotions'  # Replace with your desired path

# Create directories for each category and environment
for category in pet_categories:
    for environment in environments:
        os.makedirs(f'{base_dir}/{category}/{environment}', exist_ok=True)

# Define the prompts for each type of pet emotion
pet_emotion_prompts = {
    'happy_dog': 'a joyful dog wagging its tail, with a big smile',
    'sad_cat': 'a sad cat with drooping ears and big teary eyes',
    'angry_bird': 'an angry bird with fluffed up feathers and glaring eyes',
    'surprised_rabbit': 'a surprised rabbit with wide open eyes and raised ears',
    'relaxed_cat': 'a relaxed cat lying down with eyes half-closed and a gentle purr',
    'anxious_dog': 'an anxious dog with ears down and tail tucked, looking around nervously'
    # Add more prompts as needed for other pets and emotions
}

# Loop to generate images for each pet emotion in different environments
for pet_name, pet_prompt in pet_emotion_prompts.items():
    for environment in environments:
        for j in range(25):  # Number of images per pet type per environment
            prompt = '{} in a {}, realistic environment, photorealistic, hyperrealistic, high detail, digital art, 8k resolution'.format(pet_prompt, environment)
            negative_prompt = '3d, cartoon, anime, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ' + \
                              '((grayscale)) Low Quality, Worst Quality, plastic, fake, disfigured, deformed, blurry, bad anatomy, blurred, watermark, grainy, signature'
            
            img = pipeline(prompt, negative_prompt=negative_prompt).images[0]

            category = pet_name.split('_')[0]  # Use the first part of the pet name as the category
            img.save(f'{base_dir}/{category}/{environment}/{pet_name}_{str(j).zfill(4)}.png')

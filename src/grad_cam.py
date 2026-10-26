from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import cv2
from PIL import Image

import torch
from torchvision import models

import urllib.request

# For saving images
import sys
sys.path.append('../')

from utils.visualise import display_imagenet_output
from utils.datasets import preprocess_imagenet_image

# Download example image
import sys
sys.path.append('../')

from utils.download import save_image

# Load alternative sample image from the web
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a3/Hammerhead_shark%2C_Cocos_Island%2C_Costa_Rica.jpg/960px-Hammerhead_shark%2C_Cocos_Island%2C_Costa_Rica.jpg"
save_image(img_url, "sample_image.jpg",crop_to_square=False,resize=False)

img_path = "sample_image.jpg"
img = Image.open(img_path).convert("RGB")
img = img.crop((40, 0, 801, 761))

plt.imshow(img)
plt.axis("off")

# Load a sample image from the web
img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a1/Male_whale_shark_at_Georgia_Aquarium.jpg/960px-Male_whale_shark_at_Georgia_Aquarium.jpg"
save_image(img_url, "sample_image.jpg",crop_to_square=False,resize=False)


img = Image.open(img_path).convert("RGB")
img = img.crop((320, 0, 960, 640))  # Crop to 640x640

plt.imshow(img)
plt.axis("off")



# Load the pre-trained model (e.g., VGG16)
model = models.vgg16(pretrained=True)

# Set the model to gpu
device = torch.device('mps' if torch.backends.mps.is_built()
                      else 'cuda' if torch.cuda.is_available()
                      else 'cpu')
model.to(device)

# Set the model to evaluation mode
model.eval()
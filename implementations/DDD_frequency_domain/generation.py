import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import csv
import datetime
from io import BytesIO
from typing import Callable
from typing import List
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision.transforms as T
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image, ImageOps
from tqdm import tqdm

to_pil = T.ToPILImage()

import argparse

import open_clip
from ddd import *
from torchvision import transforms
from utils import *
from utils_text import *

model_version = "stabilityai/stable-diffusion-2-inpainting"
pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    model_version,
    variant="fp16",
    torch_dtype=torch.float32,
)
pipe_inpaint = pipe_inpaint.to("cuda")
pipe_inpaint.safety_checker = None

for name, param in pipe_inpaint.unet.named_parameters():
  param.requires_grad = False

file_iteration_names = ["008"]

for file_name in file_iteration_names:

  init_image = Image.open(f'./images/{file_name}.png').convert('RGB').resize(
      (512, 512)
  )
  mask_image = Image.open(f'./images/{file_name}_masked.png').convert('RGB')
  mask_image = ImageOps.invert(mask_image).resize((512, 512))
  adv_image = Image.open(f'./adversarial/{file_name}_adv_compressed.png')

  prompts = ['Two people on an airplane', 'Two people in front of the eiffel tower', 'Two people on a pier']

  SEED = 1007

  with torch.no_grad():
    for prompt in prompts:

      print(SEED)

      torch.manual_seed(SEED)
      strength = 0.8
      guidance_scale = 7.5

      num_inference_steps = 50

      image_nat = pipe_inpaint(
          prompt=prompt,
          image=init_image,
          mask_image=mask_image,
          eta=1,
          num_inference_steps=num_inference_steps,
          guidance_scale=guidance_scale,
          strength=strength
      ).images[0]
      image_nat = recover_image(image_nat, init_image, mask_image)

      torch.manual_seed(SEED)
      image_adv = pipe_inpaint(
          prompt=prompt,
          image=adv_image,
          mask_image=mask_image,
          eta=1,
          num_inference_steps=num_inference_steps,
          guidance_scale=guidance_scale,
          strength=strength
      ).images[0]

      image_adv = recover_image(image_adv, init_image, mask_image)

      image_adv.save(f'{file_name}_adv_compressed_{prompt}.png')
      fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))

      ax[0].imshow(init_image)
      ax[1].imshow(adv_image)
      ax[2].imshow(image_nat)
      ax[3].imshow(image_adv)

      ax[0].set_title('Source Image', fontsize=16)
      ax[1].set_title('Adv Image', fontsize=16)
      ax[2].set_title('Gen. Image Nat.', fontsize=16)
      ax[3].set_title('Gen. Image Adv.', fontsize=16)

      for i in range(4):
        ax[i].grid(False)
        ax[i].axis('off')

      fig.suptitle(f"{prompt} - {SEED}", fontsize=20)
      fig.tight_layout()
      plt.savefig(f'{file_name}_result_compressed_{prompt}.png')
      plt.show()

      plt.clf()

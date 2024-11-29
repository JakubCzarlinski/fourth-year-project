import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from PIL import Image
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np

import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import torchvision.transforms as T
from typing import Union, List, Optional, Callable
import csv
import datetime
to_pil = T.ToPILImage()

from utils import *

import open_clip
from utils_text import *
from torchvision import transforms
import argparse
from ddd import *



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

file_iteration_names = [str(i) for i in range(1, 101) if i==10]

for file_name in file_iteration_names:

    init_image = Image.open(f'./dataset/{file_name}.png').convert('RGB').resize((512,512))
    mask_image = Image.open(f'./dataset/{file_name}_masked.png').convert('RGB')
    mask_image = ImageOps.invert(mask_image).resize((512,512))
    adv_image =  Image.open(f'./dataset_adv/{file_name}_adv.png')


    prompts = ['change the background of the image to a beach']

    SEED = 1007

    with torch.no_grad():
        for prompt in prompts:

            print(SEED)

            torch.manual_seed(SEED)
            strength = 0.8
            guidance_scale = 7.5

            num_inference_steps = 50

            image_nat = pipe_inpaint(prompt=prompt,
                                image=init_image,
                                mask_image=mask_image,
                                eta=1,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                strength=strength
                                ).images[0]
            image_nat = recover_image(image_nat, init_image, mask_image)

            torch.manual_seed(SEED)
            image_adv = pipe_inpaint(prompt=prompt,
                                image=adv_image,
                                mask_image=mask_image,
                                eta=1,
                                num_inference_steps=num_inference_steps,
                                guidance_scale=guidance_scale,
                                strength=strength
                                ).images[0]

            image_adv = recover_image(image_adv, init_image, mask_image)

            image_adv.save(f'dataset_results/{file_name}_adv.png')
            image_nat.save(f'{file_name}_inpaint.png')


            fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20,6))

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
            plt.savefig(f'dataset_results/{file_name}_result.png')
            plt.show()
            
            plt.clf()
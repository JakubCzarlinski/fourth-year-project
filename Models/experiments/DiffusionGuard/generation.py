import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from PIL import Image, ImageOps
import requests
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms as tfms
totensor = tfms.ToTensor()
import torch
import requests
from tqdm import tqdm
from io import BytesIO
from diffusers import StableDiffusionInpaintPipeline
import torchvision.transforms as T
from typing import Union, List, Optional, Callable
import csv
import datetime
topil = tfms.ToPILImage()
to_pil = T.ToPILImage()
def recover_image(image, init_image, mask, background=False):
    image = totensor(image)
    mask = totensor(mask)
    init_image = totensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return topil(result)

import open_clip
from torchvision import transforms
import argparse



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

file_iteration_names = ["2"]

for file_name in file_iteration_names:

    init_image = Image.open(f'./assets/{file_name}.png').convert('RGB').resize((512,512))
    mask_image = Image.open(f'./assets/{file_name}_masked.png').convert('RGB')
    mask_image = ImageOps.invert(mask_image).resize((512,512))
    adv_image =  Image.open(f'protected_images/exp/2/adv_image_compressed_60.png')

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

            image_adv.save(f'{file_name}_adv_inpaint_60.png')
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
            plt.savefig(f'{file_name}_result.png')
            plt.show()
            
            plt.clf()
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from PIL import ImageOps
from torchvision import transforms

totensor = transforms.ToTensor()
topil = transforms.ToPILImage()

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = dtype = torch.float16

def prepare_masks(orig_images, mask_image):
    """
    Prepare the mask and masked image.
    """
    cur_mask, cur_masked_image, init_image = prepare_mask_and_masked2(
        orig_images, mask_image, no_mask=False, inverted=True
    )
    return cur_mask.to(device, dtype), cur_masked_image.to(device, dtype)

def process_images(pipe_inpaint, orig_images, cur_mask, cur_masked_image, size):
    """
    Preprocess the image and encode to latent space.

    """
    # Preprocess the image
    preprocess_transform = transforms.Compose(
        [
            transforms.Resize(
                size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
        ]
    )
    # Resize the image to correct size.
    with torch.no_grad():
        curr_images = preprocess_transform(orig_images).to(device)
        mask = F.interpolate(cur_mask, size=(size // 8, size // 8))

        if len(curr_images.shape) == 3:
            curr_images = curr_images.unsqueeze(0)
        elif len(curr_images.shape) == 5:
            curr_images = curr_images.squeeze(0)
        # Encode the image to latent space
        all_latents = encode_latents(pipe_inpaint, curr_images)
        masked_image_latents = encode_latents(pipe_inpaint, cur_masked_image)

        return all_latents, masked_image_latents, mask

def encode_latents(pipe_inpaint, image):
    # Encode the image to latent space
    return pipe_inpaint.vae.encode(image.to(dtype)).latent_dist.sample() * 0.18215

def get_embeddings(pipe_inpaint, testing_filename):
    # Get the text embeddings
    gt_embeddings = get_text_embedding(pipe_inpaint, testing_filename)
    uncond_embeddings = get_text_embedding(pipe_inpaint, "")

    return gt_embeddings, uncond_embeddings

def load_images(image_folder, filename, image_size_2d):
  # Load the original image and the masked image
  original_image = Image.open(f"{image_folder}original/{filename}.png").convert('RGB').resize(image_size_2d)
  masked_image_original = Image.open(f"{image_folder}masks/{filename}_masked.png").convert('RGB').resize(image_size_2d)
  masked_image = ImageOps.invert(masked_image_original).resize(image_size_2d)
  return original_image, masked_image_original, masked_image


def pil_to_latent(pipe, input_im):
  # Conver PIL image to latent space
  with torch.no_grad():
    latent = pipe.vae.encode(
        totensor(input_im).to(pipe.vae.dtype).unsqueeze(0).to(device) * 2 - 1
    )
  return 0.18215 * latent.latent_dist.mode()


def text_embedding(pipe, prompt):
  # Get the text embedding
  # Tokenize the prompt and get the text embeddings
  text_inputs = pipe.tokenizer(
      prompt,
      padding="max_length",
      max_length=pipe.tokenizer.model_max_length,
      return_tensors="pt",
  )
  text_input_ids = text_inputs.input_ids
  # Get the text embeddings
  text_embeddings = pipe.text_encoder(text_input_ids.to(device))[0]
  # Get the unconditioned embeddings
  uncond_tokens = [""]
  max_length = text_input_ids.shape[-1]
  # Tokenize the unconditioned prompt
  uncond_input = pipe.tokenizer(
      uncond_tokens,
      padding="max_length",
      max_length=max_length,
      truncation=True,
      return_tensors="pt",
  )
  # Get the unconditioned embeddings
  uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(device))[0]
  seq_len = uncond_embeddings.shape[1]

  text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
  text_embeddings = text_embeddings.detach()

  return text_embeddings

def sample(
    self,
    text_embeddings,
    masked_images: torch.FloatTensor | Image.Image,
    mask: torch.FloatTensor | Image.Image,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    source_latent=None,
    current_t: int = None,
    all_latents=False
):
  mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
  mask = torch.cat([mask] * 2)

  self.scheduler.set_timesteps(num_inference_steps)
  # timesteps_tensor = [current_t-i for i in range(num_inference_steps)]
  timesteps = self.scheduler.timesteps
  if source_latent is not None:
    latents = source_latent
  latent_list = []
  # For each timestep, sample the noise
  for i, t in enumerate(timesteps):
    #print(t)
    masked_image = masked_images[i][None]
    # Encode to latent space
    masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
    masked_image_latents = 0.18215 * masked_image_latents
    masked_image_latents = torch.cat([masked_image_latents] * 2)

    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = torch.cat(
        [latent_model_input, mask, masked_image_latents], dim=1
    )
    # Predict the noise
    noise_pred = self.unet(
        latent_model_input, t, encoder_hidden_states=text_embeddings
    ).sample
    # Apply guidance scale to the noise prediction
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )
    latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample
    if all_latents:
      latent_list.append(latents)
  latents = 1 / 0.18215 * latents
  image = self.vae.decode(latents).sample
  if all_latents:
    return image, latent_list
  else:
    return image, None


def recover_image(image, init_image, mask, background=False):
  # Recover the image
  image = totensor(image)
  mask = totensor(mask)
  init_image = totensor(init_image)
  if background:
    result = mask * init_image + (1 - mask) * image
  else:
    result = mask * image + (1 - mask) * init_image
  return topil(result)


def prepare_mask_and_masked(image, mask):
  """
  Prepare the images.
  """
  image = np.array(image.convert("RGB"))
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0
  mask = np.array(mask.convert("L"))
  mask = mask.astype(np.float32) / 255.0
  mask = mask[None, None]
  mask[mask < 0.5] = 0
  mask[mask >= 0.5] = 1
  mask = torch.from_numpy(mask)
  masked_image = image * (mask < 0.5)

  return mask, masked_image


def load_prompt(path):
  # Load prompt from file
  prompts = []
  with open(path, 'r') as f:
    for line in f:
      prompts.append(line)
  return prompts

def get_text_embedding(self, prompt):
  # Get the text embedding for the given prompt
  # Tokenize the prompt and get the text embeddings
  text_input_ids = self.tokenizer(
      prompt,
      padding="max_length",
      truncation=True,
      max_length=self.tokenizer.model_max_length,
      return_tensors="pt",
  ).input_ids
  text_embeddings = self.text_encoder(text_input_ids.to(device))[0]
  return text_embeddings


def prepare_mask_and_masked2(image, mask, no_mask=False, inverted=False):
  # Prepare the images.
  image = np.array(image.convert("RGB"))
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

  mask = np.array(mask.convert("L"))
  mask = mask.astype(np.float32) / 255.0
  mask = mask[None, None]
  mask[mask < 0.5] = 0
  mask[mask >= 0.5] = 1
  if inverted:
    mask[mask >= 1] = 2
    mask[mask <= 1] = 1
    mask[mask == 2] = 0
  if no_mask:
    mask[mask >= 0] = 1
  mask = torch.from_numpy(mask)
  masked_image = image * (mask < 0.5)

  return mask, masked_image, image

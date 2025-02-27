from typing import Union

import numpy as np
import torch
import utils
import xformers.ops as xops
from diffusers import StableDiffusionInpaintPipeline
from diffusers import UNet2DConditionModel
from diffusers.models.attention import Attention
from PIL import Image
from torch import nn
from torchmetrics.image.ssim import (
    MultiScaleStructuralSimilarityIndexMeasure as MSSIM
)
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure as SSIM
from torchvision import transforms
from tqdm import tqdm
import random
to_pil = transforms.ToPILImage()
from diff_jpeg import DiffJPEGCoding
diff_jpeg_coding_module = DiffJPEGCoding()


class AttnController:

  def __init__(
      self,
      post=False,
      mask=None,
      switch=True,
      criteria='MSE',
      target_depth=[256, 64]
  ) -> None:
    self.attn_probs = []
    self.logs = []
    self.post = post
    self.rand_target = None
    self.rand_targets = None
    self.lamb = 0.5
    self.masks = {'origin': mask}
    self.targets = []
    self.sources = []
    self.m_name = []
    self.criteria_name = criteria
    self.target_depth = target_depth
    self.temp = 0.1
    self.layer_weights = {4096:0.2, 1024:1, 1007:3, 256: 2, 64: 0.5}
    self.l2weight = 0.1

    if criteria == 'MSE':
      self.criteria = nn.MSELoss()
    elif criteria == 'COS' or criteria == 'COS_NORMED':
      cos_sim = nn.CosineSimilarity(dim=-1, eps=0.00001)
      self.criteria = lambda x, y: ((1 - cos_sim.forward(x, y))/self.temp).mean()
      # self.criteria = nn.CosineSimilarity(dim=-1, eps=0.00001)
    elif criteria == 'L1':
      self.criteria = nn.L1Loss()
    elif criteria == 'SSIM':
      raise NotImplementedError("SSIM is not implemented")
      # ssim = SSIM().to("cuda")
      # self.criteria = lambda x, y: ssim(x.unsqueeze(0), y.unsqueeze(0))
    elif criteria == 'MSSIM':
      self.criteria = MSSIM()
    self.switch = switch
    self.target_hidden = None

  def register_mask(self, hidden_shape):
    if hidden_shape not in self.masks:
      w_h = torch.sqrt(torch.tensor([hidden_shape])).int().item()

      new_mask = torch.nn.functional.interpolate(
          self.masks["origin"], size=(w_h, w_h)
      )
      new_mask = new_mask.flatten().unsqueeze(0).unsqueeze(2)

      if self.criteria_name == 'COS' or self.criteria_name == 'COS_NORMED':
        new_mask[new_mask == 0] = 0.01

      self.masks[hidden_shape] = new_mask

  def __call__(self, hidden, m_name):
    if self.switch:
      hidden = hidden.clone()

      uncond, cond = hidden.chunk(2)
      b, h, c = uncond.shape
      # print(h)

      self.register_mask(h)
      if h in self.target_depth:
        self.sources.append(uncond)
        self.targets.append(cond)
        self.m_name.append(m_name)

  def set_concept_positions(self, concept_positions):
    self.concept_positions = concept_positions

  @torch.compiler.disable
  def loss(self, loss_mask: bool, loss_depth: list[int] = [64]):

    losses = 0
    cur_loss = 0
    if self.target_hidden is not None:
      source = self.target_hidden
    else:
      source = self.sources

    for i, (a, b) in enumerate(zip(source, self.targets)):
      ba, h, c = a.shape
      print(h)
      loss = 0
      if loss_mask:
        if self.criteria_name == 'COS_NORMED':
          a_normed = torch.nn.functional.normalize(a, p=2, dim=-1)
          b_normed = torch.nn.functional.normalize(b, p=2, dim=-1)
          a_normed = a_normed * self.masks[h]
          b_normed = b_normed * self.masks[h]
        a = a * self.masks[h]
        b = b * self.masks[h]
      if h in loss_depth:
        if self.criteria_name == 'COS_NORMED':
          if h in [64, 256]:
            cos_loss = -self.criteria(a_normed.detach(), b_normed)
            regularizer = -torch.norm(a-b, p=2) * self.l2weight
            loss = self.layer_weights[h] * (cos_loss + regularizer)
        # loss = cos_loss + regularizer
          else:
            loss = -self.criteria(a_normed.detach(), b_normed)     
        else:
          loss = -self.criteria(a.detach(), b)
        losses += loss

      cur_loss += loss
      if i % 16 == 15:
        cur_loss = 0
    self.sources = []
    self.targets = []
    self.m_name = []
    return losses

  def zero_attn_probs(self):
    self.sources = []
    self.targets = []


class MyCrossAttnProcessor:

  def __init__(
      self, attn_controller: "AttnController", module_name, post=False
  ) -> None:
    self.attn_controller = attn_controller
    self.module_name = module_name
    self.post = False

  def __call__(
      self,
      attn: Attention,
      hidden_states: torch.Tensor,
      encoder_hidden_states: torch.Tensor | None = None,
      attention_mask: torch.Tensor | None = None,
  ):
    batch_size, sequence_length, _ = hidden_states.shape
    attention_mask = attn.prepare_attention_mask(
        attention_mask, sequence_length, batch_size=batch_size
    )

    if encoder_hidden_states is None:
      encoder_hidden_states = hidden_states

    query = attn.to_q(hidden_states)
    key = attn.to_k(encoder_hidden_states)
    value = attn.to_v(encoder_hidden_states)

    # new
    query = attn.head_to_batch_dim(query).contiguous()
    key = attn.head_to_batch_dim(key).contiguous()
    value = attn.head_to_batch_dim(value).contiguous()

    hidden_states = xops.memory_efficient_attention(
        query, key, value, attn_bias=attention_mask
    )
    hidden_states = attn.batch_to_head_dim(hidden_states)

    hidden_states = attn.to_out[0](hidden_states)
    hidden_states = attn.to_out[1](hidden_states)

    self.attn_controller(hidden_states, self.module_name)
    return hidden_states

def get_random_t(t_schedule, t_schedule_bound):
  result = np.empty((0), int)
  for i in t_schedule:
    cur_t = np.clip(
        np.random.normal(i, 6), i - t_schedule_bound, i + t_schedule_bound
    )
    cur_t = int(cur_t)
    result = np.append(result, cur_t)
  return torch.tensor(result)


def get_random_emb(embs):
  rand_pos = np.random.randint(0, len(embs))
  return embs[rand_pos]


def attack_forward(
    self: StableDiffusionInpaintPipeline,
    text_embeddings: torch.FloatTensor,
    masked_image: Union[torch.FloatTensor, Image.Image],
    mask: Union[torch.FloatTensor, Image.Image],
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    eta: float = 0.0,
    random_t: torch.Tensor = None,
):
    latents_shape = (1, self.vae.config.latent_channels, height // 8, width // 8)
    latents = torch.randn(latents_shape, device=masked_image.device, dtype=text_embeddings.dtype)
    
    mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
    mask = torch.cat([mask] * 2)
    
    masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
    masked_image_latents *= 0.18215
    masked_image_latents = torch.cat([masked_image_latents] * 2)
    
    self.scheduler.set_timesteps(num_inference_steps)
    
    for t in random_t:
        noise_pred_cfg, noise_pred_text, noise_pred_uncond = pred_noise(
            unet=self.unet,
            text_embeddings=text_embeddings,
            latents=latents,
            mask=mask,
            masked_image_latents=masked_image_latents,
            t=t,
            guidance_scale=guidance_scale,
        )
        latents = self.scheduler.step(noise_pred_cfg, t, latents).prev_sample

    return latents, [noise_pred_text, noise_pred_uncond]

def pred_noise(
    unet: UNet2DConditionModel,
    text_embeddings: torch.FloatTensor,
    latents: torch.Tensor,
    mask: torch.Tensor,
    masked_image_latents: torch.Tensor,
    t: torch.Tensor,
    guidance_scale: float,
):
  latent_model_input = torch.cat([latents] * 2)
  latent_model_input = torch.cat(
      [latent_model_input, mask, masked_image_latents], dim=1
  )
  noise_pred = unet.forward(
      sample=latent_model_input,
      timestep=t,
      encoder_hidden_states=text_embeddings,
  ).sample
  noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
  noise_pred_cfg = noise_pred_uncond + guidance_scale * (
      noise_pred_text - noise_pred_uncond
  )
  return noise_pred_cfg, noise_pred_text, noise_pred_uncond

def get_gradient(
    cur_mask: torch.Tensor,
    cur_masked_image: torch.Tensor,
    text_embeddings: torch.Tensor,
    pipe: StableDiffusionInpaintPipeline,
    attn_controller: AttnController,
    loss_depth: list,
    loss_mask: bool,
    random_t: torch.Tensor,
    quality = 4,
    diffjpeg: bool = True
):
    torch.set_grad_enabled(True)
    cur_mask.requires_grad = False
    cur_masked_image.requires_grad_()

    # Zero the gradients
    cur_mask.grad = None
    cur_masked_image.grad = None
    
    if diffjpeg:
        compressed_image = apply_diffjpeg(cur_masked_image, quality)
    
        _image_nat, _latents = attack_forward(
            pipe,
            text_embeddings=text_embeddings,
            mask=cur_mask,
            masked_image=compressed_image,
            random_t=random_t,
        )
    else:
        _image_nat, _latents = attack_forward(
              pipe,
              text_embeddings=text_embeddings,
              mask=cur_mask,
              masked_image=cur_masked_image,
              random_t=random_t,
          )
    
    loss_value = attn_controller.loss(loss_mask, loss_depth)
    grad = torch.autograd.grad(loss_value, [cur_masked_image])[0] * (1 - cur_mask)
    return grad, loss_value.item()

def apply_diffjpeg(image, quality):
    jpeg_quality = torch.tensor([quality]).to(image.device)
    compressed_image = image.clone()
    compressed_image = (compressed_image / 2 + 0.5).clamp(0, 1) * 255
    compressed_image = diff_jpeg_coding_module(image_rgb=compressed_image, jpeg_quality=jpeg_quality)
    return ((torch.abs(compressed_image) / 255 - 0.5) * 2).clamp(-1, 1).to(torch.float16)

def disrupt(
    cur_mask: torch.Tensor,
    X: torch.Tensor,
    text_embeddings: torch.Tensor,
    step_size: float,
    iters: int,
    eps: float,
    clamp_min: float,
    clamp_max: float,
    attn_controller: AttnController,
    pipe: StableDiffusionInpaintPipeline,
    t_schedule: list[int],
    t_schedule_bound: int,
    loss_depth: list[int] = [64],
    loss_mask=False,
    grad_reps=5,
    diffjpeg=False
):
    X_adv = X.clone()
    iterator = tqdm(range(iters))
    total_losses = []
    x_dim = len(X.shape) - 1
    gen = torch.Generator(device='cuda')
    gen.manual_seed(1003)
    count = 0

    for j in iterator:
        random_t = get_random_t(t_schedule, t_schedule_bound)
        all_grads, value_losses = [], []
        text_embed = get_random_emb(text_embeddings)
        
        
        for _ in range(grad_reps):
            np.random.seed(count)
            quality = (count + np.random.randint(-5,5)) % 80 + 20
            count += 1
            c_grad, loss_value = get_gradient(
                cur_mask, 
                X_adv, 
                text_embed, 
                pipe, 
                attn_controller, 
                loss_depth, 
                loss_mask, 
                random_t, 
                quality, 
                diffjpeg
            )
            all_grads.append(c_grad.detach())
            value_losses.append(loss_value)
            attn_controller.zero_attn_probs()

        grad = torch.stack(all_grads).mean(dim=0)
        total_losses.append([np.mean(value_losses)])
        iterator.set_description_str(f'Loss: {total_losses[-1][0]:.5f}')

        X_adv.data = grad_to_adv(
          X=X,
          X_adv=X_adv,
          grad=grad,
          step_size=step_size,
          iters=iters,
          eps=eps,
          clamp_min=clamp_min,
          clamp_max=clamp_max,
          x_dim=x_dim,
          iteration=j,
      )

    torch.cuda.empty_cache()
    return X_adv, total_losses


def grad_to_adv(
    X: torch.Tensor,
    X_adv: torch.Tensor,
    grad: torch.Tensor,
    step_size: float,
    iters: int,
    eps: float,
    clamp_min: float,
    clamp_max: float,
    x_dim: int,
    iteration: int,
):
  grad_norm = torch.norm(
      grad.reshape(grad.shape[0], -1),
      dim=1,
  ).view(-1, *([1] * x_dim))
  grad_normalized = grad / (grad_norm + 1e-8)

  actual_step_size = step_size * (1 - iteration / iters)
  X_adv = X_adv - grad_normalized * actual_step_size

  d_x = X_adv - X.detach()
  d_x_norm = torch.renorm(d_x, p=2, dim=0, maxnorm=eps)

  return torch.clamp(X + d_x_norm, clamp_min, clamp_max)


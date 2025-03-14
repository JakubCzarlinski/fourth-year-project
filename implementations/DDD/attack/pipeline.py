"""DDD attack pipeline."""

import argparse
import random
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_inpaint import (
    StableDiffusionInpaintPipeline
)
from diffusers.schedulers.scheduling_utils import KarrasDiffusionSchedulers
from PIL import Image
from PIL import ImageOps
from sentence_transformers.util import dot_score
from sentence_transformers.util import semantic_search
from torchvision import transforms
from transformers import CLIPImageProcessor
from transformers import CLIPTextModel
from transformers import CLIPTokenizer
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPTextTransformer
from xformers.ops.fmha import attn_bias

to_tensor_func = transforms.ToTensor()


@dataclass(slots=True)
class TokenProjectiveEmbeddingArgs:
  prompt_len: int = 8
  iters: int = 350
  decreaser_lr_step: int = 340
  eval_step: int = 50
  lr: float = 0.001
  lower_lr: float = 0.0001
  batch_size: int = 1
  weight_decay: float = 0.1


@dataclass(slots=True)
class DDDAttackArgs:
  promt_len: int = 16
  iter: int = 3000
  lr: float = 0.1
  weight_decay: float = 0.1
  prompt_bs: int = 1
  loss_weight: float = 1.0
  print_step: int = 100
  batch_size: int = 1
  clip_model: str = "ViT-H-14"
  clip_pretrain: str = "laion2b_s32b_b79k"


class DDDAttackPipeline(StableDiffusionInpaintPipeline):
  """DDD attack pipeline."""

  vae: AutoencoderKL
  text_encoder: CLIPTextModel
  tokenizer: CLIPTokenizer
  unet: UNet2DConditionModel
  scheduler: KarrasDiffusionSchedulers
  feature_extractor: CLIPImageProcessor

  args: DDDAttackArgs

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.vae.requires_grad_(False)
    self.unet.requires_grad_(False)

    self.args = DDDAttackArgs()

  @staticmethod
  def load_images(
      image_path: str,
      mask_path: str,
      width: int = 512,
      height: int = 512,
  ) -> tuple[Image.Image, Image.Image]:
    """Load the images and masks.

    Args:
      image_path: The path to the image.

      mask_path: The path to the mask.

    Returns:
      The image and mask tensors as PIL images as RGB, 3xWXH, [0, 255].
    """
    image = load_image(image_path, width, height)
    mask = load_image(mask_path, width, height)
    inverted_mask = ImageOps.invert(mask)

    return image, inverted_mask

  @staticmethod
  def prepare_image_and_mask(
      image: Image.Image,
      mask: Image.Image,
      device: torch.device,
      dtype: torch.dtype,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare the image and mask tensors.

    Args:
      image: An RGB PIL image.

      mask: An inverted mask PIL image encoded as RGB.

    Returns:
      The image and mask tensors, both of shape [1, 3, H, W]. The image is in
      the range [-1, 1] and the mask is in the range [0, 1].
    """
    image_tensor = to_tensor_func(image)[None, :, :, :] * 2.0 - 1.0
    mask_tensor = to_tensor_func(mask.convert("L"))[None, :, :, :]
    return (
        image_tensor.to(device=device, dtype=dtype),
        mask_tensor.to(device=device, dtype=dtype),
    )

  def init_prompt(
      self,
      prompt_len: int,
      prompt_batch_size: int,
      device: torch.device,
  ):
    # Randomly optimize prompt embeddings
    prompt_ids = torch.randint(
        high=len(self.tokenizer.encoder),
        size=(prompt_batch_size, prompt_len),
        device=device,
        pin_memory=False if device.type == "cpu" else True,
        dtype=torch.int32,
    )
    prompt_embeds = (
        self.text_encoder.text_model.embeddings.token_embedding
        .forward(prompt_ids).detach()
    )
    prompt_embeds.requires_grad = True

    # initialize the template
    template_text = "{}"
    padded_template_text = template_text.format(
        " ".join(["<start_of_text>"] * prompt_len)
    )
    dummy_ids = self.tokenizer.encode(padded_template_text)

    # -1 for optimized tokens
    dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
    dummy_ids = [49406] + dummy_ids + [49407]
    dummy_ids += [0] * (77 - len(dummy_ids))
    dummy_ids = torch.tensor([dummy_ids] * prompt_batch_size, device=device)

    # for getting dummy embeds; -1 won't work for token_embedding
    tmp_dummy_ids = dummy_ids.detach().clone()
    tmp_dummy_ids[tmp_dummy_ids == -1] = 0
    dummy_embeds = (
        self.text_encoder.text_model.embeddings.token_embedding
        .forward(tmp_dummy_ids).detach()
    )
    dummy_embeds.requires_grad = False

    return prompt_embeds, dummy_embeds, dummy_ids

  def token_projective_embedding_optimization(self):
    # Remember to set seed before this.
    args = TokenProjectiveEmbeddingArgs()

    prompt_embeds, dummy_embeds, dummy_ids = self.init_prompt(
        args.prompt_len, args.batch_size, self.device
    )

    input_optimizer = torch.optim.AdamW(
        [prompt_embeds],
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True,
    )

    for step in range(args.iters):
      if step == args.decreaser_lr_step:
        # Set lower learning rate
        for param_group in input_optimizer.param_groups:
          param_group["lr"] = args.lower_lr

      if step >= args.decreaser_lr_step:
        projected_embeds, nn_indices = utils_text.nn_project(
            prompt_embeds, token_embedding
        )
        tmp_embeds = projected_embeds.detach().clone().requires_grad_(True)
      else:
        tmp_embeds = prompt_embeds.detach().clone().requires_grad_(True)

    pass

  def get_text_embedding_with_embeddings(
      self,
      prompt_ids: torch.Tensor,
      prompt_embeddings: torch.FloatTensor,
      attention_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    return self.encode_embeddings(
        prompt_ids,
        prompt_embeddings,
        attention_mask=attention_mask,
    )[0]

  def encode_embeddings(
      self,
      prompt: torch.Tensor,
      prompt_embeddings: torch.FloatTensor,
      attention_mask: torch.Tensor | None = None,
  ):
    """Encode the text embeddings.

    This is effectively the `forward` method of the `CLIPTextTransformer`, but
    with the `prompt_embeddings` as input instead of the `prompt`.

    Args:
      prompt: The prompt tensor of shape `[B, S]`.

      prompt_embeddings: The prompt embeddings tensor of shape `[B, S, H]`.

      attention_mask: The attention mask tensor of shape `[B, S]`.

    Returns:
      The encoded text embeddings.
    """
    text_model: CLIPTextTransformer = self.text_encoder.text_model
    config = text_model.config
    output_attentions = config.output_attentions
    output_hidden_states = config.output_hidden_states
    return_dict = config.use_return_dict

    batch_size, seq_len = prompt.shape

    hidden_states = text_model.embeddings.forward(
        inputs_embeds=prompt_embeddings
    )

    causal_attention_mask = attn_bias._materialize_causal_mask(
        shape=(batch_size, seq_len, seq_len),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # expand attention_mask
    if attention_mask is not None:
      # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
      attention_mask = AttentionMaskConverter._expand_mask(
          mask=attention_mask,
          dtype=hidden_states.dtype,
      )

    encoder_outputs = text_model.encoder.forward(
        inputs_embeds=hidden_states,
        attention_mask=attention_mask,
        causal_attention_mask=causal_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    last_hidden_state = text_model.final_layer_norm.forward(encoder_outputs[0])

    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=prompt.device),
        prompt.to(torch.int).argmax(dim=-1),
    ]

    if not return_dict:
      return (last_hidden_state, pooled_output) + encoder_outputs[1:]

    return BaseModelOutputWithPooling(
        last_hidden_state=last_hidden_state,  # type: ignore
        pooler_output=pooled_output,  # type: ignore
        hidden_states=encoder_outputs.hidden_states,  # type: ignore
        attentions=encoder_outputs.attentions,  # type: ignore
    )


def load_image(
    image_path: str, width: int = 512, height: int = 512
) -> Image.Image:
  """Load the image.

  Args:
    image_path: The path to the image.

  Returns:
    The image tensor.
  """
  return Image.open(image_path).resize(
      (width, height),
      resample=Image.Resampling.BILINEAR,
  ).convert("RGB")


def set_random_seed(seed=0):
  torch.manual_seed(seed + 0)
  torch.cuda.manual_seed(seed + 1)
  torch.cuda.manual_seed_all(seed + 2)
  np.random.seed(seed + 3)
  torch.cuda.manual_seed_all(seed + 4)
  random.seed(seed + 5)


def dict_to_args_parser(input_dict: dict[str, Any]) -> argparse.Namespace:
  parser = argparse.ArgumentParser()
  for key, value in input_dict.items():
    parser.add_argument(f'--{key}', default=value, type=type(value))
  return parser.parse_args([])


def nn_project(
    curr_embeds: torch.Tensor,
    embedding_layer: torch.nn.Embedding,
    # print_hits=False,
):
  with torch.no_grad():
    bsz, seq_len, emb_dim = curr_embeds.shape

    # Using the sentence transformers semantic search which is
    # a dot product exact kNN search between a set of
    # query vectors and a corpus of vectors
    curr_embeds = curr_embeds.reshape((-1, emb_dim))

    # queries
    curr_embeds = torch.nn.functional.normalize(curr_embeds, p=2, dim=1)

    embedding_matrix = torch.nn.functional.normalize(
        embedding_layer.weight, p=2, dim=1
    )

    hits = semantic_search(
        curr_embeds,
        embedding_matrix,
        query_chunk_size=curr_embeds.shape[0],
        top_k=1,
        score_function=dot_score
    )

    # if print_hits:
    #   all_hits = []
    #   for hit in hits:
    #     all_hits.append(hit[0]["score"])
    #   print(f"mean hits:{mean(all_hits)}")

    nn_indices = torch.tensor(
        [hit[0]["corpus_id"] for hit in hits],
        device=curr_embeds.device,
    )
    nn_indices = nn_indices.reshape((bsz, seq_len))

    projected_embeds = embedding_layer(nn_indices)

  return projected_embeds, nn_indices

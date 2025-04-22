import copy
import json
import random
from io import BytesIO
from statistics import mean
from typing import Any

import numpy as np
import open_clip
import requests
import torch
from PIL import Image
from sentence_transformers.util import dot_score
from sentence_transformers.util import normalize_embeddings
from sentence_transformers.util import semantic_search
from transformers.modeling_outputs import BaseModelOutputWithPooling

def nn_project(curr_embeds, embedding_layer, print_hits=False):
  with torch.no_grad():
    bsz, seq_len, emb_dim = curr_embeds.shape

    # Using the sentence transformers semantic search which is
    # a dot product exact kNN search between a set of
    # query vectors and a corpus of vectors
    curr_embeds = curr_embeds.reshape((-1, emb_dim))
    curr_embeds = normalize_embeddings(curr_embeds)  # queries

    embedding_matrix = embedding_layer.weight
    embedding_matrix = normalize_embeddings(embedding_matrix)

    hits = semantic_search(
        curr_embeds,
        embedding_matrix,
        query_chunk_size=curr_embeds.shape[0],
        top_k=1,
        score_function=dot_score
    )

    if print_hits:
      all_hits = []
      for hit in hits:
        all_hits.append(hit[0]["score"])
      print(f"mean hits:{mean(all_hits)}")

    nn_indices = torch.tensor(
        [hit[0]["corpus_id"] for hit in hits], device=curr_embeds.device
    )
    nn_indices = nn_indices.reshape((bsz, seq_len))

    projected_embeds = embedding_layer(nn_indices)

  return projected_embeds, nn_indices

def decode_ids(input_ids, tokenizer, by_token=False):
  input_ids = input_ids.detach().cpu().numpy()

  texts = []

  if by_token:
    for input_ids_i in input_ids:
      curr_text = []
      for tmp in input_ids_i:
        curr_text.append(tokenizer.decode([tmp]))

      texts.append('|'.join(curr_text))
  else:
    for input_ids_i in input_ids:
      texts.append(tokenizer.decode(input_ids_i))

  return texts


def download_image(url):
  try:
    response = requests.get(url)
  except:
    return None
  return Image.open(BytesIO(response.content)).convert("RGB")


def get_target_feature(
    model,
    preprocess,
    tokenizer_funct,
    device,
    target_images=None,
    target_prompts=None
):
  if target_images is not None:
    with torch.no_grad():
      curr_images = [preprocess(i).unsqueeze(0) for i in target_images]
      curr_images = torch.concatenate(curr_images).to(device)
      all_target_features = model.encode_image(curr_images)
  else:
    texts = tokenizer_funct(target_prompts).to(device)
    all_target_features = model.encode_text(texts)

  return all_target_features


def initialize_prompt(tokenizer, token_embedding, args, device):
  prompt_len = args.prompt_len

  # randomly optimize prompt embeddings
  prompt_ids = torch.randint(
      len(tokenizer.encoder), (args.prompt_bs, prompt_len)
  ).to(device)
  prompt_embeds = token_embedding(prompt_ids).detach()
  prompt_embeds.requires_grad = True

  # initialize the template
  template_text = "{}"
  padded_template_text = template_text.format(
      " ".join(["<start_of_text>"] * prompt_len)
  )
  dummy_ids = tokenizer.encode(padded_template_text)

  # -1 for optimized tokens
  dummy_ids = [i if i != 49406 else -1 for i in dummy_ids]
  dummy_ids = [49406] + dummy_ids + [49407]
  dummy_ids += [0] * (77 - len(dummy_ids))
  dummy_ids = torch.tensor([dummy_ids] * args.prompt_bs).to(device)

  # for getting dummy embeds; -1 won't work for token_embedding
  tmp_dummy_ids = copy.deepcopy(dummy_ids)
  tmp_dummy_ids[tmp_dummy_ids == -1] = 0
  dummy_embeds = token_embedding(tmp_dummy_ids).detach()
  dummy_embeds.requires_grad = False

  return prompt_embeds, dummy_embeds, dummy_ids


def get_text_embedding_with_embeddings(
    self, prompt_ids, prompt_embeddings, attention_mask=None
):
  text_embeddings = encode_embeddings(
      self,
      prompt_ids,
      prompt_embeddings,
      attention_mask=attention_mask,
  )

  return text_embeddings[0]

@torch.compile(backend="cudagraphs")
def encode_embeddings(self, prompt, prompt_embeddings, attention_mask=None):
  output_attentions = self.text_encoder.text_model.config.output_attentions
  output_hidden_states = (
      self.text_encoder.text_model.config.output_hidden_states
  )
  return_dict = self.text_encoder.text_model.config.use_return_dict

  hidden_states = self.text_encoder.text_model.embeddings(
      inputs_embeds=prompt_embeddings
  )

  bsz, seq_len = prompt.shape[0], prompt.shape[1]

  causal_attention_mask = build_causal_attention_mask(
      bsz, seq_len, hidden_states.dtype, hidden_states.device
  )
  # expand attention_mask
  if attention_mask is not None:
    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
    attention_mask = self.text_encoder.text_model._expand_mask(
        attention_mask, hidden_states.dtype
    )

  encoder_outputs = self.text_encoder.text_model.encoder(
      inputs_embeds=hidden_states,
      attention_mask=attention_mask,
      causal_attention_mask=causal_attention_mask,
      output_attentions=output_attentions,
      output_hidden_states=output_hidden_states,
      return_dict=return_dict,
  )

  last_hidden_state = encoder_outputs[0]
  last_hidden_state = self.text_encoder.text_model.final_layer_norm(
      last_hidden_state
  )

  pooled_output = last_hidden_state[
      torch.arange(last_hidden_state.shape[0], device=prompt.device),
      prompt.to(torch.int).argmax(dim=-1)]

  if not return_dict:
    return (last_hidden_state, pooled_output) + encoder_outputs[1:]

  return BaseModelOutputWithPooling(
      last_hidden_state=last_hidden_state,
      pooler_output=pooled_output,
      hidden_states=encoder_outputs.hidden_states,
      attentions=encoder_outputs.attentions,
  )


def build_causal_attention_mask(bsz, seq_len, dtype, device):
  # lazily create causal attention mask, with full attention between the vision tokens
  # pytorch uses additive attention mask; fill with -inf
  mask = torch.empty(bsz, seq_len, seq_len, dtype=dtype, device=device)
  mask.fill_(float("-inf"))
  mask.triu_(1)  # zero out the lower diagonal
  mask = mask.unsqueeze(1)  # expand mask
  return mask

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import copy
import datetime
import sys
import ddd

import matplotlib.pyplot as plt
import torch
import utils
import utils_text
from diffusers import AutoencoderKL
from diffusers import BitsAndBytesConfig
from diffusers import StableDiffusionInpaintPipeline
from diffusers import UNet2DConditionModel
from PIL import Image
from PIL import ImageOps
from torchvision import transforms
import torchvision
from diff_jpeg import DiffJPEGCoding


diff_jpeg_coding_module = DiffJPEGCoding()
to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()
torch.backends.cuda.matmul.allow_tf32 = True

ddd_type = sys.argv[1]
if ddd_type == "original":
   diffjpeg = False
else:
   diffjpeg = True
   

DCS = True
username = "sneakers-pretrained-models"
if DCS:
  models_path = f"/dcs/large/{username}"
else:
  models_path = None

args_dict = {
    "prompt_len": 16,
    "iter": 3000,
    "lr": 0.1,
    "weight_decay": 0.1,
    "prompt_bs": 1,
    "loss_weight": 1.0,
    "print_step": 100,
    "batch_size": 1,
    "clip_model": "ViT-H-14",
    "clip_pretrain": "laion2b_s32b_b79k"
}


def dict_to_args_parser(input_dict):
  parser = argparse.ArgumentParser()
  for key, value in input_dict.items():
    parser.add_argument(f'--{key}', default=value, type=type(value))
  return parser.parse_args([])


args = dict_to_args_parser(args_dict)

device = "cuda"
dtype = torch.float16

model_version = "stabilityai/stable-diffusion-2-inpainting"

filenames = ["011"]

for testimg_filename in filenames:
  nf4_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=dtype,
  )
  unet_nf4 = UNet2DConditionModel.from_pretrained(
      model_version,
      subfolder="unet",
      quantization_config=nf4_config,
      torch_dtype=dtype,
      use_safetensors=True,
      cache_dir=models_path,
  )

  vae_nf4 = AutoencoderKL.from_pretrained(
      model_version,
      subfolder="vae",
      quantization_config=nf4_config,
      torch_dtype=dtype,
      use_safetensors=True,
      cache_dir=models_path,
  )

  pipe_inpaint: StableDiffusionInpaintPipeline = StableDiffusionInpaintPipeline.from_pretrained(
      model_version,
      variant="fp16",
      unet=unet_nf4,
      vae=vae_nf4,
      torch_dtype=dtype,
      use_safetensors=True,
      cache_dir=models_path,
  )

  # pipe_inpaint.vae = pipe_inpaint.vae.to(device, dtype)
  pipe_inpaint.text_encoder = pipe_inpaint.text_encoder.to(device, dtype)
  pipe_inpaint.unet.to(memory_format=torch.channels_last)
  pipe_inpaint = pipe_inpaint.to(device=device, memory_format=torch.channels_last)
  pipe_inpaint.safety_checker = None
  pipe_inpaint.vae.requires_grad_(False)
  pipe_inpaint.unet.requires_grad_(False)

  size = 512
  size_2d = (size, size)

  tokenizer = pipe_inpaint.tokenizer
  token_embedding = pipe_inpaint.text_encoder.text_model.embeddings.token_embedding

  preprocess = transforms.Compose(
      [
          transforms.Resize(
              size,
              interpolation=transforms.InterpolationMode.BILINEAR,
          ),
          transforms.CenterCrop(size),
          transforms.ToTensor(),
      ]
  )

  orig_images = Image.open(f'./images/{testimg_filename}.png'
                          ).convert('RGB').resize(size_2d)
  mask_image_orig = Image.open(f'./images/{testimg_filename}_masked.png'
                              ).convert('RGB').resize(size_2d)
  mask_image = ImageOps.invert(mask_image_orig).resize(size_2d)

  cur_mask, cur_masked_image, init_image = utils.prepare_mask_and_masked2(
      orig_images, mask_image, no_mask=False, inverted=True
  )
  inv_cur_mask, _, _ = utils.prepare_mask_and_masked2(
      orig_images, mask_image, no_mask=False, inverted=False
  )

  cur_mask = cur_mask.to(device, dtype)
  cur_masked_image = cur_masked_image.to(device, dtype)

  with torch.no_grad():
    curr_images = preprocess(orig_images).to(device)
    mask = torch.nn.functional.interpolate(cur_mask, size=(size // 8, size // 8))
    if len(curr_images.shape) == 3:
      curr_images = curr_images.unsqueeze(0)
    elif len(curr_images.shape) == 5:
      curr_images = curr_images.squeeze(0)
    all_latents = pipe_inpaint.vae.encode(curr_images.to(dtype)
                                        ).latent_dist.sample()
    all_latents = all_latents * 0.18215
    masked_image_latents = pipe_inpaint.vae.encode(cur_masked_image
                                                  ).latent_dist.sample() * 0.18215
    gt_embeddings = utils.get_text_embedding(pipe_inpaint, testimg_filename)
    uncond_embeddings = utils.get_text_embedding(pipe_inpaint, "")

  args.prompt_len = 8
  args.opt_iters = 350
  args.eval_step = 50
  # discrete = True
  args.lr = 0.0001


  args.lr = 0.001
  prompt_embeds, dummy_embeds, dummy_ids = utils_text.initialize_prompt(
      tokenizer, token_embedding, args, device
  )
  input_optimizer = torch.optim.AdamW(
      [prompt_embeds],
      lr=args.lr,
      weight_decay=args.weight_decay,
  )
  input_optim_scheduler = None
  best_loss = -999
  eval_loss = -99999
  best_text = ""
  best_embeds = None
  for step in range(args.opt_iters):
    if step > args.opt_iters - 10:  # Finalize with full continuous update
      args.lr = 0.0001
      projected_embeds, nn_indices = utils_text.nn_project(
          prompt_embeds, token_embedding
      )
      tmp_embeds = copy.deepcopy(prompt_embeds)
      tmp_embeds.data = projected_embeds.data
      tmp_embeds.requires_grad = True
    else:
      tmp_embeds = copy.deepcopy(prompt_embeds)
      tmp_embeds.data = prompt_embeds.data
      tmp_embeds.requires_grad = True

    padded_embeds = copy.deepcopy(dummy_embeds)
    padded_embeds[:, 1:args.prompt_len + 1] = tmp_embeds
    padded_embeds = padded_embeds.repeat(args.batch_size, 1, 1)
    padded_dummy_ids = dummy_ids.repeat(args.batch_size, 1)

    if args.batch_size is None:
      latents = all_latents
    else:
      perm = torch.randperm(len(all_latents))
      idx = perm[:args.batch_size]
      latents = all_latents[idx]

    noise = torch.randn_like(latents)
    bsz = latents.shape[0]
    timesteps = torch.randint(0, 1000, (bsz,), device=latents.device)
    timesteps = timesteps.long()

    noisy_latents = pipe_inpaint.scheduler.add_noise(latents, noise, timesteps)

    if pipe_inpaint.scheduler.config.prediction_type == "epsilon":
      target = noise
    elif pipe_inpaint.scheduler.config.prediction_type == "v_prediction":
      target = pipe_inpaint.scheduler.get_velocity(latents, noise, timesteps)
    else:
      raise ValueError(
          f"Unknown prediction type {pipe_inpaint.scheduler.config.prediction_type}"
      )

    text_embeddings = utils_text.get_text_embedding_with_embeddings(
        pipe_inpaint, padded_dummy_ids, padded_embeds
    )

    input_latent = torch.cat([noisy_latents, mask, masked_image_latents], dim=1)
    model_pred = pipe_inpaint.unet.forward(
        input_latent,
        timesteps,
        encoder_hidden_states=text_embeddings,
        return_dict=False
    )[0]
    inverted_mask = mask

    target = target * inverted_mask
    model_pred = model_pred * inverted_mask
    loss = torch.nn.functional.mse_loss(
        model_pred.float(), target.float(), reduction="mean"
    )
    prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])
    input_optimizer.step()
    input_optimizer.zero_grad()

    curr_lr = input_optimizer.param_groups[0]["lr"]


  input_text_embedding = text_embeddings.detach()
  input_text_embeddings = torch.cat([input_text_embedding] * 2)

  # pipe_inpaint.to(torch_dtype=torch.float16)

  # for testimg_filename in test_file_list:
  prefix_filename = "./images/" + testimg_filename
  init_image = Image.open(f"{prefix_filename}.png").convert("RGB").resize(size_2d)
  mask_image = Image.open(f"{prefix_filename}_masked.png").convert("RGB")
  mask_image = ImageOps.invert(mask_image).resize(size_2d)

  target_prompt = ""

  prompt = ""
  SEED = 786349
  torch.manual_seed(SEED)

  cur_mask, cur_masked_image = utils.prepare_mask_and_masked(
      init_image, mask_image
  )

  cur_mask = cur_mask.to(dtype=dtype, device=device)
  cur_masked_image = cur_masked_image.to(dtype=dtype, device=device)

  strength = 0.7
  guidance_scale = 7.5
  num_inference_steps = 4

  text_embeddings = utils.text_embedding(pipe_inpaint, target_prompt)

  latents_shape = (
      1, pipe_inpaint.vae.config.latent_channels, size // 8, size // 8
  )
  noise = torch.randn(latents_shape, device=device, dtype=text_embeddings.dtype)

  image_latent = utils.pil_to_latent(pipe_inpaint, init_image)

  prompt = ""
  SEED = 786349

  t_schedule = [720]
  t_schedule_bound = 10
  n_samples = 50
  loss_depth = [256, 64]

  torch.manual_seed(SEED)

  cur_mask, cur_masked_image = utils.prepare_mask_and_masked(
      init_image, mask_image
  )

  cur_mask = cur_mask.to(device=device, dtype=dtype)
  cur_masked_image = cur_masked_image.to(device=device, dtype=dtype)

  val_loss_criteria = "MSE"
  attn_controller = ddd.AttnController(
      post=False,
      mask=cur_mask,
      criteria=val_loss_criteria,
      target_depth=loss_depth
  )

  module_count = 0
  modes = ['', 'up', 'down']
  mode = 0
  # to collect only up or down attns, please deactivate the annotation of 'if' statement below
  for n, m in pipe_inpaint.unet.named_modules():
    # if (n.endswith('attn2') and (modes[mode] in n)) or (n.endswith('attn1') and (modes[mode] in n)): #and "down" in n:
    if (n.endswith('attn1') and (modes[mode] in n)):  #and "down" in n:
      attn_processor = ddd.MyCrossAttnProcessor(attn_controller, n)
      attn_processor.__call__ = torch.compile(
          attn_processor.__call__,
          backend="cudagraphs",
          fullgraph=True,
      )

      m.set_processor(attn_processor)
      module_count += 1

  sp_prompt = None

  for_mean = []
  for j in range(n_samples):
    with torch.no_grad():
      mask = cur_mask
      masked_image = cur_masked_image
      random_t = ddd.get_random_t(t_schedule, t_schedule_bound)
      uncond_emb = input_text_embeddings
      num_channels_latents = pipe_inpaint.vae.config.latent_channels
      latents_shape = (1, num_channels_latents, size // 8, size // 8)
      latents = torch.randn(latents_shape, device=device, dtype=uncond_emb.dtype)

      mask = torch.nn.functional.interpolate(mask, size=(size // 8, size // 8))
      mask = torch.cat([mask] * 2)
      masked_image_latents = pipe_inpaint.vae.encode(masked_image
                                                    ).latent_dist.sample()
      masked_image_latents = 0.18215 * masked_image_latents
      masked_image_latents = torch.cat([masked_image_latents] * 2)

      pipe_inpaint.scheduler.set_timesteps(num_inference_steps)
      timesteps_tensor = pipe_inpaint.scheduler.timesteps.to(device)
      timesteps_tensor = random_t

      for i, t in enumerate(timesteps_tensor):

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = torch.cat(
            [latent_model_input, mask, masked_image_latents], dim=1
        )
        pipe_inpaint.unet.forward(
            sample=latent_model_input,
            timestep=t,
            encoder_hidden_states=uncond_emb,
            return_dict=False,
        )
        # _ = pipe_inpaint.unet(
        #     latent_model_input, t, encoder_hidden_states=uncond_emb
        # ).sample
      for_mean.append(attn_controller.targets)
      attn_controller.zero_attn_probs()
  meaned = []
  for feature in range(len(for_mean[0])):
    temp = 0
    for idx in range(n_samples):
      temp += for_mean[idx][feature]
    temp /= n_samples
    meaned.append(temp)

  attn_controller.target_hidden = meaned
  del for_mean, meaned
  text_embeddings = [uncond_emb]

  now = datetime.datetime.now()
  now = str(now)[:19].replace(" ", "_")
  nickname = f"run"
  directory = f"{nickname}_{now}"
  path = f"figures/{directory}"
  img_path = path + "/img"
  adv_path = path + "/adv"
  os.makedirs(path)
  os.makedirs(img_path)
  os.makedirs(adv_path)

  infer_dict = dict()
  infer_dict["prompt"] = utils.load_prompt(f"prompts/{testimg_filename}.txt")[0]
  infer_dict["num_inference_steps"] = 20
  infer_dict["guidance_scale"] = 7.5
  infer_dict["strength"] = 0.8
  infer_dict["init_image"] = init_image
  infer_dict["mask"] = mask_image
  infer_dict["prefix"] = 'inter'
  infer_dict["path"] = img_path
  infer_dict["inter_print"] = []

  # iters = 100
  iters = 250
  grad_reps = 7
  loss_mask = True
  eps = 12
  step_size = 3
  pixel_loss = 0
  # val_loss_criteria = "MSE"

  pipe_inpaint.text_encoder = pipe_inpaint.text_encoder.to(device="cpu")
  pipe_inpaint.vae.decoder = pipe_inpaint.vae.decoder.to(device="cpu")
  pipe_inpaint.vae.post_quant_conv = pipe_inpaint.vae.post_quant_conv.to(
      device="cpu"
  )

  os.makedirs("./adversarial", exist_ok=True)

  result, total_losses = ddd.disrupt(
      (cur_mask),
      (cur_masked_image),
      text_embeddings=text_embeddings,
      eps=eps,
      step_size=step_size,
      iters=iters,
      clamp_min=-1,
      clamp_max=1,
      eta=1,
      num_inference_steps=num_inference_steps,
      guidance_scale=guidance_scale,
      grad_reps=grad_reps,
      attn_controller=attn_controller,
      pipe=(pipe_inpaint),
      loss_depth=loss_depth,
      loss_mask=loss_mask,
      pixel_loss=pixel_loss,
      t_schedule=t_schedule,
      diffjpeg=diffjpeg,
      t_schedule_bound=t_schedule_bound,
      infer_dict=infer_dict,
      infer_unet=pipe_inpaint.unet,
      inter_print=infer_dict["inter_print"],
  )
  torch.save(result, f'{adv_path}/adv.pt')

  pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
      model_version,
      variant="fp16",
      torch_dtype=torch.float32,
      use_safetensors=True,
      cache_dir=models_path,
  )
    
  pipe_inpaint = pipe_inpaint.to("cuda")
  pipe_inpaint.safety_checker = None

  for name, param in pipe_inpaint.unet.named_parameters():
    param.requires_grad = False

  directory = f"{nickname}_{now}"
  path = f"figures/{directory}"
  img_path = path + "/img"
  adv_path = path + "/adv"
  result = torch.load(f'{adv_path}/adv.pt')[0]
  adv_X = (result / 2 + 0.5)#.clamp(0, 1)
  adv_image = to_pil(adv_X.to(torch.float32)).convert("RGB")
  adv_image = utils.recover_image(
      adv_image, init_image, mask_image, background=True
  )

  # Show the difference between the original image and the adversarial image
  diff = to_tensor(adv_image) - to_tensor(init_image)
  diff = to_pil(diff)


  if diffjpeg:
      adv_image.save(f'./adversarial/{testimg_filename}_diffjpeg_adv.png')
  else:
      adv_image.save(f'./adversarial/{testimg_filename}_adv.png')

  os.environ["CUDA_VISIBLE_DEVICES"] = "0"

  if diffjpeg:
    test_case = [3, 4]
  else:
    test_case = [1, 2]

  for test in test_case:
    init_image = Image.open(f'./images/{testimg_filename}.png').convert('RGB').resize((512, 512))
    mask_image = Image.open(f'./images/{testimg_filename}_masked.png').convert('RGB')
    mask_image = ImageOps.invert(mask_image).resize((512, 512))

    if test==1:
        adv_image = Image.open(f'./adversarial/{testimg_filename}_adv.png').resize((512, 512))
    if test==2:
        adv_image = torchvision.io.read_image(f'./adversarial/{testimg_filename}_adv.png').float()[None]
        jpeg_quality = torch.tensor([50])
        image_coded = diff_jpeg_coding_module(image_rgb=adv_image, jpeg_quality=jpeg_quality)
        torch_image = image_coded.squeeze(0)
        to_pil = torchvision.transforms.ToPILImage()
        adv_image = to_pil(torch_image / 255).resize((512, 512))
    if test==3:
        adv_image = Image.open(f'./adversarial/{testimg_filename}_diffjpeg_adv.png').resize((512, 512))
    if test==4:
        adv_image = torchvision.io.read_image(f'./adversarial/{testimg_filename}_diffjpeg_adv.png').float()[None]
        jpeg_quality = torch.tensor([50])
        image_coded = diff_jpeg_coding_module(image_rgb=adv_image, jpeg_quality=jpeg_quality)
        torch_image = image_coded.squeeze(0)
        to_pil = torchvision.transforms.ToPILImage()
        adv_image = to_pil(torch_image / 255).resize((512, 512))

    prompts = utils.load_prompt(f"prompts/{testimg_filename}.txt")
    print(prompts)

    # prompts = ["a woman with a hat on a beach in black and white"]

    SEED = 1007

    with torch.no_grad():
      for promptnum, prompt in enumerate(prompts):

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
        image_nat = utils.recover_image(image_nat, init_image, mask_image)

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

        image_adv = utils.recover_image(image_adv, init_image, mask_image)

        
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
              
        if test==1:
            os.makedirs(f"./{testimg_filename}/original", exist_ok=True)
            image_adv.save(f"./{testimg_filename}/original/prompt{promptnum}.png")
            plt.savefig(f'./{testimg_filename}/original/result_prompt{promptnum}.png')
        elif test==2:
            os.makedirs(f"./{testimg_filename}/original_compressed", exist_ok=True)
            image_adv.save(f"./{testimg_filename}/original_compressed/prompt{promptnum}.png")
            plt.savefig(f'./{testimg_filename}/original_compressed/result_prompt{promptnum}.png')
        elif test==3:
            os.makedirs(f"./{testimg_filename}/diffjpeg", exist_ok=True)
            image_adv.save(f"./{testimg_filename}/diffjpeg/prompt{promptnum}.png")
            plt.savefig(f'./{testimg_filename}/diffjpeg/result_prompt{promptnum}.png')
        elif test==4:
            os.makedirs(f"./{testimg_filename}/diffjpeg_compressed", exist_ok=True)
            image_adv.save(f"./{testimg_filename}/diffjpeg_compressed/prompt{promptnum}.png")
            plt.savefig(f'./{testimg_filename}/diffjpeg_compressed/result_prompt{promptnum}.png')

        plt.clf()
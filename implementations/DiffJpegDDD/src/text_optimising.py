from utils_text import initialize_prompt, nn_project, get_text_embedding_with_embeddings
import torch
import copy
from ddd import AttnController, MyCrossAttnProcessor, get_random_t

class TextOptimizer:
    def __init__(self, tokenizer, token_embedding, args, device, pipe_inpaint, all_latents, mask, masked_image_latents):
        self.args = args
        self.device = device
        self.pipe_inpaint = pipe_inpaint
        self.all_latents = all_latents
        self.mask = mask
        self.masked_image_latents = masked_image_latents
        self.token_embedding = token_embedding
        self.args.prompt_len = 8
        self.args.opt_iters = 350
        self.args.eval_step = 50
        self.args.lr = 0.001  # Updated learning rate
        
        self.prompt_embeds, self.dummy_embeds, self.dummy_ids = initialize_prompt(
            tokenizer, token_embedding, args, device
        )
        self.input_optimizer = torch.optim.AdamW(
            [self.prompt_embeds],
            lr=self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.best_loss = -999
        self.best_text = ""
        self.best_embeds = None
    
    def optimize(self):
        for step in range(self.args.opt_iters):
            self._update_learning_rate(step)
            tmp_embeds = self._prepare_embeddings(step)
            padded_embeds, padded_dummy_ids = self._prepare_padded_embeddings(tmp_embeds)
            latents = self._select_latents()
            loss = self._compute_loss(latents, padded_embeds, padded_dummy_ids)
            
            self.prompt_embeds.grad, = torch.autograd.grad(loss, [tmp_embeds])
            self.input_optimizer.step()
            self.input_optimizer.zero_grad()
        return self.text_embeddings
            
    def _update_learning_rate(self, step):
        if step > self.args.opt_iters - 10:
            self.args.lr = 0.0001
    
    def _prepare_embeddings(self, step):
        if step > self.args.opt_iters - 10:
            projected_embeds, _ = nn_project(self.prompt_embeds, self.token_embedding)
            tmp_embeds = copy.deepcopy(self.prompt_embeds)
            tmp_embeds.data = projected_embeds.data
        else:
            tmp_embeds = copy.deepcopy(self.prompt_embeds)
            tmp_embeds.data = self.prompt_embeds.data
        
        tmp_embeds.requires_grad = True
        return tmp_embeds
    
    def _prepare_padded_embeddings(self, tmp_embeds):
        padded_embeds = copy.deepcopy(self.dummy_embeds)
        padded_embeds[:, 1:self.args.prompt_len + 1] = tmp_embeds
        padded_embeds = padded_embeds.repeat(self.args.batch_size, 1, 1)
        padded_dummy_ids = self.dummy_ids.repeat(self.args.batch_size, 1)
        return padded_embeds, padded_dummy_ids
    
    def _select_latents(self):
        if self.args.batch_size is None:
            return self.all_latents
        else:
            perm = torch.randperm(len(self.all_latents))
            idx = perm[:self.args.batch_size]
            return self.all_latents[idx]
    
    def _compute_loss(self, latents, padded_embeds, padded_dummy_ids):
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        timesteps = torch.randint(0, 1000, (bsz,), device=latents.device).long()
        noisy_latents = self.pipe_inpaint.scheduler.add_noise(latents, noise, timesteps)
        
        target = self._get_target(latents, noise, timesteps)
        self.text_embeddings = get_text_embedding_with_embeddings(
            self.pipe_inpaint, padded_dummy_ids, padded_embeds
        )
        
        input_latent = torch.cat([noisy_latents, self.mask, self.masked_image_latents], dim=1)
        model_pred = self.pipe_inpaint.unet.forward(
            input_latent,
            timesteps,
            encoder_hidden_states=self.text_embeddings,
            return_dict=False
        )[0]
        
        target *= self.mask
        model_pred *= self.mask
        return torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")
    
    def _get_target(self, latents, noise, timesteps):
        if self.pipe_inpaint.scheduler.config.prediction_type == "epsilon":
            return noise
        elif self.pipe_inpaint.scheduler.config.prediction_type == "v_prediction":
            return self.pipe_inpaint.scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f"Unknown prediction type {self.pipe_inpaint.scheduler.config.prediction_type}")


class SemanticCentroids:
    def __init__(self, pipe_inpaint, device, dtype, size, num_inference_steps, input_text_embeddings):
      self.pipe_inpaint = pipe_inpaint
      self.device = device
      self.dtype = dtype
      self.size = size
      self.num_inference_steps = num_inference_steps
      self.input_text_embeddings = input_text_embeddings

    def get_attention(self, cur_mask, loss_criteria, loss_depth):
        return AttnController(
                post=False,
                mask=cur_mask,
                criteria=loss_criteria,
                target_depth=loss_depth
            )
    
    def attention_processors(self, attncontroller):
        module_count = 0
        modes = ['', 'up', 'down']
        mode = 0
        for n, m in self.pipe_inpaint.unet.named_modules():
            if (n.endswith('attn1') and (modes[mode] in n)):  #and "down" in n:
                attn_processor = MyCrossAttnProcessor(attncontroller, n)
                attn_processor.__call__ = torch.compile(
                    attn_processor.__call__,
                    backend="cudagraphs",
                    fullgraph=True,
                )

                m.set_processor(attn_processor)
                module_count += 1

    def generate_samples(self, cur_mask, cur_masked_image, t_schedule, t_schedule_bound, n_samples, attncontroller):
        results = []

        for _ in range(n_samples):
            with torch.no_grad():
                mask, masked_image = cur_mask, cur_masked_image
                random_t = get_random_t(t_schedule, t_schedule_bound)
                latents, mask, masked_image_latents = self._initialise_latents(mask, masked_image)
                self.pipe_inpaint.scheduler.set_timesteps(self.num_inference_steps)

                for t in random_t:
                    latent_model_input = torch.cat([latents] * 2)
                    latent_model_input = torch.cat(
                        [latent_model_input, mask, masked_image_latents], dim=1
                    )
                    self.pipe_inpaint.unet.forward(
                        sample=latent_model_input,
                        timestep=t,
                        encoder_hidden_states=self.input_text_embeddings,
                        return_dict=False,
                    )
                results.append(attncontroller.targets)
                attncontroller.zero_attn_probs()
        return self._compute_mean(results, n_samples, attncontroller)

    def _initialise_latents(self, mask, masked_image):
        num_channels_latents = self.pipe_inpaint.vae.config.latent_channels
        latents_shape = (1, num_channels_latents, self.size // 8, self.size // 8)
        latents = torch.randn(latents_shape, device=self.device, dtype=self.input_text_embeddings.dtype)
        
        mask = torch.nn.functional.interpolate(mask, size=(self.size // 8, self.size // 8))
        mask = torch.cat([mask] * 2)
        masked_image_latents = self.pipe_inpaint.vae.encode(masked_image
                                                    ).latent_dist.sample()
        masked_image_latents = 0.18215 * masked_image_latents
        masked_image_latents = torch.cat([masked_image_latents] * 2)

        
        return latents, mask, masked_image_latents
    
    def _compute_mean(self, results, n_samples, attncontroller):
        means = [sum([results[idx][feature] for idx in range(n_samples)]) / n_samples for feature in range(len(results[0]))]
        attncontroller.target_hidden = means
        del results, means
        return [self.input_text_embeddings]
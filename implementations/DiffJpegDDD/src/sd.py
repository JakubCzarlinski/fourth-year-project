import torch
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL, BitsAndBytesConfig
from PIL import Image, ImageOps
import torchvision
from diff_jpeg import DiffJPEGCoding
import matplotlib.pyplot as plt
import os
from utils import load_prompt, recover_image

diff_jpeg_coding_module = DiffJPEGCoding()


class StableDiffusionInpaint:
    def __init__(self, model_version="stabilityai/stable-diffusion-2-inpainting", device="cuda", models_path=None):
        self.device = device
        self.model_version = model_version
        self.models_path = models_path
        self.dtype = torch.float16
        self.nf4_config = self._get_nf4()
        
        self.unet = self._load_unet()
        self.vae = self._load_vae()
        self.pipe_inpaint = self._initialize_sd_pipeline()
        self._configure_pipeline()
    
    def _get_nf4(self):
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype,
        )
    
    def _load_unet(self):
        return UNet2DConditionModel.from_pretrained(
            self.model_version,
            subfolder="unet",
            quantization_config=self.nf4_config,
            torch_dtype=self.dtype,
            use_safetensors=True,
            cache_dir=self.models_path,
        )
    
    def _load_vae(self):
        return AutoencoderKL.from_pretrained(
            self.model_version,
            subfolder="vae",
            quantization_config=self.nf4_config,
            torch_dtype=self.dtype,
            use_safetensors=True,
            cache_dir=self.models_path,
        )
    
    def _initialize_sd_pipeline(self):
        return StableDiffusionInpaintPipeline.from_pretrained(
            self.model_version,
            variant="fp16",
            unet=self.unet,
            vae=self.vae,
            torch_dtype=self.dtype,
            use_safetensors=True,
            cache_dir=self.models_path,
        )
    
    def _configure_pipeline(self):
        self.pipe_inpaint.text_encoder = self.pipe_inpaint.text_encoder.to(self.device, self.dtype)
        self.pipe_inpaint.unet.to(memory_format=torch.channels_last)
        self.pipe_inpaint = self.pipe_inpaint.to(device=self.device, memory_format=torch.channels_last)
        self.pipe_inpaint.safety_checker = None
        self.pipe_inpaint.vae.requires_grad_(False)
        self.pipe_inpaint.unet.requires_grad_(False)
    
    def get_pipeline(self):
        return self.pipe_inpaint
    

class Inference:
    def __init__(self, image_folder, testing_filename, image_name, model_version, models_path, diffjpeg=False, device="cuda"):
        self.image_name = image_name
        self.testing_filename = testing_filename
        self.device = device
        self.diffjpeg = diffjpeg
        self.test_cases = [3, 4] if diffjpeg else [1, 2]
        self.seed = 1007
        self.image_folder = image_folder
        self.pipe_inpaint = self.load_sd_pipeline(model_version, models_path)

    def load_sd_pipeline(self, model_version, models_path):
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_version,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=models_path,
        )
        pipe = pipe.to(self.device)
        pipe.safety_checker = None
        for name, param in pipe.unet.named_parameters():
            param.requires_grad = False
        return pipe

    def load_images(self, test):
        init_image = Image.open(f'{self.image_folder}original/{self.image_name}.png').convert('RGB').resize((512, 512))
        mask_image = ImageOps.invert(Image.open(f'{self.image_folder}masks/{self.image_name}_masked.png').convert('RGB')).resize((512, 512))
        adv_image = self.load_adversarial_image(test)
        return init_image, mask_image, adv_image

    def load_adversarial_image(self, test):        
        if test==1:
            return Image.open(f'{self.testing_filename}/original_adversarial.png').resize((512, 512))
        if test==2:
            adv_image = torchvision.io.read_image(f'{self.testing_filename}/original_adversarial.png').float()[None]
            jpeg_quality = torch.tensor([50])
            image_coded = diff_jpeg_coding_module(image_rgb=adv_image, jpeg_quality=jpeg_quality)
            torch_image = image_coded.squeeze(0)
            to_pil = torchvision.transforms.ToPILImage()
            adv_image = to_pil(torch_image / 255).resize((512, 512))
            return adv_image
        if test==3:
            adv_image = Image.open(f'{self.testing_filename}/diffjpeg_adversarial.png').resize((512, 512))
            return adv_image
        if test==4:
            adv_image = torchvision.io.read_image(f'{self.testing_filename}/diffjpeg_adversarial.png').float()[None]
            jpeg_quality = torch.tensor([50])
            image_coded = diff_jpeg_coding_module(image_rgb=adv_image, jpeg_quality=jpeg_quality)
            torch_image = image_coded.squeeze(0)
            to_pil = torchvision.transforms.ToPILImage()
            adv_image = to_pil(torch_image / 255).resize((512, 512))
            return adv_image

    def infer_images(self):
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        prompts = load_prompt(f"{self.image_folder}prompts/{self.image_name}_prompts.txt")

        for test in self.test_cases:
            init_image, mask_image, adv_image = self.load_images(test)
            torch.manual_seed(self.seed)
            with torch.no_grad():
                for promptnum, prompt in enumerate(prompts):
                    print(self.seed)
                    torch.manual_seed(self.seed)
                    strength, guidance_scale, num_inference_steps = 0.8, 7.5, 50

                    image_nat = self.generate_inpainted_image(init_image, init_image, mask_image, prompt, strength, guidance_scale, num_inference_steps)
                    torch.manual_seed(self.seed)
                    image_adv = self.generate_inpainted_image(init_image, adv_image, mask_image, prompt, strength, guidance_scale, num_inference_steps)
                    self.store_generated_images(test, promptnum, init_image, adv_image, image_nat, image_adv, prompt)

    def generate_inpainted_image(self, init_image, image, mask_image, prompt, strength, guidance_scale, num_inference_steps):
        return recover_image(self.pipe_inpaint(
            prompt=prompt,
            image=image,
            mask_image=mask_image,
            eta=1,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength
        ).images[0], init_image, mask_image)

    def store_generated_images(self, test, promptnum, init_image, adv_image, image_nat, image_adv, prompt):
        fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(20, 6))
        titles = ['Source Image', 'Adv Image', 'Gen. Image Nat.', 'Gen. Image Adv.']
        images = [init_image, adv_image, image_nat, image_adv]

        for i in range(4):
            ax[i].imshow(images[i])
            ax[i].set_title(titles[i], fontsize=16)
            ax[i].grid(False)
            ax[i].axis('off')

        fig.suptitle(f"{prompt} - {self.seed}", fontsize=20)
        fig.tight_layout()

        save_dirs = ["ddd_fast", "ddd_fast_compressed", "diffjpeg", "diffjpeg_compressed"]
        os.makedirs(f"{self.testing_filename}/{save_dirs[test-1]}", exist_ok=True)
        image_adv.save(f"{self.testing_filename}/{save_dirs[test-1]}/prompt{promptnum}.png")
        plt.savefig(f'{self.testing_filename}/{save_dirs[test-1]}/result_prompt{promptnum}.png')
        plt.clf()
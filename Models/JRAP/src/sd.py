import torch
from diffusers import StableDiffusionInpaintPipeline, UNet2DConditionModel, AutoencoderKL, BitsAndBytesConfig
from PIL import Image, ImageOps
import torchvision
from diff_jpeg import DiffJPEGCoding
import matplotlib.pyplot as plt
import os
from utils import load_prompt, recover_image

# Load the DiffJPEG module
diff_jpeg_coding_module = DiffJPEGCoding()


class StableDiffusionInpaint:
    """
    Class to load and configure the Stable Diffusion Inpainting model."""
    def __init__(self, model_version="stabilityai/stable-diffusion-2-inpainting", device="cuda", models_path=None):
        self.device = device
        self.model_version = model_version
        self.models_path = models_path
        self.dtype = torch.float16
        # Load models such as VAE, UNet, and StableDiffusionInpaintPipeline
        self.nf4_config = self._get_nf4()
        
        self.unet = self._load_unet()
        self.vae = self._load_vae()
        self.pipe_inpaint = self._initialize_sd_pipeline()
        self._configure_pipeline()
    
    def _get_nf4(self):
        """
        Get the configuration for 4-bit quantization using NF4 format."""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=self.dtype,
        )
    
    def _load_unet(self):
        """
        Load the UNet model with 4 bit quantization.
        """
        return UNet2DConditionModel.from_pretrained(
            self.model_version,
            subfolder="unet",
            quantization_config=self.nf4_config,
            torch_dtype=self.dtype,
            use_safetensors=True,
            cache_dir=self.models_path,
        )
    
    def _load_vae(self):
        """
        Load the VAE model with 4 bit quantization.
        """
        return AutoencoderKL.from_pretrained(
            self.model_version,
            subfolder="vae",
            quantization_config=self.nf4_config,
            torch_dtype=self.dtype,
            use_safetensors=True,
            cache_dir=self.models_path,
        )
    
    def _initialize_sd_pipeline(self):
        """
        Initialise the stable diffusion pipeline with the UNet and VAE models.
        """
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
        """
        Configure the pipeline by moving the models to CUDA and setting the safety checker to None.
        """
        self.pipe_inpaint.text_encoder = self.pipe_inpaint.text_encoder.to(self.device, self.dtype)
        self.pipe_inpaint.unet.to(memory_format=torch.channels_last)
        self.pipe_inpaint = self.pipe_inpaint.to(device=self.device, memory_format=torch.channels_last)
        self.pipe_inpaint.safety_checker = None
        self.pipe_inpaint.vae.requires_grad_(False)
        self.pipe_inpaint.unet.requires_grad_(False)
    
    def get_pipeline(self):
        return self.pipe_inpaint
    

class Inference:
    """
    Class to perform inference using the Stable Diffusion model.
    """
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
        """
        Load the Stable diffusion pipeline with specific version.
        """
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            model_version,
            variant="fp16",
            torch_dtype=torch.float16,
            use_safetensors=True,
            cache_dir=models_path,
        )
        # Load the VAE and UNet models to CUDA
        pipe = pipe.to(self.device)
        pipe.safety_checker = None
        for name, param in pipe.unet.named_parameters():
            param.requires_grad = False
        return pipe

    def load_images(self, test):
        """
        Load original, mask, and adversarial images for inference.
        """
        init_image = Image.open(f'{self.image_folder}original/{self.image_name}.png').convert('RGB').resize((512, 512))
        mask_image = ImageOps.invert(Image.open(f'{self.image_folder}masks/{self.image_name}_masked.png').convert('RGB')).resize((512, 512))
        adv_image = self.load_adversarial_image(test)
        return init_image, mask_image, adv_image

    def load_adversarial_image(self, test):
        """
        Load the adversarial protected image based on if the DDD_fast or JRAP method is used.
        """
        # Loads DDD_fast adversarial image without compression        
        if test==1:
            return Image.open(f'{self.testing_filename}/original_adversarial.png').resize((512, 512))
        # Loads DDD_fast adversarial image with compression
        if test==2:
            adv_image = torchvision.io.read_image(f'{self.testing_filename}/original_adversarial.png').float()[None]
            jpeg_quality = torch.tensor([50])
            image_coded = diff_jpeg_coding_module(image_rgb=adv_image, jpeg_quality=jpeg_quality)
            torch_image = image_coded.squeeze(0)
            to_pil = torchvision.transforms.ToPILImage()
            adv_image = to_pil(torch_image / 255).resize((512, 512))
            return adv_image
        # Loads JRAP adversarial image without compression
        if test==3:
            adv_image = Image.open(f'{self.testing_filename}/jrap_adversarial.png').resize((512, 512))
            return adv_image
        # Loads JRAP adversarial image with compression
        if test==4:
            adv_image = torchvision.io.read_image(f'{self.testing_filename}/jrap_adversarial.png').float()[None]
            jpeg_quality = torch.tensor([50])
            image_coded = diff_jpeg_coding_module(image_rgb=adv_image, jpeg_quality=jpeg_quality)
            torch_image = image_coded.squeeze(0)
            to_pil = torchvision.transforms.ToPILImage()
            adv_image = to_pil(torch_image / 255).resize((512, 512))
            return adv_image

    def infer_images(self):
        """
        Perform inference on the images using Stable diffusion inpainting models.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        prompts = load_prompt(f"{self.image_folder}prompts/{self.image_name}_prompts.txt")

        # Load images and generate inpainted images for each case.
        for test in self.test_cases:
            init_image, mask_image, adv_image = self.load_images(test)
            torch.manual_seed(self.seed)
            with torch.no_grad():
                for promptnum, prompt in enumerate(prompts):
                    print(self.seed)
                    torch.manual_seed(self.seed)
                    # Set predefined parameters for inpainting
                    strength, guidance_scale, num_inference_steps = 0.8, 7.5, 50

                    image_nat = self.generate_inpainted_image(init_image, init_image, mask_image, prompt, strength, guidance_scale, num_inference_steps)
                    torch.manual_seed(self.seed)
                    image_adv = self.generate_inpainted_image(init_image, adv_image, mask_image, prompt, strength, guidance_scale, num_inference_steps)
                    self.store_generated_images(test, promptnum, init_image, adv_image, image_nat, image_adv, prompt)

    def generate_inpainted_image(self, init_image, image, mask_image, prompt, strength, guidance_scale, num_inference_steps):
        """
        Generate and construct the inpainted image using the stable diffusion pipeline.
        """
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
        """
        Store the generated images in the specified directory. This follows a structure
        """
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
        # Create directories for saving images
        os.makedirs(f"{self.testing_filename}/original", exist_ok=True)
        image_nat.save(f"{self.testing_filename}/original/prompt{promptnum}.png")
        save_dirs = ["ddd_fast", "ddd_fast_compressed", "jrap", "jrap_compressed"]
        os.makedirs(f"{self.testing_filename}/{save_dirs[test-1]}", exist_ok=True)
        image_adv.save(f"{self.testing_filename}/{save_dirs[test-1]}/prompt{promptnum}.png")
        plt.savefig(f'{self.testing_filename}/{save_dirs[test-1]}/result_prompt{promptnum}.png')
        plt.clf()

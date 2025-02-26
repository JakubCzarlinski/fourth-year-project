from diffusers import StableDiffusionInpaintPipeline
import torch

# Load the U-Net
pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting")
unet = pipe.unet.to("cuda")

# Define a hook to log tensor shapes
def shape_hook(module, input, output, name):
    print(f"\nLayer: {name}")
    if isinstance(input, tuple):
        print(f"Input shapes: {[x.shape for x in input if x is not None]}")
    else:
        print(f"Input shape: {input.shape}")
    print(f"Output shape: {output.shape}")

# Attach hooks to key layers (e.g., downsampling, upsampling, mid-blocks)
hook_handles = []
for name, module in unet.named_modules():
    if "down_blocks" in name or "up_blocks" in name or "mid_block" in name:
        handle = module.register_forward_hook(
            lambda m, i, o, name=name: shape_hook(m, i, o, name)
        )
        hook_handles.append(handle)

# Run a dummy forward pass
with torch.no_grad():
    latent = torch.randn(320, 9, 3, 3).to("cuda")  # Latent space input (B, C, H, W)
    timestep = torch.tensor([50]).to("cuda")        # Example timestep
    encoder_hidden_states = torch.randn(1, 77, 1024).to("cuda")  # Text embeddings
    output = unet(latent, timestep, encoder_hidden_states).sample

# Remove hooks to clean up
for handle in hook_handles:
    handle.remove()
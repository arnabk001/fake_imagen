# https://github.com/runwayml/stable-diffusion#inpainting-with-stable-diffusion

from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",
    torch_dtype=torch.float16,
)
pipe.to("cuda")
prompt = "a red boeing airplane"

#image and mask_image should be PIL images.
#The mask structure is white for inpainting and black for keeping as is
image = Image.open('/home/ubuntu/arnab/MMFusion-IML/data/CocoGlide/real/airplane_44652.png')
mask_image = Image.open('/home/ubuntu/arnab/MMFusion-IML/data/CocoGlide/mask/airplane_44652_mask.png')
image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]
image.save("./airplane.png")

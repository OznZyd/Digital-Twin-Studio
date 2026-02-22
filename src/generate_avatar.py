from diffusers import StableDiffusionXLPipeline
import torch
import os
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()

hf_token = os.getenv("HF_Token")
login(hf_token)

os.makedirs("output", exist_ok=True)

print("Status: Loading model to MPS ...")
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/sdxl-turbo" ,
        torch_dtype=torch.float16,
        variant="fp16")

pipe.to("mps")


print("Status: Generating image ... This may take a moment")
prompt = "A professional portrait of a 33-year-old male engineer, short hair, cinematic lighting, 8k resolution, realistic skin texture"
negative_prompt = "cartoon, anime, illustration, painting, blurry, low quality, deformed"

image = pipe(
    prompt = prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=4,
    guidance_scale=0
).images[0]


print("Status: Saving image to output/avatar.png")
image.save("output/avatar.png")
print("Status: Task completed!")
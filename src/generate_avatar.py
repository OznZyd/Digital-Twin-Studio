import sys
from torchvision.transforms import functional as F

sys.modules['torchvision.transforms.functional_tensor'] = F

from gfpgan import GFPGANer
import torch
import os
import cv2
import numpy as np
from diffusers import StableDiffusionXLImg2ImgPipeline
from huggingface_hub import login
from dotenv import load_dotenv
from insightface.model_zoo import get_model
from insightface.app import FaceAnalysis
from PIL import Image


init_image = Image.open('data/me.jpeg')

init_image=init_image.convert("RGB")

gfpgan = GFPGANer(model_path='models/GFPGANv1.4.pth',
        upscale=2,
        arch= 'clean',
        channel_multiplier=2,
        device=torch.device('mps'))


app=FaceAnalysis(name='buffalo_l',
    providers=['CPUExecutionProvider'])

app.prepare(ctx_id=0,
    det_size=(640, 640))


source_img=cv2.imread("data/me.jpeg")

if source_img is None:
    print("File no found")
    exit()

faces=app.get(source_img)

if len(faces) == 0:
    print("Faces no found")
    exit()

source_faces = faces[0]

swapper=get_model('models/inswapper_128.onnx')

load_dotenv()

hf_token = os.getenv("HF_Token")
login(hf_token)

os.makedirs("output", exist_ok=True)

print("Status: Loading model to MPS ...")
pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/sdxl-turbo" ,
        torch_dtype=torch.float16,
        variant="fp16")

pipe.to("mps")


print("Status: Generating image ... This may take a moment")
prompt = "A high-quality portrait of a man in his early 30s, dark brown hair combed back, dark brown eyes, short neat beard, wearing a crisp white button-down shirt, cinematic lighting, sharp focus, highly detailed, 8k"
negative_prompt = "cartoon, anime, illustration, painting, blurry, low quality, deformed, thick lips, over-detailed eyes, plastic surgery look, botox, face fat, airbrushed, glowing skin:1.2), blurry, out of focus"

image = pipe(
    prompt = prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=4,
    guidance_scale=0,
    image=init_image,
    strength= 0.55
).images[0]



target_matris=np.array(image)

bgr_img = cv2.cvtColor(target_matris, cv2.COLOR_RGB2BGR)

target_faces = app.get(bgr_img)

if len(target_faces) > 0:

    res_img = bgr_img

    for face in target_faces:
        res_img = swapper.get(res_img,face, source_faces, paste_back=True)
    

    _, _, restored_image = gfpgan.enhance(res_img,
                            has_aligned=True,
                            only_center_face=False,
                            paste_back=True,
                            weight= 0.3)
    
    if restored_image is None:
        print("Warning: GFPGAN failed, using swapped image (res_img) instead")
        restored_image = res_img
        
    final_rgb=cv2.cvtColor(restored_image, cv2.COLOR_BGR2RGB)
    result_image = Image.fromarray(final_rgb)
    print("Status: Saving SWAPPED image to output/avatar_final.png")
    result_image.save("output/avatar_final.png")
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    restored_image = cv2.filter2D(restored_image, -1, kernel)
    
else:
    print("No faces detected in generated image")


print("Status: Saving image to output/avatar.png")
image.save("output/avatar.png")
print("Status: Task completed!")

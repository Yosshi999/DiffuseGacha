from base64 import b64encode, b64decode
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import numpy as np
import cv2
import torch
import safetensors.torch
import json

def mitsua_credit(image: Image.Image) -> Image.Image:
    image = np.array(image)
    image = cv2.putText(image, "Generated by Mitsua Likes", (5, image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)
    image = cv2.putText(image, "Generated by Mitsua Likes", (5, image.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    image = Image.fromarray(image)
    return image

def save_image_with_metadata(image: Image.Image, path: str, latent: torch.Tensor, generation_config: dict):
    tensordata = safetensors.torch.save({"latent": latent})
    metadata = PngInfo()
    metadata.add_text(
        "latent_tensor",
        b"data:application/octet-stream;base64," + b64encode(tensordata),
        zip=True
    )
    metadata.add_text(
        "generation_config",
        json.dumps(generation_config)
    )
    image.save(path, pnginfo=metadata)

def load_image_with_metadata(path: str):
    image = Image.open(path)
    latent = safetensors.torch.load(b64decode(image.info["latent_tensor"].split(",", 1)[1]))["latent"]
    generation_config = json.loads(image.info["generation_config"])
    # image is tainted by mitsua credits
    return None, latent, generation_config
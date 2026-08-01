import os
import logging
import textwrap
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms

def blip_inference_on_image(image_path, model, processor):
    """Runs image captioning inference on an input image.
    Input
        image_path(path): path of image to run inference on
        huggingface_model_name(str): Name of huggingface model to use
    Output
        Sample image of image and image caption.
    """
    image = Image.open(image_path).convert("RGB")

    imgtensor_bb = processor(images=image, return_tensors="pt").pixel_values #returns image values as tensor
    txttokens_bb = model.generate(pixel_values=imgtensor_bb, max_length=50) #image encoder encodes image tensor --> language decoder  generates text tokens using image tokens as context
    caption_bb = processor.batch_decode(txttokens_bb, skip_special_tokens=True)[0] #decode token id's into words
    wrapped_caption = "\n".join(textwrap.wrap(caption_bb, width=60))
    plt.imshow(np.asarray(image))
    plt.xlabel(wrapped_caption, fontsize=10, fontweight='bold')
    plt.title(f"{model.name_or_path.split("/")[-1]} Image Caption")
    plt.savefig(f"{model.name_or_path.split("/")[-1]}_image_caption.png", bbox_inches='tight', pad_inches=0.5)
    plt.close()



def main():
    image_path = "data/images/art2_028.jpg"
    model_finetuned = BlipForConditionalGeneration.from_pretrained("blip-finetuned-model")
    processor_finetuned = BlipProcessor.from_pretrained("blip-finetuned-model")

    # Run inference on one sample image to visualize the result.
    blip_inference_on_image(image_path, model_finetuned, processor_finetuned)

if __name__ == "__main__":
    main()
#%%
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import logging
import textwrap
from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from transformers import TrainingArguments, Trainer, BlipProcessor, BlipForConditionalGeneration
from datasets import Dataset
from sklearn.model_selection import train_test_split

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)

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

def preprocess(example, image_dir):
    image_path = os.path.join(image_dir, example['image'])
    image = Image.open(image_path).convert("RGB")

    encoding = processor(
        images=image,
        text=example['text_output'],
        padding="max_length",
        max_length=50,
        truncation=True,
        return_tensors="pt"
    )
    encoding = {k: v.squeeze(0) for k, v in encoding.items()}
    encoding["labels"] = encoding["input_ids"].clone()
    return encoding

def hf_train(model, processor, train_dataset, val_dataset):
    """BlipForConditionalGeneration wraps BlipTextLMHeadModel which is trained with torch.nn.CrossEntropyLoss.
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=self.label_smoothing)
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
    Inputs
    - Image is encoded
    - Text caption gets tokenized: ["A", "dog", "running", "in", "park"]
    - Labels are the same sequence shifted: ["dog", "running", "in", "park", "[EOS]"]
    Prediction Generation
    - Model outputs prediction_scores with shape [batch_size, seq_len, vocab_size]
    - These are raw logits for each token position predicting the next token.
    Label Shifting
    - Predictions: Model predicts what comes after each input token
    - Labels: Ground truth of what should come next
    CrossEntropyLoss
    - For each position in the sequence, compare predicted token probabilities against groundtruth next token.
    - Computes loss for ALL positions simultaneously
    - Averages the loss across all positions and batch items
    """
    training_args = TrainingArguments(
        output_dir="./blip-finetuned-checkpoints",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy = "epoch",
        eval_steps = 50,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=processor,
    )
    metrics = trainer.evaluate()
    metrics["step"] = 0
    trainer.state.log_history.append(metrics)

    trainer.train()
    logs = trainer.state.log_history
    train_steps = []
    train_losses = []
    val_steps = []
    val_losses = []
    for log in logs:
        if "loss" in log:
            train_steps.append(log["step"])
            train_losses.append(log["loss"])
        if "eval_loss" in log:  # validation loss
            val_steps.append(log["step"])
            val_losses.append(log["eval_loss"])
    model.save_pretrained("blip-finetuned-model")
    processor.save_pretrained("blip-finetuned-model")

    plt.figure(figsize=(8, 4))
    plt.plot(train_steps, train_losses, label="Training Loss")
    plt.plot(val_steps, val_losses, label="Validation Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.legend()
    plt.savefig("TrainingLossCurve.png")
    plt.close()

def data_split(image_dir, minimap_annotations_path):
    with open(minimap_annotations_path, "r") as f:
        annotations = json.load(f)
        train_data, eval_data = train_test_split(annotations, test_size=0.2, train_size=0.8)
        val_data, test_data = train_test_split(eval_data, test_size=0.5, train_size=0.5)
        train_dataset = Dataset.from_list(train_data).map(lambda x: preprocess(x, image_dir))
        val_dataset = Dataset.from_list(val_data).map(lambda x: preprocess(x, image_dir))
        test_dataset = Dataset.from_list(test_data).map(lambda x: preprocess(x, image_dir))
    return train_dataset, val_dataset, test_dataset, annotations

def test(model_finetuned, processor_finetuned, testset, annotations):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_finetuned.to(device)
    model_finetuned.eval()

    captions_finetuned = {}
    for annot in tqdm(testset):  # test_data is your list of dicts
        img_path = os.path.join(image_dir, annot["image"])
        img = Image.open(img_path).convert("RGB")

        inputs = processor_finetuned(images=img, return_tensors="pt").to(device)
        out = model_finetuned.generate(**inputs, max_length=50)
        caption_bf = processor_finetuned.batch_decode(out, skip_special_tokens=True)[0]
        captions_finetuned[annot["image"]] = caption_bf

    gt_judgements = {}
    for entry in annotations:
        if entry["image"] in testset['image']:
            first_word = entry["text_output"].split('.')[0].lower().strip()
            gt_judgements[entry["image"]] = first_word

    blip_judgements = {
        k: v.split('.')[0].strip().lower()
        for k, v in captions_finetuned.items()
    }

    matches = 0
    total = 0
    for img in gt_judgements:
        if img in blip_judgements:
            total += 1
            if blip_judgements[img] == gt_judgements[img]:
                matches += 1
    logger.info(blip_judgements)
    logger.info(gt_judgements)
    logger.info(f"Match count: {matches}/{total} = {matches / total:.2f}")



#processor class internally pre-processes the image and BPE tokenizes the caption
image_path = "data/images/art21_019.jpg"
image_dir = "data/images"
annotations_path = "data/minimap_annot.json"

model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# blip_inference_on_image(image_path, model, processor)
trainset, valset, testset, annotations = data_split(image_dir, annotations_path)
hf_train(model, processor, trainset, valset)

model_finetuned = BlipForConditionalGeneration.from_pretrained("blip-finetuned-model")
processor_finetuned = BlipProcessor.from_pretrained("blip-finetuned-model")
blip_inference_on_image(image_path, model_finetuned, processor_finetuned)

test(model_finetuned, processor_finetuned, testset, annotations)


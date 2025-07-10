# model/caption_generator.py

from transformers import (
    BlipProcessor, BlipForConditionalGeneration,
    VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
)
from PIL import Image
import torch
import gc
import logging

# Suppress Hugging Face warnings for cleaner output
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("transformers.image_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

class CaptionGenerator:
    def __init__(self, model_name: str, device: str = None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if device is None else device
            
        self.model_name = model_name
        self.processor = None
        self.model = None
        self.tokenizer = None # Needed for ViT-GPT2

        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.model_name} to device: {self.device}...")

        if self.model_name == "blip-image-captioning-base":
            hf_model_id = "Salesforce/blip-image-captioning-base"
            self.processor = BlipProcessor.from_pretrained(hf_model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(hf_model_id).to(self.device)

        elif self.model_name == "noamrot/FuseCap":
            hf_model_id = "noamrot/FuseCap"
            self.processor = BlipProcessor.from_pretrained(hf_model_id)
            self.model = BlipForConditionalGeneration.from_pretrained(hf_model_id).to(self.device)

        elif self.model_name == "nlpconnect/vit-gpt2-image-captioning":
            hf_model_id = "nlpconnect/vit-gpt2-image-captioning"
            self.processor = ViTImageProcessor.from_pretrained(hf_model_id)
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            self.model = VisionEncoderDecoderModel.from_pretrained(hf_model_id).to(self.device)
            
            if self.model.config.decoder_start_token_id is None:
                self.model.config.decoder_start_token_id = self.tokenizer.bos_token_id
            if self.model.config.eos_token_id is None:
                self.model.config.eos_token_id = self.tokenizer.eos_token_id

        else:
            raise ValueError(f"Unsupported model: {self.model_name}. Please choose from the allowed list.")
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()

    def generate_caption(self, image: Image.Image) -> str:
        if self.processor is None or self.model is None:
            raise RuntimeError("Model and processor not loaded. Call _load_model() first.")

        max_new_tokens = 50
        num_beams_blip_fusecap = 3 
        num_beams_vitgpt2 = 1     
        early_stopping = True

        english_caption = ""
        if self.model_name in ["blip-image-captioning-base", "noamrot/FuseCap"]:
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams_blip_fusecap,
                early_stopping=early_stopping
            )
            english_caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            if english_caption.startswith(","):
                english_caption = english_caption.lstrip(', ').strip()


        elif self.model_name == "nlpconnect/vit-gpt2-image-captioning":
            pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)
            generated_ids = self.model.generate(
                pixel_values=pixel_values,
                max_length=max_new_tokens, # ViT-GPT2 uses max_length
                num_beams=num_beams_vitgpt2,
                do_sample=False,
                early_stopping=early_stopping
            )
            english_caption = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        else:
            raise ValueError(f"Unknown model type for generation: {self.model_name}")

        return english_caption
# model/caption_generator.py

from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    AutoProcessor, AutoModelForCausalLM, # For GIT
    AutoModelForSeq2SeqLM # For OFA
)
from PIL import Image
import torch
import gc # For memory management

class CaptionGenerator:
    """
    Manages loading and generating captions from various Vision-Language Models.
    """
    def __init__(self, model_name: str, device: str = None):
        """
        Initializes the CaptionGenerator with a specific model.

        Args:
            model_name (str): Name of the model to load (e.g., "blip2-flan-t5-xxl", "ofa-base", "git-large").
            device (str): Device to use ('cuda' or 'cpu'). Defaults to auto-detection.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model_name = model_name
        self.processor = None
        self.model = None

        self._load_model()

    def _load_model(self):
        """Loads the specified model and its corresponding processor."""
        print(f"Loading model: {self.model_name} to device: {self.device}...")

        if "blip2" in self.model_name:
            if self.model_name == "blip2-flan-t5-xxl":
                hf_model_id = "Salesforce/blip2-flan-t5-xxl"
            elif self.model_name == "blip2-opt-2.7b":
                hf_model_id = "Salesforce/blip2-opt-2.7b"
            else:
                raise ValueError(f"Unsupported BLIP-2 model variant: {self.model_name}")

            self.processor = Blip2Processor.from_pretrained(hf_model_id)
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                hf_model_id,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
                # load_in_8bit=True if self.device == 'cuda' else False # Chỉ sử dụng nếu có bitsandbytes và GPU
            ).to(self.device)

        elif "git" in self.model_name:
            if self.model_name == "git-base":
                hf_model_id = "microsoft/git-base"
            elif self.model_name == "git-large":
                hf_model_id = "microsoft/git-large"
            else:
                raise ValueError(f"Unsupported GIT model variant: {self.model_name}")

            self.processor = AutoProcessor.from_pretrained(hf_model_id)
            self.model = AutoModelForCausalLM.from_pretrained(hf_model_id).to(self.device)

        elif "ofa" in self.model_name:
            if self.model_name == "ofa-base":
                hf_model_id = "OFA-Sys/ofa-base"
            elif self.model_name == "ofa-large":
                hf_model_id = "OFA-Sys/ofa-large"
            else:
                raise ValueError(f"Unsupported OFA model variant: {self.model_name}")
            
            self.processor = AutoProcessor.from_pretrained(hf_model_id)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_id).to(self.device)

        else:
            raise ValueError(f"Unsupported model: {self.model_name}. Please choose from blip2-*, git-*, ofa-*")

        print(f"Model {self.model_name} loaded successfully.")
        
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            gc.collect()


    def generate_caption(self, image: Image.Image) -> str:
        """
        Generates an English caption for the given image.
        """
        if self.processor is None or self.model is None:
            raise RuntimeError("Model and processor not loaded. Call _load_model() first.")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        max_new_tokens = 50
        num_beams = 5
        early_stopping = True

        if "blip2" in self.model_name:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping
            )
        elif "git" in self.model_name:
            generated_ids = self.model.generate(
                pixel_values=inputs.pixel_values,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping
            )
        elif "ofa" in self.model_name:
            ofa_task_prefix = "What does the image describe?" # OFA often benefits from a task prefix
            input_ids = self.processor(text=ofa_task_prefix, return_tensors="pt").input_ids.to(self.device)
            
            generated_ids = self.model.generate(
                input_ids=input_ids,
                patch_images=inputs.pixel_values, # OFA uses patch_images
                no_repeat_ngram_size=3,
                num_beams=num_beams,
                max_length=max_new_tokens,
                early_stopping=early_stopping
            )
        else: # Fallback
             generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=early_stopping
            )

        caption = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return caption
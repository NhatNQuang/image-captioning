# app.py

import gradio as gr
from PIL import Image
import os
import torch
import gc
from typing import Dict, List
import logging 

# --- IMPORT CÁC MODULE CỦA DỰ ÁN ---
from model.caption_generator import CaptionGenerator
from model.translator import Translator
from utils.data_loader import load_ground_truth_captions_for_test, get_ground_truth_for_single_image
from utils.evaluator import evaluate_captions
# --- KẾT THÚC IMPORT CÁC MODULE CỦA DỰ ÁN ---

# Suppress Hugging Face warnings for cleaner output
logging.getLogger("transformers.generation_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.file_download").setLevel(logging.ERROR)
logging.getLogger("transformers.image_utils").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# Initialize Translator once
translator = Translator()

# Define the models to be used (fixed list)
CAPTION_MODELS = [
    "blip-image-captioning-base",
    "noamrot/FuseCap",
    "nlpconnect/vit-gpt2-image-captioning",
]

# Global dictionary to store CaptionGenerator instances for each model
model_generators: Dict[str, 'CaptionGenerator'] = {} 

def get_or_load_generator(model_name: str) -> CaptionGenerator:
    if model_name not in model_generators or model_generators[model_name] is None:
        print(f"Loading {model_name} for the first time...")
        try:
            model_generators[model_name] = CaptionGenerator(model_name=model_name)
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to initialize CaptionGenerator for {model_name}: {e}")
            model_generators[model_name] = None
            raise
    return model_generators[model_name]

def generate_captions_for_all_models(image, selected_sample_image_path=None):
    if image is None:
        return "Please upload an image or select a sample image.", *(["", ""] * len(CAPTION_MODELS))

    output_captions = []
    for model_name in CAPTION_MODELS:
        generator = get_or_load_generator(model_name)
        try:
            english_caption = generator.generate_caption(image)
            vietnamese_caption = translator.translate(english_caption)
            output_captions.extend([english_caption, vietnamese_caption])
        except Exception as e:
            output_captions.extend([f"Error: {e}", f"Lỗi: {e}"])
            print(f"Error generating caption for {model_name}: {e}")

    ground_truth_text = ""
    if selected_sample_image_path:
        # Handle temp path or dict from Gradio Gallery
        if isinstance(selected_sample_image_path, dict) and 'image' in selected_sample_image_path:
            image_filename = selected_sample_image_path['image'].get('orig_name', os.path.basename(selected_sample_image_path['image']['path']))
        elif isinstance(selected_sample_image_path, dict) and 'name' in selected_sample_image_path:
            image_filename = os.path.basename(selected_sample_image_path['name'])
        else:
            image_filename = os.path.basename(selected_sample_image_path)
            
        gt_comments = get_ground_truth_for_single_image("./captions.xlsx", image_filename)
        if gt_comments and gt_comments != ["N/A"]:
            ground_truth_text = "Ground Truth:\n" + "\n".join([f"- {cmt}" for cmt in gt_comments])
        else:
            ground_truth_text = "Ground Truth: Not found for this image."

    return ground_truth_text, *output_captions

def load_sample_image_and_trigger_generation(select_data: gr.SelectData):
    print(f"Debug: select_data structure: {select_data.__dict__}")  # Debug select_data content
    image_path = None

    if select_data and hasattr(select_data, 'value'):
        if isinstance(select_data.value, dict):
            # Handle Gradio Gallery's nested image path
            if 'image' in select_data.value and 'path' in select_data.value['image']:
                image_path = select_data.value['image']['path']
            elif 'path' in select_data.value:
                image_path = select_data.value['path']
            elif 'name' in select_data.value:
                image_path = select_data.value['name']
        elif isinstance(select_data.value, str):
            image_path = select_data.value
    elif select_data and hasattr(select_data, 'name') and select_data.name:
        image_path = select_data.name
    elif isinstance(select_data, str):
        image_path = select_data
    elif isinstance(select_data, dict) and 'name' in select_data:
        image_path = select_data['name']

    if image_path is None:
        print(f"No valid image path extracted from select_data: {select_data}")
        return None, None

    # Verify the image exists
    if os.path.exists(image_path):
        try:
            image_pil = Image.open(image_path).convert("RGB")
            # Use the original filename for ground truth lookup
            original_filename = os.path.basename(image_path)
            # Optionally, map to test_images directory if needed
            test_images_path = os.path.join("test_images", original_filename)
            return image_pil, test_images_path if os.path.exists(test_images_path) else image_path
        except Exception as e:
            print(f"Error loading sample image {image_path}: {e}")
            return None, None
    print(f"Sample image path not found or invalid: {image_path}")
    return None, None

def run_total_evaluation():
    evaluation_results_text = "--- Overall BLEU Evaluation Results (First 10 Test Images) ---\n\n"

    evaluable_models = []
    for model_name in CAPTION_MODELS:
        try:
            generator = get_or_load_generator(model_name)
            if generator is not None:
                evaluable_models.append(model_name)
            else:
                evaluation_results_text += f"--- Model: {model_name} ---\nStatus: Model failed to load, skipping evaluation.\n\n"
        except Exception as e:
            evaluation_results_text += f"--- Model: {model_name} ---\nStatus: Model failed to load due to error: {e}, skipping evaluation.\n\n"
            print(f"Skipping evaluation for {model_name} due to loading error: {e}")

    if not evaluable_models:
        return evaluation_results_text + "No models successfully loaded for evaluation."

    test_images_dir = "./test_images"
    all_test_image_filenames = sorted([f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    limited_test_image_filenames = all_test_image_filenames[:10]

    try:
        ground_truth_data = load_ground_truth_captions_for_test(
            file_path="./captions.xlsx", 
            test_image_names=limited_test_image_filenames
        )
        if not ground_truth_data:
            return evaluation_results_text + "No ground truth data available for evaluation."
    except Exception as e:
        return evaluation_results_text + f"Error loading ground truth data: {e}"

    for model_name in evaluable_models:
        generator = model_generators.get(model_name)
        if generator is None:
            evaluation_results_text += f"--- Model: {model_name} ---\nStatus: Model not loaded, skipping evaluation.\n\n"
            continue

        def caption_func_for_eval(image: Image.Image) -> str:
            return generator.generate_caption(image)

        results = evaluate_captions(
            caption_func_for_eval,
            ground_truth_data,
            test_images_dir
        )

        evaluation_results_text += f"--- Model: {model_name} ---\n"
        if 'bleu' in results:
            evaluation_results_text += f"BLEU Score: {results['bleu']:.2f}\n\n"
        elif 'status' in results:
            evaluation_results_text += f"Status: {results['status']}\n\n"
        else:
            evaluation_results_text += "No BLEU score calculated. Check for errors.\n\n"

    return evaluation_results_text

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("# Multilingual Image Captioning Application")
    gr.Markdown("Upload your image or select a sample to generate captions in English and Vietnamese from multiple models.")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Image")
            
            sample_images_dir = "test_images"
            all_sample_image_paths = sorted([os.path.join(sample_images_dir, f) for f in os.listdir(sample_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            limited_sample_image_paths = all_sample_image_paths[:10]

            gr.Markdown("Or select a sample image (from the first 10 test images):")
            sample_gallery = gr.Gallery(
                value=[(img_path, os.path.basename(img_path)) for img_path in limited_sample_image_paths],
                columns=[5],
                rows=[2],
                height=250,
                object_fit="contain",
                label="Sample Image Gallery",
                type="filepath"
            )
            
            submit_btn = gr.Button("Generate Captions for All Models")

        with gr.Column():
            ground_truth_output = gr.Markdown("## Ground Truth Captions (For Sample Images)", label="Ground Truth")

            model_output_components = []
            for model_name in CAPTION_MODELS:
                gr.Markdown(f"### Captions from {model_name}")
                with gr.Row():
                    english_out = gr.Textbox(label="English Caption", interactive=False)
                    vietnamese_out = gr.Textbox(label="Vietnamese Caption", interactive=False)
                    model_output_components.extend([english_out, vietnamese_out])

    submit_btn.click(
        fn=generate_captions_for_all_models,
        inputs=[image_input, gr.State(None)],
        outputs=[ground_truth_output, *model_output_components]
    )
    
    selected_sample_image_state = gr.State(None)
    sample_gallery.select(
        fn=load_sample_image_and_trigger_generation,
        outputs=[image_input, selected_sample_image_state]
    ).then(
        fn=generate_captions_for_all_models,
        inputs=[image_input, selected_sample_image_state],
        outputs=[ground_truth_output, *model_output_components]
    )

    gr.Markdown("## Overall Model Evaluation")
    run_eval_btn = gr.Button("Run BLEU Evaluation (on first 10 test images)")
    evaluation_summary_output = gr.Textbox(label="Evaluation Summary", interactive=False, lines=10)

    run_eval_btn.click(
        fn=run_total_evaluation,
        inputs=[],
        outputs=[evaluation_summary_output]
    )

demo.launch(share=True)
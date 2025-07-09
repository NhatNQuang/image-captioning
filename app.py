# app.py

import gradio as gr
from PIL import Image
import os
import torch
import gc # Để quản lý bộ nhớ

from model.caption_generator import CaptionGenerator
from model.translator import Translator
from utils.data_loader import load_ground_truth_captions_for_test
from utils.evaluator import evaluate_captions

# Khởi tạo Translator (chỉ cần 1 instance)
translator = Translator()

# Danh sách các mô hình có thể lựa chọn từ Hugging Face
AVAILABLE_MODELS = [
    "blip2-flan-t5-xxl",
    "blip2-opt-2.7b",
    "git-base",
    "ofa-base",
]

# Biến toàn cục để lưu trữ instance của CaptionGenerator đang hoạt động
current_caption_generator = None
current_model_name = None

def get_caption_generator(model_name: str):
    global current_caption_generator, current_model_name
    
    if current_caption_generator is None or current_model_name != model_name:
        print(f"Switching model to: {model_name}")
        if current_caption_generator:
            del current_caption_generator
            torch.cuda.empty_cache()
            gc.collect()

        current_caption_generator = CaptionGenerator(model_name=model_name)
        current_model_name = model_name
    return current_caption_generator

def generate_captions_and_translate(image, model_choice):
    if image is None:
        return "Vui lòng tải lên một ảnh hoặc chọn ảnh mẫu.", ""

    generator = get_caption_generator(model_choice)
    
    try:
        english_caption = generator.generate_caption(image)
        vietnamese_caption = translator.translate(english_caption)
        
        return english_caption, vietnamese_caption
    except Exception as e:
        return f"Lỗi khi sinh caption: {e}", "Không thể dịch do lỗi sinh caption."

def load_sample_image(image_file):
    if isinstance(image_file, dict) and 'name' in image_file:
        image_path = image_file['name']
    elif isinstance(image_file, str):
        image_path = image_file
    else:
        return None

    if os.path.exists(image_path):
        return Image.open(image_path)
    return None

def run_evaluation(model_choice):
    generator = get_caption_generator(model_choice)
    
    test_images_dir = "./test_images"
    test_image_filenames = [f for f in os.listdir(test_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    try:
        ground_truth_data = load_ground_truth_captions_for_test(
            file_path="./captions.xlsx", 
            test_image_names=test_image_filenames
        )
    except Exception as e:
        return f"Lỗi khi tải dữ liệu đánh giá từ captions.xlsx: {e}"

    results = evaluate_captions(generator, ground_truth_data, test_images_dir)
    
    results_text = f"--- Kết quả Đánh giá cho Mô hình: {model_choice} ---\n\n"
    if 'bleu' in results:
        results_text += f"Điểm BLEU (sacrebleu): {results['bleu']:.2f}\n"
    elif 'status' in results:
        results_text += results['status']
    else:
        results_text += "Không thể tính toán điểm. Vui lòng kiểm tra log terminal để biết lỗi.\n"
        
    return results_text

with gr.Blocks() as demo:
    gr.Markdown("# Ứng dụng Image Captioning Song ngữ")
    gr.Markdown("Tải lên ảnh của bạn hoặc chọn ảnh mẫu để sinh caption tiếng Anh và tiếng Việt.")

    with gr.Row():
        with gr.Column():
            model_selector = gr.Dropdown(
                choices=AVAILABLE_MODELS,
                value=AVAILABLE_MODELS[0],
                label="Chọn Mô hình Captioning",
                interactive=True
            )
            image_input = gr.Image(type="pil", label="Tải ảnh lên")
            
            sample_images_dir = "test_images"
            sample_images = [os.path.join(sample_images_dir, f) for f in os.listdir(sample_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            gr.Markdown("Hoặc chọn ảnh mẫu (từ bộ test 10 ảnh):")
            sample_gallery = gr.Gallery(
                value=[(img, os.path.basename(img)) for img in sample_images],
                columns=[5],
                rows=[1],
                height=150,
                object_fit="contain",
                label="Thư viện ảnh mẫu"
            )
            
            submit_btn = gr.Button("Sinh Caption")

        with gr.Column():
            english_output = gr.Textbox(label="Caption (Tiếng Anh)")
            vietnamese_output = gr.Textbox(label="Caption (Tiếng Việt)")
            
    with gr.Row():
        gr.Markdown("## Đánh giá chất lượng mô tả trên bộ Test 10 ảnh")
        run_evaluation_btn = gr.Button("Chạy Đánh giá (Tính BLEU)")
        evaluation_output = gr.Textbox(label="Kết quả Đánh giá", lines=10, show_copy_button=True)

    submit_btn.click(
        fn=generate_captions_and_translate,
        inputs=[image_input, model_selector],
        outputs=[english_output, vietnamese_output]
    )
    
    sample_gallery.select(
        fn=load_sample_image,
        inputs=gr.SelectedData(),
        outputs=[image_input]
    ).then(
        fn=generate_captions_and_translate,
        inputs=[image_input, model_selector],
        outputs=[english_output, vietnamese_output]
    )

    run_evaluation_btn.click(
        fn=run_evaluation,
        inputs=[model_selector],
        outputs=evaluation_output
    )

demo.launch(share=False)
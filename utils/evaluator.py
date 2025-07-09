# utils/evaluator.py
from evaluate import load
from PIL import Image
import os

def evaluate_captions(caption_generator_instance, ground_truth_data: dict, images_dir: str) -> dict:
    """
    Evaluates caption quality using metrics like BLEU.

    Args:
        caption_generator_instance: An instance of CaptionGenerator.
        ground_truth_data (dict): Dictionary where keys are image_names and values are lists of reference captions.
        images_dir (str): Path to the directory containing test images.

    Returns:
        dict: Evaluation results (e.g., {'bleu': score}).
    """
    bleu_metric = load("sacrebleu")
    
    predictions = []
    references_for_metrics = [] # List of lists of strings, for sacrebleu

    sorted_image_names = sorted(ground_truth_data.keys())

    for img_name in sorted_image_names:
        ref_captions = ground_truth_data[img_name]
        
        if not ref_captions: # Skip if no reference captions available
            print(f"Skipping {img_name}: No reference captions found.")
            continue

        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"Warning: Image {img_name} not found at {img_path}. Skipping evaluation for this image.")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            predicted_caption = caption_generator_instance.generate_caption(image)

            predictions.append(predicted_caption)
            references_for_metrics.append(ref_captions) # Add list of references here

        except Exception as e:
            print(f"Error processing {img_name} for evaluation: {e}")
            continue

    results = {}
    if predictions and references_for_metrics:
        bleu_score = bleu_metric.compute(predictions=predictions, references=references_for_metrics)
        results['bleu'] = bleu_score['bleu']
    else:
        results['status'] = "Không có đủ dữ liệu để đánh giá hoặc xảy ra lỗi trong quá trình sinh caption."

    return results
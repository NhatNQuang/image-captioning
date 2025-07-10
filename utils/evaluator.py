# utils/evaluator.py

from evaluate import load
from PIL import Image
import os
from typing import Callable, List, Dict

def evaluate_captions(caption_func: Callable[[Image.Image], str], ground_truth_data: Dict[str, List[str]], images_dir: str) -> Dict[str, float]:
    """
    Evaluates caption quality using metrics like BLEU.

    Args:
        caption_func (Callable[[Image.Image], str]): A function that takes a PIL Image
                                                     and returns a string caption.
        ground_truth_data (dict): Dictionary where keys are image_names and values are lists of reference captions.
        images_dir (str): Path to the directory containing test images.

    Returns:
        dict: Evaluation results (e.g., {'bleu': score}).
    """
    bleu_metric = load("sacrebleu")
    
    predictions = []
    references_for_metrics = []

    sorted_image_names = sorted(ground_truth_data.keys())

    for img_name in sorted_image_names:
        ref_captions = ground_truth_data[img_name]
        
        if not ref_captions:
            print(f"[Eval WARN]: Skipping {img_name}: No reference captions found in ground truth.")
            continue

        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            print(f"[Eval WARN]: Image {img_name} not found at {img_path}. Skipping evaluation for this image.")
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            if not hasattr(image, 'filename'):
                image.filename = img_name
            
            predicted_caption = caption_func(image)

            predictions.append(predicted_caption)
            references_for_metrics.append(ref_captions)

        except Exception as e:
            print(f"[Eval ERROR]: Failed to process image '{img_name}'. Error: {e}")
            continue

    results = {}
    if predictions and references_for_metrics and len(predictions) == len(references_for_metrics) and len(predictions) > 0:
        bleu_score = bleu_metric.compute(predictions=predictions, references=references_for_metrics)
        
        if 'score' in bleu_score:
            results['bleu'] = bleu_score['score'] # score is already 0-100
        else:
            results['status'] = "Error: 'score' key not found in BLEU evaluation results."
    else:
        results['status'] = "No sufficient data for evaluation or errors occurred during caption generation."

    return results
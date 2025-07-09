# utils/data_loader.py
import pandas as pd
import os

def load_ground_truth_captions_for_test(file_path: str, test_image_names: list) -> dict:
    """
    Reads the .xlsx file containing ground truth captions and formats them.
    Filters for specific test image names and expects multiple comments per image.

    Args:
        file_path (str): Path to the captions.xlsx file.
        test_image_names (list): A list of image filenames expected in the test set.

    Returns:
        dict: Dictionary with image_name as keys and a list of English captions as values.
              Example: {'image_01.jpg': ['caption_1', 'caption_2', ...], ...}
    """
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File captions.xlsx not found at: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading captions.xlsx: {e}")

    required_cols = ['image_name', 'comment']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Required columns {required_cols} not found in captions.xlsx")

    # Filter DataFrame to include only test images
    df_test = df[df['image_name'].isin(test_image_names)].reset_index(drop=True)

    # Group by image_name and collect all comments into a list
    ground_truth_data = df_test.groupby('image_name')['comment'].apply(list).to_dict()

    # Ensure all test_image_names have entries in the ground_truth_data
    for img_name in test_image_names:
        if img_name not in ground_truth_data:
            print(f"Warning: No ground truth captions found for {img_name} in captions.xlsx. Skipping.")
            ground_truth_data[img_name] = [] # Add empty list to avoid KeyError

    return ground_truth_data
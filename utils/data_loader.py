# utils/data_loader.py

import pandas as pd
import os
from typing import List, Dict

def load_ground_truth_captions_for_test(file_path: str, test_image_names: List[str], limit: int = None) -> Dict[str, List[str]]:
    try:
        df = pd.read_excel(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"File captions.xlsx not found at: {file_path}")
    except Exception as e:
        raise Exception(f"Error reading captions.xlsx: {e}")

    required_cols = ['image_name', 'comment']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Required columns {required_cols} not found in captions.xlsx")

    df_test = df[df['image_name'].isin(test_image_names)].reset_index(drop=True)

    if limit is not None and limit > 0:
        unique_image_names = df_test['image_name'].unique()
        limited_image_names = unique_image_names[:limit]
        df_test = df_test[df_test['image_name'].isin(limited_image_names)].reset_index(drop=True)
    else:
        limited_image_names = test_image_names # If no limit, use all specified test_image_names

    ground_truth_data = df_test.groupby('image_name')['comment'].apply(list).to_dict()

    for img_name in limited_image_names:
        if img_name not in ground_truth_data:
            print(f"Warning: No ground truth captions found for {img_name} in captions.xlsx. Skipping.")
            ground_truth_data[img_name] = []

    return ground_truth_data

def get_ground_truth_for_single_image(file_path: str, image_name: str) -> List[str]:
    try:
        df = pd.read_excel(file_path)
        comments = df[df['image_name'] == image_name]['comment'].tolist()
        return [str(c) for c in comments] # Ensure comments are strings
    except Exception as e:
        print(f"Error loading ground truth for {image_name}: {e}")
        return ["N/A"]
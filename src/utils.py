import numpy as np
import pandas as pd
import cv2
from PIL import Image
from typing import List, Dict
import re
from Levenshtein import ratio
import yaml

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
    return configs

def intersect(box_a: tuple, box_b: tuple) -> bool:
    """
    Check if two boxes intersect (overlap).

    Args:
    box_a (tuple): Tuple representing the coordinates of box A in the format (left, top, right, bottom).
    box_b (tuple): Tuple representing the coordinates of box B in the format (left, top, right, bottom).

    Returns:
    bool: True if the boxes intersect, False otherwise.
    """
    a_left, a_top, a_right, a_bottom = box_a
    b_left, b_top, b_right, b_bottom = box_b
    return not (a_right < b_left or a_left > b_right or a_bottom < b_top or a_top > b_bottom)

def get_intersection(box_a: tuple, box_b: tuple) -> tuple:
    """
    Get the intersection box of two boxes.

    Args:
    box_a (tuple): Tuple representing the coordinates of box A in the format (left, top, right, bottom).
    box_b (tuple): Tuple representing the coordinates of box B in the format (left, top, right, bottom).

    Returns:
    tuple: Tuple representing the coordinates of the intersection box in the format (left, top, right, bottom).
    """
    a_left, a_top, a_right, a_bottom = box_a
    b_left, b_top, b_right, b_bottom = box_b
    return max(a_left, b_left), max(a_top, b_top), min(a_right, b_right), min(a_bottom, b_bottom)

def hex_to_hsv(hex_color: str) -> np.ndarray:
    """
    Convert a hexadecimal color code to HSV color space.

    Args:
    hex_color (str): Hexadecimal color code (e.g., '#RRGGBB').

    Returns:
    tuple: Tuple representing the HSV color space values (Hue, Saturation, Value).
    """
    # Convert hex to RGB
    hex_color = hex_color.lstrip('#')
    rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    bgr_color = tuple(reversed(rgb_color))

    hsv_color = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]

    return hsv_color

def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    Convert a PIL Image object to a cv2 image object.
    
    Args:
    pil_image (PIL.Image.Image): The PIL Image object to convert.
    
    Returns:
    np.ndarray: The converted cv2 image object.
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    Convert a cv2 image object to a PIL Image object.
    
    Args:
    cv2_image (np.ndarray): The cv2 image object to convert.
    
    Returns:
    PIL.Image.Image: The converted PIL Image object.
    """
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))

def ensure_image_format(input_image, format: str):
    """
    Converts an image to the specified format (PIL or cv2).

    Args:
        input_image: The image to be converted.
        format (str): The desired format ('pil' or 'cv2').

    Raises:
        KeyError: If the specified format is not supported.
    """
    if format == 'pil':
        if not isinstance(input_image, Image.Image):
            return cv2_to_pil(input_image)
        else:
            return input_image
    elif format == 'cv2':
        if isinstance(input_image, Image.Image):
            return pil_to_cv2(input_image)
        else:
            return input_image
    else:
        raise KeyError(f"Format not supported: {format}. Expected 'pil' or 'cv2' format")

def process_entities(entities: List[Dict]) -> Dict:
    """
    Process a list of entity dictionaries to consolidate entities of the same type and handle subwords marked with '##'.

    This function iterates through a list of dictionaries, each representing an entity with a type and word, consolidates the words of the same entity type into a single string per type, and handles subwords by removing the '##' and concatenating them directly to the previous word without adding extra spaces for subwords.

    Args:
    entities (List[Dict]): A list of dictionaries, where each dictionary contains an 'entity' key representing the entity type, and a 'word' key representing the word associated with that entity.

    Returns:
    Dict: A dictionary where each key is an entity type, and the value is a string consisting of all words of that entity type, concatenated together with spaces, and subwords concatenated directly to the preceding word without spaces.
    """
    processed_entities = {}
    for entity in entities:
        entity_type = entity['entity'].replace('B-', '').replace('I-', '')
        word = entity['word']
        # Check if the word is a subword and handle accordingly
        if word.startswith('##'):
            # Remove '##' and concatenate directly to the last word without adding a space
            processed_entities[entity_type] = processed_entities[entity_type] + word[2:]
        else:
            if entity_type not in processed_entities:
                processed_entities[entity_type] = word
            else:
                # Add a space before the word if it's not a subword
                processed_entities[entity_type] += ' ' + word
    return processed_entities

def extract_summary_pii_information(text: str) -> Dict:
    """
    Extracts Personally Identifiable Information (PII) such as phone number, insurance code, date of birth, and personal ID from a given text.

    This function searches the input text for patterns matching phone numbers, insurance codes, dates of birth, and personal IDs using regular expressions. It returns the first occurrence of each type of information found. If multiple insurance codes are found and the first one matches the phone number, the second insurance code is returned instead.

    Args:
    text (str): The text from which to extract PII information.

    Returns:
    tuple: A tuple containing the first phone number, insurance code, date of birth, and personal ID found in the text. Each element is a string, and if a particular type of information is not found, an empty string is returned for that element.
    """
    phone_numbers = re.findall(r'\b(?:0|\+84)\d{9,10}\b', text)
    phone_number = phone_numbers[0] if phone_numbers else ''
    
    insurance_codes = re.findall(r'\b\d{10}\b', text)
    insurance_code = ''
    if insurance_codes:
        insurance_code = insurance_codes[1] if insurance_codes[0] == phone_number and len(insurance_codes) > 1 else insurance_codes[0]
    
    date_of_births = re.findall(r'\b\d{2}/\d{2}/\d{4}\b', text)
    date_of_birth = date_of_births[0] if date_of_births else ''
    
    personal_ids = re.findall(r'\b\d{9}\b|\b\d{12}\b', text)
    personal_id = personal_ids[0] if personal_ids else ''
    
    return phone_number, insurance_code, date_of_birth, personal_id


def extract_number_of_months(input_string: str) -> int:
    """
    Extracts the total number of months from a given string by identifying and summing the years and months mentioned in it.

    This function searches for words in the input string that closely match 'year' and 'month' patterns (accounting for common misspellings) and then attempts to convert the word immediately preceding each identified pattern into a number, interpreting it as the quantity of years or months. The total number of months is calculated by converting years to months (multiplying by 12) and adding the number of months found.

    Args:
    input_string (str): The string from which to extract the number of months.

    Returns:
    int: The total number of months extracted from the string.
    """

    year_patterns = ["year", "yeer", "năm"]
    month_patterns = ["month", "monh", "tháng"]
    
    years = 0
    months = 0

    input_string = input_string.lower()
    words = input_string.split()    
    for i, word in enumerate(words):
        if any(ratio(word, year_pattern) > 0.5 for year_pattern in year_patterns):
            try:
                years = int(words[i-1])
            except ValueError:
                pass
        
        if any(ratio(word, month_pattern) > 0.5 for month_pattern in month_patterns):
            try:
                months = int(words[i-1])
            except ValueError:
                pass
    
    total_months = years * 12 + months
    return total_months
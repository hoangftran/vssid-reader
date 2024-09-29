import cv2
import pandas as pd
import numpy as np

from typing import Dict
from PIL import Image

from model import EasyOCRModel, TableDetectionModel, TableStructureRecognitionModel, NERModel
from utils import intersect, get_intersection, hex_to_hsv, process_entities, extract_summary_pii_information, ensure_image_format, load_config

class InsuranceContributionScreenshotProcessor:
    def __init__(self, detection_model: TableDetectionModel, structure_recognition_model: TableStructureRecognitionModel, easyocr_model: EasyOCRModel):
        self.detection_model = detection_model
        self.structure_recognition_model = structure_recognition_model
        self.easyocr_model = easyocr_model

        self.dark_blue_range = [(max(0, 57 - 30), max(0, 103 - 30), max(0, 158 - 30)), (min(255, 57 + 30), min(255, 103 + 30), min(255, 158 + 30))]
        self.light_grey_range = [(max(0, 244 - 10), max(0, 244 - 10), max(0, 242 - 10)), (min(255, 244 + 10), min(255, 244 + 10), min(255, 242 + 10))]
        
    
    def crop_detected_table_image(self, input_image: Image.Image, detection_results: Dict) -> Image.Image:
        """
        Crop the detected table image based on the highest scoring box from detection results.

        Args:
            input_image (Image): The input image containing the detected table.
            detection_results (Dict): The results of the table detection model.

        Returns:
            Image: The cropped detected table image.
        """
        highest_score_index = detection_results['scores'].argmax()
        box = detection_results['boxes'][highest_score_index].detach().numpy()
        left, top, right, bottom = box
        image_width, image_height = input_image.size
        padding = min(image_width, image_height) // 15
        left = max(left - padding, 0)
        top = max(top - padding*3.5, 0)
        right = min(right + padding, image_width)
        bottom = min(bottom + padding, image_height)
        detected_table_image = input_image.crop((left, top, right, bottom))

        return detected_table_image

    def table_reconstruction(self, detected_table_image: Image.Image, structure_recognition_results: Dict) -> Dict:
        """
        Perform table reconstruction based on the detected table image and structure recognition results.

        Args:
            detected_table_image (Image): The cropped detected table image.
            structure_recognition_results (Dict): The results of the structure recognition model.

        Returns:
            Dict: A dictionary containing the reconstructed table data.
        """
        rows = sorted([box.detach().numpy() for box, label in zip(structure_recognition_results['boxes'], structure_recognition_results['labels']) if label == 2], key=lambda x: x[1])
        columns = sorted([box.detach().numpy() for box, label in zip(structure_recognition_results['boxes'], structure_recognition_results['labels']) if label == 1], key=lambda x: x[0])

        num_rows = len(rows)
        num_columns = len(columns)
        cell_texts = [['' for _ in range(num_columns)] for _ in range(num_rows - 1)]
        is_header_row = False
        
        for i, row in enumerate(rows[1:], start=1):
            for j, column in enumerate(columns):
                if intersect(row, column):
                    cell_box = get_intersection(row, column)
                    cell_image = detected_table_image.crop(cell_box)
                    easyocr_result = self.easyocr_model.fit(cell_image.convert("L"), paragraph=True)
                    cell_text = ' '.join([para[1] for para in easyocr_result])
                    cell_text = cell_text.replace('\n', ' ').strip()
                    cell_texts[i-1][j] = cell_text

                    cell_pixel_color = cell_image.getpixel((1, 1))
                    if (self.dark_blue_range[0][0] <= cell_pixel_color[0] <= self.dark_blue_range[1][0] and self.dark_blue_range[0][1] <= cell_pixel_color[1] <= self.dark_blue_range[1][1] and self.dark_blue_range[0][2] <= cell_pixel_color[2] <= self.dark_blue_range[1][2]) or \
                    (self.light_grey_range[0][0] <= cell_pixel_color[0] <= self.light_grey_range[1][0] and self.light_grey_range[0][1] <= cell_pixel_color[1] <= self.light_grey_range[1][1] and self.light_grey_range[0][2] <= cell_pixel_color[2] <= self.light_grey_range[1][2]):
                        is_header_row = True
        
        if is_header_row:
            table_reconstructed = pd.DataFrame(cell_texts[1:], columns=cell_texts[0])
        else:
            table_reconstructed = pd.DataFrame(cell_texts)
        
        if table_reconstructed.columns[-1] == '' or all(value == '' for value in table_reconstructed.iloc[:, -1]):
            table_reconstructed = table_reconstructed.iloc[:, :-1]
        
        if len(table_reconstructed.columns) == 4:
            table_reconstructed.columns = ['FROM', 'TO', 'EMPLOYER', 'JOB_TITLE']
        elif len(table_reconstructed.columns) == 5:
            table_reconstructed.columns = ['FROM', 'TO', 'EMPLOYER', 'JOB_TITLE', '']
        
        return table_reconstructed.to_json(orient="records")

    def process(self, input_image) -> Dict:
        input_image = ensure_image_format(input_image, format='pil')
        detection_results = self.detection_model.fit(input_image)
        detected_table_image = self.crop_detected_table_image(input_image, detection_results)
        structure_recognition_results = self.structure_recognition_model.fit(detected_table_image)
        table_reconstructed = self.table_reconstruction(detected_table_image, structure_recognition_results)

        return table_reconstructed

class SummaryScreenshotProcessor:
    def __init__(self, ner_model: NERModel, easyocr_model: EasyOCRModel):
        self.ner_model = ner_model
        self.easyocr_model = easyocr_model
        self.hex_color = "#e3edf3"
        self.hsv_color = hex_to_hsv(self.hex_color)

        self.sensitivity = 15
        self.light_blue = np.array([self.hsv_color[0] - self.sensitivity, max(self.hsv_color[1] - self.sensitivity, 0), max(self.hsv_color[2] - self.sensitivity, 0)])
        self.dark_blue = np.array([self.hsv_color[0] + self.sensitivity, min(self.hsv_color[1] + self.sensitivity, 255), min(self.hsv_color[2] + self.sensitivity, 255)])
    
    def detect(self, input_image):
        hsv = cv2.cvtColor(input_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.light_blue, self.dark_blue)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        return x, y, w, h
    
    def extract_pii(self, easyocr_result):
        input_string = ' '.join([para[1] for para in easyocr_result])
        ner_result = self.ner_model.fit(input_string)
        entities = process_entities(ner_result)
        name = entities.get('PERSON', '')
        address = entities.get('LOCATION', '')
        phone_number, insurance_code, date_of_birth, personal_id = extract_summary_pii_information(input_string)

        pii_extracted = {
            'name': [name],
            'phone_number': [phone_number],
            'insurance_code': [insurance_code],
            'date_of_birth': [date_of_birth],
            'personal_id': [personal_id],
            'address': [address],
        }

        return pii_extracted
    
    def process(self, input_image):
        input_image = ensure_image_format(input_image, format='cv2')
        x, y, w, h = self.detect(input_image)
        cropped_image = cv2.cvtColor(input_image[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

        easyocr_result = self.easyocr_model.fit(cropped_image, paragraph=True)
        pii_extracted = self.extract_pii(easyocr_result)

        return pii_extracted

if __name__ == "__main__":
    config = load_config('../config/models.yaml')

    easyocr_model = EasyOCRModel(config)
    table_detection_model = TableDetectionModel(config)
    table_structure_model = TableStructureRecognitionModel(config)
    ner_model = NERModel(config)

    insurance_contribution_screenshot_processor = InsuranceContributionScreenshotProcessor(
        detection_model = table_detection_model, 
        structure_recognition_model = table_structure_model,
        easyocr_model = easyocr_model
    )

    summary_screenshot_processor = SummaryScreenshotProcessor(
        ner_model = ner_model,
        easyocr_model = easyocr_model
    )

    insurance_contribution_image_path = '../data/screenshot/detail/1.png'
    insurance_contribution_image = cv2.imread(insurance_contribution_image_path)
    insurance_contribution_result = insurance_contribution_screenshot_processor.process(insurance_contribution_image)

    summary_image_path = '../data/screenshot/summary/1.png'
    summary_image = cv2.imread(summary_image_path)
    summary_result = summary_screenshot_processor.process(summary_image)

    print(insurance_contribution_result)
    print(summary_result)
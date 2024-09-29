import torch
import easyocr
from transformers import pipeline
from transformers import AutoImageProcessor, AutoModelForTokenClassification, AutoTokenizer, TableTransformerForObjectDetection
from utils import load_config, ensure_image_format

from PIL import Image

class BaseModel:
    def __init__(self, model_path=None):
        self.model_path = model_path
        # TODO: Set up GPU when feed to models
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"


class EasyOCRModel(BaseModel):
    def __init__(self, config):
        super().__init__(None)
        self.languages = config.get('languages', ['vi'])
        self.model = easyocr.Reader(lang_list=self.languages, gpu=False)

    def fit(self, input_image, paragraph=None):
        input_image = ensure_image_format(input_image, format='cv2')
        return self.model.readtext(input_image, paragraph=paragraph)

class TableDetectionModel(BaseModel):
    def __init__(self, config):
        super().__init__(config['table-transformer']['detection'])
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = TableTransformerForObjectDetection.from_pretrained(self.model_path)

    def fit(self, input_image):
        try:
            input_image = ensure_image_format(input_image, format='pil')
            detection_inputs = self.image_processor(images=input_image, return_tensors='pt')
            detection_outputs = self.model(**detection_inputs)
            target_sizes = torch.tensor([input_image.size[::-1]])
            detection_results = self.image_processor.post_process_object_detection(detection_outputs, threshold=0.8, target_sizes=target_sizes)[0]

            return detection_results
        except Exception as e:
            print(f"Error during table detection: {e}")
            return None

class TableStructureRecognitionModel(BaseModel):
    def __init__(self, config):
        super().__init__(config['table-transformer']['structure-recognition'])
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_path)
        self.model = TableTransformerForObjectDetection.from_pretrained(self.model_path)

        if self.model_path == 'microsoft/table-transformer-structure-recognition-v1.1-all':
            self.image_processor.size['shortest_edge'] = 800

    def fit(self, input_image):
        try:
            input_image = ensure_image_format(input_image, format='pil')
            structure_recognition_inputs = self.image_processor(images=input_image, return_tensors='pt')
            structure_recognition_outputs = self.model(**structure_recognition_inputs)
            target_sizes = torch.tensor([input_image.size[::-1]])
            structure_recognition_results = self.image_processor.post_process_object_detection(structure_recognition_outputs, threshold=0.6, target_sizes=target_sizes)[0]

            return structure_recognition_results

        except Exception as e:
            print(f"Error during table structure recognition: {e}")
            return None

class NERModel(BaseModel):
    def __init__(self, config):
        super().__init__(config['ner'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
        self.nlp = pipeline("ner", model=self.model, tokenizer=self.tokenizer)

    def fit(self, input_string):
        try:
            return self.nlp(input_string)
        except Exception as e:
            print(f"Error during NER task: {e}")
            return None

if __name__ == "__main__":
    config = load_config('../config/models.yaml')

    easy_ocr_model = EasyOCRModel(config)
    table_detection_model = TableDetectionModel(config)
    table_structure_model = TableStructureRecognitionModel(config)
    ner_model = NERModel(config)
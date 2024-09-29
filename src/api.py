import logging
import os
from typing import List, Optional

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel, validator

from model import EasyOCRModel, TableDetectionModel, TableStructureRecognitionModel, NERModel
from processor import InsuranceContributionScreenshotProcessor, SummaryScreenshotProcessor
from utils import load_config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRModelManager:
    """
    Manages loading and accessing OCR models.
    """
    def __init__(self, config):
        self.config = config
        self._easyocr_model = None
        self._table_detection_model = None
        self._table_structure_model = None
        self._ner_model = None

    def get_models(self):
        """
        Lazily loads necessary OCR models if not already loaded.

        Returns:
            tuple: A tuple containing the loaded models (easyocr, table detection, table structure, ner).
        """
        if not any([self._easyocr_model, self._table_detection_model, self._table_structure_model, self._ner_model]):
            self._easyocr_model = EasyOCRModel(self.config)
            self._table_detection_model = TableDetectionModel(self.config)
            self._table_structure_model = TableStructureRecognitionModel(self.config)
            self._ner_model = NERModel(self.config)
            logger.info("Models initialized successfully.")
        return self._easyocr_model, self._table_detection_model, self._table_structure_model, self._ner_model

def get_config(config_path: str = os.path.join('..', 'config', 'models.yaml')) -> dict:
    """
    Loads the configuration from the specified file path.

    Args:
        config_path (str, optional): Path to the configuration file. Defaults to os.path.join('..', 'config', 'models.yaml').

    Returns:
        dict: The loaded configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is not found.
        Exception: If there's an error loading the configuration.
    """
    try:
        config = load_config(config_path)
        logger.info("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        logger.error("Configuration file not found.")
        raise HTTPException(status_code=500, detail="Configuration file not found.")
    except Exception as e:
        logger.error(f"Error loading configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading configuration: {str(e)}")

app = FastAPI(
    title='VSSID OCR',
    version='0.0.1',
    contact={
        'name': 'Hoang Tran Van',
        'email': 'hoang.tranv2@homecredit.vn',
    }
)

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred. Please try again later."},
    )

class OCRRequest(BaseModel):
    image: UploadFile   

    @validator('image')
    def validate_image(cls, value):
        """
        Validates the uploaded image for format and size.

        Args:
            value (UploadFile): The uploaded image file.

        Returns:
            UploadFile: The validated image file.

        Raises:
            HTTPException: If the image format is not supported or the file size exceeds the limit.
        """
        allowed_formats = ['image/jpeg', 'image/png']
        max_file_size = 2 * 1024 * 1024  # 2MB

        if value.content_type not in allowed_formats:
            logger.warning("Attempt to upload unsupported image format.")
            raise HTTPException(status_code=400, detail="Only JPEG and PNG images are allowed.")
        
        with value.file as file:
            file_content = file.read()
            file_size = len(file_content)
            file.seek(0)  # Reset file pointer after reading

        if file_size > max_file_size:
            logger.warning("File size exceeds limit.")
            raise HTTPException(status_code=400, detail="File size exceeds 2MB limit.")

        return value

@app.post('/ocr/insurance_contribution')
async def ocr_insurance_contribution(request: OCRRequest):
    """
    Endpoint for OCR processing of insurance contribution screenshots.

    Args:
        request (OCRRequest): The request containing the image to be processed.

    Returns:
        dict: The OCR processing result.

    Raises:
        HTTPException: If there's an error during processing.
    """
    try:
        config = get_config(os.path.join('..', 'config', 'models.yaml'))
        model_manager = OCRModelManager(config)
        insurance_processor = InsuranceContributionScreenshotProcessor(
            detection_model=model_manager.get_models()[1],
            structure_recognition_model=model_manager.get_models()[2],
            easyocr_model=model_manager.get_models()[0]
        )
        image_data = await request.image.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Invalid image format received.")
            raise HTTPException(status_code=400, detail="Invalid image format.")
        result = insurance_processor.process(image)
        logger.info("Insurance contribution OCR processed successfully.")
    except Exception as e:
        logger.error(f"Error processing insurance contribution screenshot: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")
    return result

@app.post('/ocr/summary')
async def ocr_summary(request: OCRRequest):
    """
    Endpoint for OCR processing of summary screenshots.

    Args:
        request (OCRRequest): The request containing the image to be processed.

    Returns:
        dict: The OCR processing result.

    Raises:
        HTTPException: If there's an error during processing.
    """
    try:
        config = get_config(os.path.join('..', 'config', 'models.yaml'))
        model_manager = OCRModelManager(config)
        summary_processor = SummaryScreenshotProcessor(
            ner_model=model_manager.get_models()[3],
            easyocr_model=model_manager.get_models()[0]
        )
        image_data = await request.image.read()
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Invalid image format received.")
            raise HTTPException(status_code=400, detail="Invalid image format.")
        result = summary_processor.process(image)
        logger.info("Summary OCR processed successfully.")
    except Exception as e:
        logger.error(f"Error processing summary screenshot: {str(e)}")
        raise HTTPException(status_code=500, detail="An error occurred during processing.")
    return result

if __name__ == '__main__':  
    # cmd: uvicorn api:app --host 0.0.0.0 --port 8000
    uvicorn.run('api:app', host='0.0.0.0', port=8000, reload=True)
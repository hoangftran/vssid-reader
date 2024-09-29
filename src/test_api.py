import pytest
from httpx import AsyncClient
from fastapi import UploadFile, HTTPException
from api import app, get_config, get_models, get_processors

@pytest.fixture
def test_client():
    return AsyncClient(app=app, base_url="http://test")

@pytest.mark.asyncio
async def test_get_config_success(mocker):
    # Test successful retrieval of configuration
    mocker.patch('api.load_config', return_value={'dummy': 'config'})
    assert await get_config() == {'dummy': 'config'}

@pytest.mark.asyncio
async def test_get_config_failure(mocker):
    # Test failure to retrieve configuration
    mocker.patch('api.load_config', side_effect=FileNotFoundError)
    with pytest.raises(HTTPException) as exc_info:
        await get_config()
    assert exc_info.value.status_code == 500
    assert "Configuration file not found." in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_get_models_success(mocker):
    # Test successful retrieval of models
    mocker.patch('api.get_config', return_value={'dummy': 'config'})
    mocker.patch('api.EasyOCRModel')
    mocker.patch('api.TableDetectionModel')
    mocker.patch('api.TableStructureRecognitionModel')
    mocker.patch('api.NERModel')
    models = await get_models()
    assert models is not None

@pytest.mark.asyncio
async def test_get_processors_success(mocker):
    # Test successful retrieval of processors
    mocker.patch('api.get_models', return_value=(None, None, None, None))
    processors = await get_processors()
    assert processors is not None

@pytest.mark.asyncio
async def test_ocr_insurance_contribution_success(test_client, mocker):
    # Test successful OCR of insurance contribution
    mocker.patch('api.get_processors', return_value=(mocker.MagicMock(), mocker.MagicMock()))
    mocker.patch('cv2.imdecode', return_value='decoded_image')
    response = await test_client.post("/ocr/insurance_contribution", files={"image": ("filename", b"fakeimage", "image/jpeg")})
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_ocr_insurance_contribution_invalid_format(test_client):
    # Test OCR of insurance contribution with invalid image format
    response = await test_client.post("/ocr/insurance_contribution", files={"image": ("filename", b"fakeimage", "image/gif")})
    assert response.status_code == 400
    assert "Only JPEG and PNG images are allowed." in response.json()['detail']

@pytest.mark.asyncio
async def test_ocr_summary_success(test_client, mocker):
    # Test successful OCR of summary
    mocker.patch('api.get_processors', return_value=(mocker.MagicMock(), mocker.MagicMock()))
    mocker.patch('cv2.imdecode', return_value='decoded_image')
    response = await test_client.post("/ocr/summary", files={"image": ("filename", b"fakeimage", "image/jpeg")})
    assert response.status_code == 200

@pytest.mark.asyncio
async def test_ocr_summary_invalid_format(test_client):
    # Test OCR of summary with invalid image format
    response = await test_client.post("/ocr/summary", files={"image": ("filename", b"fakeimage", "image/gif")})
    assert response.status_code == 400
    assert "Only JPEG and PNG images are allowed." in response.json()['detail']
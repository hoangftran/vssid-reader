# OCR 
## About
This project provides a pipeline to extract information (PII, insurance contribution and ) from VSSID app's screenshot
## Dependencies
```bash
conda create -n vssid-ocr python=3.9 -y
conda activate vssid-ocr
pip install -r requirements.txt
```
## Installing Tessaract
You can either [Install Tesseract via pre-built binary package](https://tesseract-ocr.github.io/tessdoc/Installation.html) or [build it from source](https://tesseract-ocr.github.io/tessdoc/Compiling.html).

A C++ compiler with good C++17 support is required for building Tesseract from source.
## References

[Table Transformer](https://github.com/microsoft/table-transformer)

[Tesseract](https://github.com/tesseract-ocr/tesseract)
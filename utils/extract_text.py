import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using pdfplumber, fall back to OCR if needed."""
    try:
        # First attempt with pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            
            if text.strip():
                logger.info(f"Successfully extracted text from {pdf_path} using pdfplumber")
                return text

        # Fallback to OCR if no text extracted
        logger.warning(f"No text extracted with pdfplumber for {pdf_path}, attempting OCR")
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image) + "\n"
        
        logger.info(f"Extracted text from {pdf_path} using OCR")
        return text

    except Exception as e:
        logger.error(f"Failed to extract text from {pdf_path}: {e}")
        raise
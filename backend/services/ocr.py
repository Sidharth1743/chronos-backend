from agentic_doc.parse import parse
from dotenv import load_dotenv
import os

class OCRProcessor:
    def __init__(self):
        """Initialize the OCR processor."""
        load_dotenv()
        self.api_key = os.getenv("VISION_AGENT_API_KEY")
        if not self.api_key:
            raise ValueError("VISION_AGENT_API_KEY environment variable is not set")

    def extract_text(self, file_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text in markdown format
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        result = parse(file_path)
        if result and len(result) > 0:
            return result[0].markdown
        return ""

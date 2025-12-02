# from tools.pdf_converter import convert_markdown_to_pdf
from tools.text_generator import video_to_text
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent  

file_path = BASE_DIR / "src" /"transcriber"/ "Sample.mp4"

text_content = video_to_text(file_path)

print(text_content)

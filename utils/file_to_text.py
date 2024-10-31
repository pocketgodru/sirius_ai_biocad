import os
import fitz  # Correct import for pymupdf
from docx import Document
from PIL import Image
import mimetypes
import io
import base64

# Функция для кодирования изображения в base64 без сохранения на диск
def encode_image_bytes(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

def process_file(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    
    if mime_type == 'application/pdf':
        return process_pdf(file_path)
    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        return process_docx(file_path)
    elif mime_type == 'text/plain':
        return process_txt(file_path)
    else:
        print(f"Unsupported file type: {mime_type}")
        return None, []

# Загрузка текста и изображений из различных типов документов
def load_text(file_name: str):
    ext = os.path.splitext(file_name)[1].lower()
    
    try:
        if ext == '.pdf':
            return process_pdf(file_name)
        elif ext == '.txt':
            return process_txt(file_name)
        elif ext in ['.doc', '.docx']:
            return process_docx(file_name)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None, []

# Обработка PDF файлов
def process_pdf(file_path):
    text = ""
    images = []
    pdf_document = fitz.open(file_path)

    # Извлечение текста
    for page_num in range(len(pdf_document)):
        text += pdf_document[page_num].get_text("text")

    # Извлечение изображений
    for page_num in range(len(pdf_document)):
        for _, img in enumerate(pdf_document.get_page_images(page_num, full=True)):
            xref = img[0]  # Получаем ссылку на изображение
            base_image = pdf_document.extract_image(xref)  # Извлекаем изображение
            image_bytes = base_image["image"]  # Изображение в виде байтов
            image_ext = base_image["ext"]  # Расширение изображения (png, jpeg и т.д.)

            # Конвертация изображения в base64 без сохранения на диск
            base64_image = encode_image_bytes(image_bytes)
            image_data = f"data:image/{image_ext};base64,{base64_image}"

            images.append({"type": "image_url", "image_url": image_data})

    return text, images

# Обработка DOCX файлов
def process_docx(file_path):
    doc = Document(file_path)
    text = ""
    images = []

    # Извлечение текста
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"

    # Извлечение изображений
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            img_data = rel.target_part.blob
            img = Image.open(io.BytesIO(img_data))
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            image_base64 = encode_image_bytes(buffered.getvalue())
            images.append({"type": "image_url", "image_url": f"data:image/jpeg;base64,{image_base64}"})

    return text, images

# Обработка TXT файлов
def process_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()
    return text, []

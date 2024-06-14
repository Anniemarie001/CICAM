import fitz
from PIL import Image

def convert_to_image(file_path):
    if file_path.endswith('.pdf'):
        # Handle PDF files
        pdf_file = fitz.open(file_path)
        images = []
        for page_num in range(len(pdf_file)):
            page = pdf_file[page_num]
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            images.append(img)
        return images
    
    elif file_path.lower().endswith(('.png', '.jpeg', '.jpg')):
        # Handle PNG, JPG, and JPEG files
        img = Image.open(file_path).convert('RGB')
        return [img]
    else:
        raise ValueError("Unsupported file format. Supported formats are: PDF, PNG, JPG, JPEG.")
    #else:
        # Handle PNG and JPG files
        #img = Image.open(file_path)
        #return [img]
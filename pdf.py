import PyPDF2
import pdfplumber
import streamlit as st
from PIL import Image
import io

def parse_pdf(file):
    pdf_text = ""
    images = []

    # Extract text using PyPDF2
    pdf_reader = PyPDF2.PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()

    # Extract images using pdfplumber
    file.seek(0)  # Reset file pointer to the beginning
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            for img in page.images:
                # Extract the raw image data (PDF coordinates and image stream)
                img_x0 = img['x0']
                img_x1 = img['x1']
                img_top = img['top']
                img_bottom = img['bottom']
                # Get the image stream from the page
                image = page.within_bbox((img_x0, img_top, img_x1, img_bottom)).to_image()
                
                # Save the image to an in-memory bytes buffer
                img_buffer = io.BytesIO()
                image.save(img_buffer, format="PNG")  # Save as PNG
                img_buffer.seek(0)
                
                # Open as a PIL image
                pil_img = Image.open(img_buffer)
                images.append(pil_img)

    return pdf_text, images
    
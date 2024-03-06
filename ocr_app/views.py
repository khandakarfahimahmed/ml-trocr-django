from django.shortcuts import render
from django.http import HttpResponse

# Create your views here.
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import JsonResponse
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

import pdf2image
import io
import fitz

from pdf2image  import convert_from_bytes
import numpy as np

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

class OCRImageView(APIView):
    def post(self, request):
        print("Received request")
        if 'image' in request.FILES:
            print("Image file received")
            image_file = request.FILES['image']
            image = Image.open(image_file).convert("RGB")
            cropped_image = image.crop((0, 120, image.size[0], 320)) #cropping the image

            ocr_text = self.process_image(cropped_image)
            return JsonResponse({'ocr_text': ocr_text})
            
        elif 'pdf' in request.FILES:
            print("PDF file received")
            # Process PDF file
            pdf_file = request.FILES['pdf']
            # ocr_text = self.process_pdf(pdf_file)
            ocr_text = self.pdftoimage(pdf_file)
            return JsonResponse({'ocr_text': ocr_text})
        else:
            return JsonResponse({'error': 'No image or PDF provided'}, status=400)

    def process_image(self, image):
        print("Processing image")
        input_data = {"images": image}
        pixel_values = processor(**input_data, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        ocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return ocr_text
    
    def process_pdf(self, pdf_file):
        print("Processing PDF")
        ocr_text = ""
        # Read the PDF file as bytes
        pdf_bytes = pdf_file.read()
        # Convert the PDF pages to images
        images = pdf2image.convert_from_bytes(pdf_bytes, fmt='jpg')
        # Process each page image
        for page_img in images:
            # Crop the image if needed
            cropped_image = page_img.crop((0, 120, page_img.size[0], 320))
            # Process the cropped image
            page_ocr_text = self.process_image(cropped_image)
            ocr_text += page_ocr_text + "\n"
        return ocr_text
    def pdftoimage(self, pdf_file):
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        # Convert the first page to an image
        page = pdf_document[0]
        pix = page.get_pixmap()
        # Check if the image has 4 channels (RGBA), if not, convert to 3 channels (RGB)
        if pix.n == 4:
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 4))
        else:
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape((pix.height, pix.width, 3))
            page_ocr_text = self.process_image(img)
            
        return page_ocr_text


        







def index(request):
    return HttpResponse("Hey there, all is well!")

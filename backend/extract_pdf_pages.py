import os
import pymupdf  # PyMuPDF
from PIL import Image


def pdf_to_images(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(input_folder, filename)
            pdf_document = pymupdf.open(pdf_path)

            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                # Save the image
                image_filename = f"{os.path.splitext(filename)[0]}_page_{page_num + 1}.jpg"
                img.save(os.path.join(output_folder, image_filename), "JPEG")


if __name__ == "__main__":
    input_folder = "/Users/jbuhler/Development/images/LifeMagazine"  # Replace with your input folder path
    output_folder = "/Users/jbuhler/Development/images/LifeMagazine/images"  # Replace with your output folder path
    pdf_to_images(input_folder, output_folder)

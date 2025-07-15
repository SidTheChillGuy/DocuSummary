allset = True
print("Loading module 'cnvpdf2img'. Starting all Imports.")

import os
try:
    from pdf2image import convert_from_path
except:
    print("The module 'pdf2image' or its components: 'poppler-utils' is not installed or added to the path.")
    print("Please install the missing dependencies and retry.")
    allset = False

if allset == False:
    raise ImportError("Fix the existing issues.")

if len(os.listdir("tmp"))>0:
    print("Cleaning tempdir...")
    for i in os.listdir("tmp"):
        os.remove(f"tmp/{i}")

print("Converting all PDF files into Images.")

def convert_pdfs_to_images(pdf_folder):
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    for pdf_file in pdf_files:
        print(f"Processing file {pdf_file}")
        pdf_path = os.path.join(pdf_folder, pdf_file)
        images = convert_from_path(pdf_path)
        for i in range(len(images)):
            images[i].save(f"tmp/id_][_{pdf_file}_][_value_][_{i}_][_.png")

all_images = convert_pdfs_to_images("data")

print("Image files for data extraction are created.")
print("Module 'cnvpdf2img' completed... Please wait while next function loads.")
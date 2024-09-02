import spacy
from pdfminer.high_level import extract_text

nlp = spacy.load("en_core_web_sm")

def parse_resume(file_path):
    resume_text = extract_text(file_path)
    doc = nlp(resume_text)

    extracted_data = {}
    for ent in doc.ents:
        extracted_data[ent.label_] = ent.text

    return extracted_data

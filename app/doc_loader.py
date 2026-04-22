from docx import Document
import os

def load_docx(folder_path: str):
    docs = []

    for file in os.listdir(folder_path):
        if file.endswith(".docx"):
            doc = Document(os.path.join(folder_path, file))
            text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            docs.append(text)

    return docs
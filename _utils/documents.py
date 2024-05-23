import io
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter





# async def get_file_extension(filename):
#     if '.' in filename:
#         return filename.rsplit('.', 1)[-1]
#     else:
#         return ""

async def get_pdf_text(file):
    reader = PdfReader(file)
    extracted_text = ""
    for page in reader.pages:
        extracted_text += page.extract_text()
    return extracted_text


async def get_file_extension(filename):
    if '.' in filename:
        return filename.rsplit('.', 1)[-1]
    else:
        return ""

async def get_chunks(files):
    splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=20,separators='')
    chunks = []
    files_not_supported = []

    for file in files:
        if file.filename.endswith('.txt'):
            file_content = await file.read()
            text = file_content.decode('utf-8')
            chunks.extend(splitter.create_documents(texts=[text],metadatas=[{'file_name':file.filename}]))

        elif file.filename.endswith('.doc'):
            file_content = await file.read()
            print(file_content)
            text = file_content.decode('utf-8')
            chunks.extend(splitter.create_documents(texts=[text]))

        elif file.filename.endswith('.docx'):
            file_content = await file.read()
            doc = DocxDocument(io.BytesIO(file_content))
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            chunks.extend(splitter.create_documents(texts=[text]))

        elif file.filename.endswith('.pdf'):
            text = await get_pdf_text(file.file)
            chunks.extend(splitter.create_documents(texts=[text]))

        else:
            files_not_supported.append(get_file_extension(file.filename))


    return chunks, files_not_supported
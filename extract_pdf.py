import sys

def extract_pdf():
    try:
        import fitz # PyMuPDF
        doc = fitz.open('puerto-rico-rules-en.pdf')
        text = ""
        for page in doc:
            text += page.get_text()
        with open('rules.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Successfully extracted text with PyMuPDF.")
        return
    except ImportError:
        pass

    try:
        from pypdf import PdfReader
        reader = PdfReader('puerto-rico-rules-en.pdf')
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        with open('rules.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Successfully extracted text with pypdf.")
        return
    except ImportError:
        pass

    try:
        import PyPDF2
        reader = PyPDF2.PdfReader('puerto-rico-rules-en.pdf')
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        with open('rules.txt', 'w', encoding='utf-8') as f:
            f.write(text)
        print("Successfully extracted text with PyPDF2.")
        return
    except ImportError:
        print("Error: No PDF extraction libraries found. Please run: pip install pypdf")
        sys.exit(1)

if __name__ == "__main__":
    extract_pdf()

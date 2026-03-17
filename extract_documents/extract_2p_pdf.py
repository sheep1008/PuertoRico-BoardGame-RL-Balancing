import PyPDF2
reader = PyPDF2.PdfReader('puerto-rico-2players-rule.pdf')
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"
print(text)

import pandas as pd 
import PyPDF2
from nltk import sent_tokenize

path = r"C:\Users\de larrard alexandre\Documents\repos_github\PEA\data\extracted_data\MAUP.PA\financials\results-for-the-first-half-of-2021.pdf"

file = open(path, 'rb')
fileReader = PyPDF2.PdfFileReader(file)

# print the number of pages in pdf file
pages = {}
for i in range(fileReader.numPages):
    pages[i] = fileReader.getPage(i).extractText().split("\n")
    pages[i] = " ".join([x for x in pages[i] if x != " "])
    pages[i] = sent_tokenize(pages[i])
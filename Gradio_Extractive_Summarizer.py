import gradio as gr
import numpy as np


import pytesseract
from PIL import Image
import re

##pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
import tkinter as tk 
from tkinter import filedialog 

import gensim
from gensim.summarization.summarizer import summarize
import re
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict
import string
from heapq import nlargest
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import string


#load stopwords
stop = nltk.corpus.stopwords.words('english')


#remove stopwords as for text summarization purposes,
#these words add no value to word ranking.
def remove_stopwords(sentence):
    filtered_sentence = " ".join([i for i in sentence if i not in stop])
    return filtered_sentence



def sepia(img , how_much_sentence_summary_needed , textual_input):
                        
  
	"""
	root = tk.Tk() 
	root.withdraw() 
	file_path = filedialog.askopenfilename() 

	##print(file_path)

	img = Image.open(file_path)

	print(img)
	"""

	text = pytesseract.image_to_string(img)
	#print(text)   

	"""
	with open('new_file.txt','w') as f :
		print ('Image text: \n ', text, file = f)

	#### python .\OCR.py > img2_summary.txt ...this command can be used directly to generate o/p into a txt file


	string = open('new_file.txt').read()
	new_str = re.sub('[^a-zA-Z0-9/n/.]', ' ', string)
	#new_str = [nstr.replace('[^a-zA-Z0-9/n/.]','\n') for nstr in new_str]
	new_str = re.sub('[ \t\n]+', ' ', new_str)

	with open('b.txt', 'w') as f:
		##print(new_str , "")
		open('b.txt', 'w').write(new_str)

	"""
	"""
	with open('file.txt', 'r') as f:
	    lines = f.readlines()

	# remove spaces
	lines = [line.replace('  ', '') for line in lines]

	# finally, write lines in the file
	with open('file.txt', 'w') as f:
	    f.writelines(lines)


	"""


	#Split paragraph into sentences. We want to know how similar each sentence is with each other.
	sentences = sent_tokenize(text)

	
	 
	#Pre-process your text. Remove punctuation, special characterts, numbers, etc. As the only thing we care
	#about are the actual words in the text.
	clean_sentences = [s.translate(string.punctuation) for s in sentences]
	clean_sentences  = [s.translate(string.digits) for s in clean_sentences]
	#lowercase
	clean_sentences = [s.lower() for s in clean_sentences]
	 

	clean_sentences = [remove_stopwords(s.split()) for s in clean_sentences]

	
	word_embeddings = {}
	file_ = open('C:/Users/LENOVO/Desktop/Python_revision/NLP_prac/glove.6B.300d.txt' , encoding = 'utf-8')
	for line in file_:
	    values = line.split()
	    word = values[0]
	    coefs = np.asarray(values[1:], dtype='float')
	    word_embeddings[word] = coefs
	file_.close()


	sentence_vectors = []
	for i in clean_sentences:
	    if len(i) != 0:
	        vector = sum([word_embeddings.get(w, np.zeros((300,))) for w in i.split()])/(len(i.split())+0.001)
	    else:
	        vector = np.zeros((300,))
	    sentence_vectors.append(vector)


	#Compute sentence similaritiy, initiate with zeros
	similarity_matrix = np.zeros([len(sentences), len(sentences)])
	for i in range(len(sentences)):
	    for j in range(len(sentences)):
	        if i != j:
	            similarity_matrix[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,300), sentence_vectors[j].reshape(1,300))[0,0]



	sim_graph = nx.from_numpy_matrix(similarity_matrix)
	scores = nx.pagerank(sim_graph)


	for i in range(len(sentences)):
		sentences[i] = re.sub('[^a-zA-Z0-9/n/.]', ' ', sentences[i])


	#Sentence Ranking
	ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)
	#print(ranked_sentences)
	length = (how_much_sentence_summary_needed)
	##print("How many sentence summary do you wish to have: ")
	##length = input()
	#Choose desired number of sentences
	print("**Summary is:** ")
	#print(ranked_sentences)
	new_summary = []
	for i in range(int(length)):
	    new_summary.append(ranked_sentences[i][1])

	return (new_summary)
	#print(new_summary)

iface = gr.Interface(sepia, [gr.inputs.Image(shape=(1240, 1754)) , "text", gr.inputs.Textbox(lines=2, placeholder="Or else type the text here...")] , "text")
iface.launch()


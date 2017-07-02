import nltk

#aim= tagging part of speech to word
from nltk.corpus import state_union
##contains speeches of various presidents

from nltk.tokenize import PunktSentenceTokenizer
## unsupervised machine learning based tokenizer , comes pre trained but can be trained again

train_text = state_union.raw("2005-GWBush.txt")
sample = state_union.raw("2006-GWBush.txt")


cust_sent_tokenizer =  PunktSentenceTokenizer(train_text)

tokenized = cust_sent_tokenizer.tokenize(sample)

def process_Content():
	try:
		for i in tokenized :
			words = nltk.word_tokenize(i)
			tagged= nltk.pos_tag(words)

#regular expressions are used 
#chunking lets you know who is the matter of subject in all the nouns present in sentences 
#			print(tagged)
			chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?} """
			chunkParser = nltk.RegexpParser(chunkGram)
			chunked = chunkParser.parse(tagged)
			print (chunked)
			chunked.draw()
	except Exception as e:
		print (str(e))


process_Content()

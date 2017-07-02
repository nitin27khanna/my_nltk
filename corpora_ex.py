from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw('bible-kjv.txt')

for sen in (sent_tokenize(sample))[5:10]:
	print(sen)


#we can find the nltk_data by print(nltk.__file__)

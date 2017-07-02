from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


#to reduce the processing , we make the words smaller 
#by removing the extensions and only keeping the root
#its called Stemming and done by ps() i.e PorterStemmer

ps = PorterStemmer()

example_words = "It is impotant for me to go to a good product based company to improvish my career to the right track" 

for w in word_tokenize(example_words) : 
	print(ps.stem(w))

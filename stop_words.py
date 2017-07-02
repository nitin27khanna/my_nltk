from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

## stop words are removed because they dont change the meaning of the sentence 
## with or without precense 
example = " my name is nitin khanna , this is second program ,this is a word , word its "
stop_words = set(stopwords.words("english"))

#print (stop_words)
solution = []
for word in (word_tokenize(example)):
	if word not in stop_words:
		solution.append(word)

print(solution)

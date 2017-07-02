
from nltk import sent_tokenize, word_tokenize


#to update
#nltk.download()

#########################################################################
#tokenize - word based or sentence based
# lexicon and corporas 
# corpora - body of text. ex medical , journals, presidential speech 
# lexicon - words  and their meaning

# investor-speak ---- regular english speak

# investor speaks 'bull' = positive about market
# english speaks 'bull ' = animal
#########################################################################

example_text = "hellow , hi are you today, i am enjoying python Mr. , what about you ."

#sent_tokenize() give output as list of sentences
print (sent_tokenize(example_text))

#word_tokenizer() give output as list of words
print(word_tokenize(example_text))

for words in (word_tokenize(example_text)):
	print (words)

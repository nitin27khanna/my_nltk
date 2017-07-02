##wordnet contains word with their meanings 

from nltk.corpus import wordnet

syns = wordnet.synsets("program")

print (syns)


#synset

print(syns[0].name())

#just the word
print (syns[0].lemmas())

#definition 
print (syns[0].definition())

#example
print (syns[0].examples())


synonyms = []
antonyms = []

for  w in wordnet.synsets("good"):
	for i in w.lemmas():
		synonyms.append(i.name())
		if i.antonyms():
			antonyms.append(i.antonyms()[0].name())
print (synonyms)
print (antonyms)

w1 = wordnet.synsets("boat.n.01")
w2 = wordnet.synsets("ship.n.01")
print (w1.wup_similarity(w2))

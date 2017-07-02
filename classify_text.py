import nltk 
import random
from nltk.corpus import movie_reviews


##import movie_reviews package and classify words 



document = [(list(movie_reviews.words(fileid)),category )
		for category in movie_reviews.categories()
		for fileid in movie_reviews.fileids(category)]

random.shuffle(document)
print (document[1])

#print (document["stupid"])
##checking how many times comes in frequency
all_words = [w.lower() for w in movie_reviews.words()]

print (all_words[:10])

all_words = nltk.FreqDist(all_words)

##how many times most freq comes

print(all_words.most_common(5))

#how many times stupid comes
print(all_words["stupid"])





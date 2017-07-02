import nltk
import random	
from nltk.corpus import movie_reviews
import pickle
documents = [(list(movie_reviews.words(fileid)),category)
		for category in movie_reviews.categories()
		for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

all_words = []

for w in movie_reviews.words():
	all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)

word_features = list(all_words.keys())[:3000]

def find_features(document):
	words = set(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)
	
	return features

featuresets = [(find_features(rev),category) for (rev,category) in documents]

testing_set = featuresets[:1900]
training_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print ("Naive Bayes Accuracy : ",nltk.classify.accuracy(classifier,testing_set)*100)
classifier.show_most_informative_features(10)


##to save time in training classifiers again and again we dump classifiers to 
##pickle file as shown
##wb is write in bytes
save_classifier = open("naivebayes.pickle","wb")

pickle.dump(classifier,save_classifier)
save_classifier.close()

##to load the classifier 
#classifier_f = open("clas.pickle","rb")
#classifier = pickle.load(classifier_f)
#classifier_f.close()

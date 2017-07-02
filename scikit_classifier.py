import nltk 
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier

from sklearn.naive_bayes import MultinomialNB , GaussianNB , BernoulliNB
from sklearn.linear_model import LogisticRegression , SGDClassifier 
from sklearn.svm import LinearSVC,SVC,NuSVC


##import movie_reviews package and classify words 



document = [(list(movie_reviews.words(fileid)),category )
		for category in movie_reviews.categories()
		for fileid in movie_reviews.fileids(category)]

random.shuffle(document)
#print (document[1])

#print (document["stupid"])
##checking how many times comes in frequency
all_words = [w.lower() for w in movie_reviews.words()]
#############################################################3
###print (all_words[:10])
all_words = nltk.FreqDist(all_words)

###how many times most freq comes

###print(all_words.most_common(5))

#how many times stupid comes
###print(all_words["stupid"])
##################################################################

words_features = list(all_words.keys())[:3000]

def find_features(documents):
	words = set(documents)
	features = {}
	for w in words_features :
		features[w] = (w in words)
	
	return features

#print ((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev),category) for (rev,category) in document]
print(featuresets[1:10])

##Naive Bayes Algorithm
##posterior = prior occurences x likelihood / evidence 
training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print ("Naive Bayes Algo accuracy :" , (nltk.classify.accuracy(classifier,testing_set)) * 100)

#MNB_CLASSIFIER
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print ("MultinomialNB Algo accuracy :" , (nltk.classify.accuracy(MNB_classifier,testing_set)) * 100)

#GAUSSIAN_CLASSIFIER
#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print ("GaussianNB Algo accuracy :" , (nltk.classify.accuracy(GaussianNB_classifier,testing_set)) * 100)

#Bernoulli_CLASSIFIER
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(BernoulliNB_classifier,testing_set)) * 100)




#classifier.show_most_informative_features(15)

#LogisticRegression , SGDClassifier 
#from sklearm.svm import LinearSVC,SVC,NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(LogisticRegression_classifier,testing_set)) * 100)

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(SGDClassifier_classifier,testing_set)) * 100)


SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(SVC_classifier,testing_set)) * 100)

LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(LinearSVC_classifier,testing_set)) * 100)

NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(NuSVC_classifier,testing_set)) * 100)

#BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
#BernoulliNB_classifier.train(training_set)
#print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(BernoulliNB_classifier,testing_set)) * 100)


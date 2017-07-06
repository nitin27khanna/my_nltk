import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.tokenize import word_tokenize
from sklearn.naive_bayes import MultinomialNB , GaussianNB , BernoulliNB
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.svm import LinearSVC,SVC,NuSVC

from nltk.classify import ClassifierI
from statistics import mode


##import movie_reviews package and classify words



#inherits ClassifierI from nltk
class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers = classifiers
   
    def classify(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return max((votes))
   
    def confidence(self,features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf









#############Trying new data set
#document = [(list(movie_reviews.words(fileid)),category )
#        for category in movie_reviews.categories()
#        for fileid in movie_reviews.fileids(category)]
#
#random.shuffle(document)
#print (document[1])

#print (document["stupid"])
##checking how many times comes in frequency
#all_words = [w.lower() for w in movie_reviews.words()]
#############################################################3
###print (all_words[:10])

#pos_file = open("data_sets/pos.txt","r",encoding = "latin-1").read()
#neg_file = open("data_sets/neg.txt","r",encoding = "latin-1").read()

#all_words = []
#document = []

#j is adjective , currently only readin adjectives

#allowed_word_types = ["J"]

#print ("FILES READ")

#for p in pos_file.split('\n'):
#    document.append((p,"pos"))
#    words = word_tokenize(p)
#    pos = nltk.pos_tag(words)   
#    for w in pos:
#        if w[1][0] in allowed_word_types:
#            all_words.append(w[0].lower())
#
#
#for p in neg_file.split('\n'):
#    document.append((p,"neg"))
#    words = word_tokenize(p)
#    pos = nltk.pos_tag(words)   
#    for w in pos:
#        if w[1][0] in allowed_word_types:
#            all_words.append(w[0].lower())
###



documents_f = open("pickled_algos/documents.pickle","rb")
documents = pickle.load(documents_f)
documents_f.close()





words_features_f = open("pickled_algos/words_feature5k.pickle","rb")
words_features = pickle.load(words_features_f)
words_features_f.close()

def find_features(documents):
    words = word_tokenize(documents)
    features = {}
    for w in words_features :
        features[w] = (w in words)
   
    return features

#print ((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets_f = open("pickled_algos/featuresets.pickle","rb")
featuresets = pickle.load(featuresets_f)
featuresetes_f.close()

random.shuffle(featuresets)

print(len(feature made))
##Naive Bayes Algorithm
##posterior = prior occurences x likelihood / evidence

training_set = featuresets[:10000]
testing_set = featuresets[10000:]
print("training made")

##Naive Bayes
open_file= open("pickle_algos/naivesbayes.pickle","rb")
classifier = pickle.load(open_file)
open_file.close()


#MNB_CLASSIFIER
open_file= open("pickle_algos/MNB_Classifier.pickle","rb")
MNB_classifier = pickle.load(open_file)
open_file.close()


#GAUSSIAN_CLASSIFIER
#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print ("GaussianNB Algo accuracy :" , (nltk.classify.accuracy(GaussianNB_classifier,testing_set)) * 100)

#Bernoulli_CLASSIFIER
#BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
#BernoulliNB_classifier.train(training_set)
#print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(BernoulliNB_classifier,testing_set)) * 100)

open_file = open("pickled_algos/BernoulliNB_classifier.pickle","rb")
BernoulliNB_classifier = pickle.load(open_file)
open_file.close()



#classifier.show_most_informative_features(15)

#LogisticRegression , SGDClassifier
#from sklearm.svm import LinearSVC,SVC,NuSVC

#LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
#LogisticRegression_classifier.train(training_set)
#print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(LogisticRegression_classifier,testing_set)) * 100)


open_file = open("pickled_algos/LogisticRegression_classifier.pickle","rb")
LogisticRegression_classifier = pickle.load(open_file)
open_file.close()

#SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
#SGDClassifier_classifier.train(training_set)
#print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(SGDClassifier_classifier,testing_set)) * 100)


open_file = open("pickled_algos/SGDClassifier_classifier.pickle","rb")
SGDClassifier_classifier = pickle.load(open_file)
open_file.close()



#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(SVC_classifier,testing_set)) * 100)

open_file = open("pickled_algos/SVC_classifier.pickle","rb")
SVC_classifier = pickle.load(open_file)
open_file.close()




#LinearSVC_classifier = SklearnClassifier(LinearSVC())
#LinearSVC_classifier.train(training_set)
#print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(LinearSVC_classifier,testing_set)) * 100)

open_file = open("pickled_algos/LinearSVC_classifier.pickle","rb")
LinearSVC_classifier = pickle.load(open_file)
open_file.close()






#NuSVC_classifier = SklearnClassifier(NuSVC())
#NuSVC_classifier.train(training_set)
#print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(NuSVC_classifier,testing_set)) * 100)

open_file = open("pickled_algos/NuSVC_classifier.pickle","rb")
NuSVC_classifier = pickle.load(open_file)
open_file.close()




#BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
#BernoulliNB_classifier.train(training_set)
#print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(BernoulliNB_classifier,testing_set)) * 100)

votedClassifier = VoteClassifier(
                                  classifier,
                                  MNB_classifier,
                                 SVC_classifier,
                                 SGDClassifier_classifier,
                                 LogisticRegression_classifier)

#print ("Voted Classifeir accuracy : " , (nltk.classify.accuracy(votedClassifier,testing_set))*100)



def sentiments(text):
    feats = find_features(text)
    return voted_classifier.classify(feats),voted_classifier.confidence(feats)
    

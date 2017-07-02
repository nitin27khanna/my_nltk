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

pos_file = open("data_sets/pos.txt","r",encoding = "latin-1").read()
neg_file = open("data_sets/neg.txt","r",encoding = "latin-1").read()

all_words = []
document = []

#j is adjective , currently only readin adjectives

allowed_word_types = ["J"]


print ("FILES READ")

for p in pos_file.split('\n'):
    document.append((p,"pos"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)   
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())


for p in neg_file.split('\n'):
    document.append((p,"neg"))
    words = word_tokenize(p)
    pos = nltk.pos_tag(words)   
    for w in pos:
        if w[1][0] in allowed_word_types:
            all_words.append(w[0].lower())




save_documents = open("pickled_algos/documents.pickle","wb")
pickle.dump(document,save_documents)
save_documents.close()




all_words = nltk.FreqDist(all_words)

print ("all_words made")

###how many times most freq comes

###print(all_words.most_common(5))

#how many times stupid comes
###print(all_words["stupid"])
##################################################################

words_features = list(all_words.keys())[:5000]

save_words_features = open("pickled_algos/words_feature5k.pickle","wb")
pickle.dump(document,save_words_features)
save_words_features.close()

def find_features(documents):
    words = word_tokenize(documents)
    features = {}
    for w in words_features :
        features[w] = (w in words)
   
    return features

#print ((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev),category) for (rev,category) in document]
#print(featuresets[1:10])

random.shuffle(featuresets)

print("feature made")
##Naive Bayes Algorithm
##posterior = prior occurences x likelihood / evidence
training_set = featuresets[:10000]
testing_set = featuresets[10000:]
print("training made")


classifier = nltk.NaiveBayesClassifier.train(training_set)
print ("Naive Bayes Algo accuracy :" , (nltk.classify.accuracy(classifier,testing_set)) * 100)

save_naive_bayes = open("pickled_algos/naivebayes.pickle","wb")
pickle.dump(classifier,save_naive_bayes)
save_naive_bayes.close()

#MNB_CLASSIFIER
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print ("MultinomialNB Algo accuracy :" , (nltk.classify.accuracy(MNB_classifier,testing_set)) * 100)

save_MNB_classifier = open("pickled_algos/MNB_classifier.pickle","wb")
pickle.dump(MNB_classifier,save_MNB_classifier)
save_MNB_classifier.close()


#GAUSSIAN_CLASSIFIER
#GaussianNB_classifier = SklearnClassifier(GaussianNB())
#GaussianNB_classifier.train(training_set)
#print ("GaussianNB Algo accuracy :" , (nltk.classify.accuracy(GaussianNB_classifier,testing_set)) * 100)

#Bernoulli_CLASSIFIER
BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
BernoulliNB_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(BernoulliNB_classifier,testing_set)) * 100)

save_BernoulliNB_classifier = open("pickled_algos/BernoulliNB_classifier.pickle","wb")
pickle.dump(MNB_classifier,save_BernoulliNB_classifier)
save_BernoulliNB_classifier.close()



#classifier.show_most_informative_features(15)

#LogisticRegression , SGDClassifier
#from sklearm.svm import LinearSVC,SVC,NuSVC

LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(LogisticRegression_classifier,testing_set)) * 100)


save_LogisticRegression_classifier = open("pickled_algos/LogisticRegression_classifier.pickle","wb")
pickle.dump(LogisticRegression_classifier,save_LogisticRegression_classifier)
save_LogisticRegression_classifier.close()

SGDClassifier_classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(SGDClassifier_classifier,testing_set)) * 100)


save_SGDClassifier_classifier = open("pickled_algos/SGDClassifier_classifier.pickle","wb")
pickle.dump(SGDClassifier_classifier,save_SGDClassifier_classifier)
save_SGDClassifier_classifier.close()



SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(SVC_classifier,testing_set)) * 100)

save_SVC_classifier = open("pickled_algos/SVC_classifier.pickle","wb")
pickle.dump(SVC_classifier,save_SVC_classifier)
save_SVC_classifier.close()



LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(LinearSVC_classifier,testing_set)) * 100)


save_LinearSVC_classifier = open("pickled_algos/LinearSVC_classifier.pickle","wb")
pickle.dump(LinearSVC_classifier,save_LinearSVC_classifier)
save_LinearSVC_classifier.close()





NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(NuSVC_classifier,testing_set)) * 100)



save_NuSVC_classifier = open("pickled_algos/NuSVC_classifier.pickle","wb")
pickle.dump(SVC_Nuclassifier,save_NuSVC_classifier)
save_NuSVC_classifier.close()


#BernoulliNB_classifier = SklearnClassifier(BernoulliNB())
#BernoulliNB_classifier.train(training_set)
#print ("BernoulliNB Algo accuracy :" , (nltk.classify.accuracy(BernoulliNB_classifier,testing_set)) * 100)

votedClassifier = VoteClassifier(classifier,MNB_classifier,BernoulliNB_classifier,NuSVC_classifier,LinearSVC_classifier,SVC_classifier,SGDClassifier_classifier,LogisticRegression_classifier)

print ("Voted Classifeir accuracy : " , (nltk.classify.accuracy(votedClassifier,testing_set))*100)



def sentiments(text):
    feats = find_features(text)
    return votedClassifier.classify(feats)

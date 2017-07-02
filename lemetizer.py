from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize("cata"))

print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("green"))
print(lemmatizer.lemmatize("rocks"))
print(lemmatizer.lemmatize("python"))

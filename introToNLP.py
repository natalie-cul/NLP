#https://pub.towardsai.net/natural-language-processing-nlp-with-python-tutorial-for-beginners-1f54e610a1a0
import nltk
#nltk.download('punkt') ## only need to run this once
from nltk import word_tokenize,sent_tokenize

text_file = open("banana.txt")
text = text_file.read()


#print text type, text and length of text
#print(type(text))
#print("\n")

#print(text)
#print("\n")

#print (len(text))

#divide text into sentences, and print
sentences = sent_tokenize(text)
#print(len(sentences))
#for x in sentences:
#    print(x)
#    print("----")

#divide text into words, and print
words = word_tokenize(text)
#print(len(words))
#for x in words:
#    print(x)
#    print("----")

#############################
#Find the frequence of words in text
from nltk.probability import FreqDist

fdist = FreqDist(words)
#print the 10 most common words
mostCommon10 = fdist.most_common(10)
#for x in mostCommon10:
#    print(x)
#plot a graph of word distribution
import matplotlib.pyplot as plot
#fdist.plot(10)

############
#remove punctuation marks
words_no_punc = []
for w in words:
    if w.isalpha():
        words_no_punc.append(w.lower())
#print(words_no_punc)
############
#most common words after removing punc
fdistNoPunc = FreqDist(words_no_punc)
#print the 10 most common words
mostCommon10 = fdistNoPunc.most_common(10)
for x in mostCommon10:
    print(x)
#graph of word distribution without punc
fdistNoPunc.plot(10)

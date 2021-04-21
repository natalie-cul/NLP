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
#for x in mostCommon10:
#    print(x)
#graph of word distribution without punc
#fdistNoPunc.plot(10)

##Stopwords
from nltk.corpus import stopwords
#run once
#nltk.download('stopwords')
stopwords = stopwords.words("english")
#print(stopwords)
clean_words = []
for w in words_no_punc:
    if w not in stopwords:
        clean_words.append(w)
#print(clean_words)
#print("\n")
#print(len(clean_words))

############
#most common words after removing stopwords
fdistNoStopWords = FreqDist(clean_words)
#print the 10 most common words
mostCommon10 = fdistNoStopWords.most_common(10)
#for x in mostCommon10:
#    print(x)
#graph of word distribution without punc
#fdistNoStopWords.plot(10)
#wordcloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

###original wordcloud format
#wordcloud = WordCloud().generate(text)
#plt.figure(figsize = (12,12))
#plt.imshow(wordcloud)

#plt.axis("off")
#plt.show()

##Circle wordcloud format
import numpy as np
from PIL import Image

char_mask = np.array(Image.open("circle.png"))
wordcloud = WordCloud(background_color="black", normalize_plurals=True,mask=char_mask).generate(text)
plt.figure(figsize = (8,8))
plt.imshow(wordcloud)

plt.axis("off")
plt.show()

#Stems
from nltk.stem import PorterStemmer

porter = PorterStemmer()
word_list = ["banana", "bananas"]
#for w in word_list:
#    print(porter.stem(w))
#run once
#nltk.download('wordnet')

from nltk import WordNetLemmatizer
lemma = WordNetLemmatizer()
#word_list = ["Study", "Studying", "Studies", "Studied"]
word_list = ["am", "is", "are", "was", "were"]
#for w in word_list:
#    print(lemma.lemmatize(w,pos = "v"))

#PoS tagging:
#run once:
nltk.download('averaged_perceptron_tagger')
sentence = "Buuny the cat is being very rude"
tokenized_words = word_tokenize(sentence)

for words in tokenized_words:
    tagged_words = nltk.pos_tag(tokenized_words)

print(tagged_words)

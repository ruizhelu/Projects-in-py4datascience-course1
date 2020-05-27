import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('words')
import pandas as pd
import numpy as np

# If you would like to work with the raw text you can use 'moby_raw'
with open('moby.txt', 'r') as f:
    moby_raw = f.read()

# If you would like to work with the novel in nltk.Text format you can use 'text1'
moby_tokens = nltk.word_tokenize(moby_raw)
text1 = nltk.Text(moby_tokens)

def example_one():

    return len(nltk.word_tokenize(moby_raw)) # or alternatively len(text1)

example_one()

def example_two():

    return len(set(nltk.word_tokenize(moby_raw))) # or alternatively len(set(text1))

example_two()

from nltk.stem import WordNetLemmatizer

def example_three():

    lemmatizer = WordNetLemmatizer()
    lemmatized = [lemmatizer.lemmatize(w,'v') for w in text1]

    return len(set(lemmatized))

example_three()

def answer_one():

    a1 = len(set(text1))/len(text1)
    return a1

answer_one()

def answer_two():

    dist = nltk.FreqDist(text1)
    a2 = ((dist['whale'] + dist['Whale'])/len(text1))/100
    return a2

answer_two()

def answer_three():

    a3 = nltk.FreqDist(text1).most_common(20)

    return a3

answer_three()

def answer_four():

    dist = nltk.FreqDist(text1)
    vocab1 = dist.keys()
    a4 = [w for w in vocab1 if len(w) > 5 and dist[w] > 150]

    return sorted(a4)

answer_four()

def answer_five():

    dist = nltk.FreqDist(text1)
    vocab1 = dist.keys()
    length = max(len(w) for w in vocab1)
    longest_word = [w for w in vocab1 if len(w) == length]
    a5 = (longest_word[0], length)

    return a5

answer_five()

def answer_six():

    dist = nltk.FreqDist(text1)
    word = [w for w in set(text1) if dist[w] > 2000 and w.isalpha() == True]
    frequency = [dist[w] for w in word]
    a6 = [(frequency[i], word[i]) for i in range(0, len(frequency))]
    a6 = sorted(a6, key=lambda x:x[0], reverse=True)

    return a6

answer_six()

def answer_seven():

    sens = nltk.sent_tokenize(moby_raw)
    a7 = len(moby_tokens)/len(sens)

    return a7

answer_seven()

def answer_eight():
    parts = nltk.pos_tag(text1)
    parts = [parts[i][1] for i in range(len(parts))]
    a8 = nltk.FreqDist(parts).most_common(5)

    return a8

answer_eight()

from nltk.corpus import words

correct_spellings = words.words()

def answer_nine(entries=['cormulent', 'incendenece', 'validrate']):

    a9=[]
    correcters=[]
    jd=[]
    for i in range(len(entries)):
        x=entries[i]
        for y in correct_spellings:
            if y.startswith(x[0]):
                correcters.append(y)
                jd.append([nltk.jaccard_distance(set(nltk.ngrams(x,n=3)), set(nltk.ngrams(y,n=3)))][0])
        a9.append(correcters[np.argmin(jd)])
        correcters=[]
        jd=[]

    return a9

answer_nine()

def answer_ten(entries=['cormulent', 'incendenece', 'validrate']):

    a10=[]
    correcters=[]
    jd=[]
    for i in range(len(entries)):
        x=entries[i]
        for y in correct_spellings:
            if y.startswith(x[0]):
                correcters.append(y)
                jd.append([nltk.jaccard_distance(set(nltk.ngrams(x,n=4)), set(nltk.ngrams(y,n=4)))][0])
        a10.append(correcters[np.argmin(jd)])
        correcters=[]
        jd=[]

    return a10

answer_ten()

def answer_eleven(entries=['cormulent', 'incendenece', 'validrate']):

    a11=[]
    correcters=[]
    edit=[]
    for i in range(len(entries)):
        x=entries[i]
        for y in correct_spellings:
            if y.startswith(x[0]):
                correcters.append(y)
                edit.append([nltk.edit_distance(x, y, transpositions=True)][0])
        a11.append(correcters[np.argmin(edit)])
        correcters=[]
        edit=[]

    return a11

answer_eleven()

import pandas as pd
import numpy as np

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(spam_data['text'],
                                                    spam_data['target'],
                                                    random_state=0)

def answer_one():

    a1 = len(spam_data[spam_data['target']==1])*100/len(spam_data)

    return a1

answer_one()

from sklearn.feature_extraction.text import CountVectorizer

def answer_two():

    vect = CountVectorizer().fit(X_train)

    vers = [x for x in vect.vocabulary_.keys()]
    long = [len(x) for x in vect.vocabulary_.keys()]

    a2 = vers[np.argmax(long)]

    return a2

answer_two()

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():

    vect = CountVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)

    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
    y_predicted = clf.predict(vect.transform(X_test))
    a3 = roc_auc_score(y_test, y_predicted)

    return a3

answer_three()

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():

    vect = TfidfVectorizer().fit(X_train)
    X_train_vectorized = vect.transform(X_train)

    features = np.array(vect.get_feature_names())
    sorted_tfidf_index = X_train_vectorized.max(0).toarray()[0].argsort()
    sorted_tfidf_values = X_train_vectorized.max(0).toarray()[0][sorted_tfidf_index]

    smallest = pd.Series(sorted_tfidf_values[:20], index=features[sorted_tfidf_index[:20]])
    largest = pd.Series(sorted_tfidf_values[-20:], index=features[sorted_tfidf_index[-20:]])

    a4 = (smallest, largest)

    return a4

answer_four()

def answer_five():

    vect = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vect.transform(X_train)

    clf = MultinomialNB(alpha=0.1).fit(X_train_vectorized, y_train)
    y_predicted = clf.predict(vect.transform(X_test))
    a5 = roc_auc_score(y_test, y_predicted)

    return a5

answer_five()

def answer_six():

    not_spam_d = spam_data[spam_data['target']==0]
    spam_d = spam_data[spam_data['target']==1]

    a6 = (np.mean([len(x) for x in not_spam_d['text']]), np.mean([len(x) for x in spam_d['text']]))

    return a6

answer_six()

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

from sklearn.svm import SVC

def answer_seven():

    vect = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    train_length = [len(x) for x in X_train]
    test_length = [len(x) for x in X_test]

    X_train_vectorized = add_feature(X_train_vectorized, train_length)
    X_test_vectorized = add_feature(X_test_vectorized, test_length)

    clf = SVC(C=10000).fit(X_train_vectorized, y_train)
    y_predicted = clf.predict(X_test_vectorized)

    a7 = roc_auc_score(y_test, y_predicted)

    return a7

answer_seven()

def answer_eight():

    not_spam_d = spam_data[spam_data['target']==0]
    spam_d = spam_data[spam_data['target']==1]
    n_not_spam_d = []
    n_spam_d = []
    n = 0
    for x in not_spam_d['text']:
        for y in x:
            if y.isdigit():
                n = n + 1
        n_not_spam_d.append(n)
        n = 0

    for x in spam_d['text']:
        for y in x:
            if y.isdigit():
                n += 1
        n_spam_d.append(n)
        n = 0

    a8 = (np.mean(n_not_spam_d), np.mean(n_spam_d))

    return a8

answer_eight()

from sklearn.linear_model import LogisticRegression

def answer_nine():

    vect = TfidfVectorizer(min_df=5, ngram_range=(1,3)).fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    train_length = [len(x) for x in X_train]
    test_length = [len(x) for x in X_test]

    n_train = []
    n_test = []
    n = 0
    for x in X_train:
        for y in x:
            if y.isdigit():
                n = n + 1
        n_train.append(n)
        n = 0

    for x in X_test:
        for y in x:
            if y.isdigit():
                n += 1
        n_test.append(n)
        n = 0

    X_train_vectorized = add_feature(X_train_vectorized, [train_length, n_train])
    X_test_vectorized = add_feature(X_test_vectorized, [test_length, n_test])

    clf = LogisticRegression(C=100).fit(X_train_vectorized, y_train)
    y_predicted = clf.predict(X_test_vectorized)

    a9 = roc_auc_score(y_test, y_predicted)

    return a9

    answer_nine()

    def answer_ten():

    not_spam_d = spam_data[spam_data['target']==0]
    spam_d = spam_data[spam_data['target']==1]
    n_not_spam = not_spam_d['text'].str.count('\W')
    n_spam = spam_d['text'].str.count('\W')

    a10 = (np.mean(n_not_spam), np.mean(n_spam))

    return a10

    answer_ten()

def answer_eleven():

    vect = CountVectorizer(min_df=5, ngram_range=(2,5), analyzer='char_wb').fit(X_train)
    X_train_vectorized = vect.transform(X_train)
    X_test_vectorized = vect.transform(X_test)

    train_length = [len(x) for x in X_train]
    test_length = [len(x) for x in X_test]

    n_train = []
    n_test = []
    n = 0
    for x in X_train:
        for y in x:
            if y.isdigit():
                n = n + 1
        n_train.append(n)
        n = 0

    for x in X_test:
        for y in x:
            if y.isdigit():
                n += 1
        n_test.append(n)
        n = 0

    n_X_train = X_train.str.count('\W')
    n_X_test = X_test.str.count('\W')

    X_train_vectorized = add_feature(X_train_vectorized, [train_length, n_train, n_X_train])
    X_test_vectorized = add_feature(X_test_vectorized, [test_length, n_test, n_X_test])

    clf = LogisticRegression(C=100).fit(X_train_vectorized, y_train)
    y_predicted = clf.predict(X_test_vectorized)

    auc = roc_auc_score(y_test, y_predicted)

    feature_names = np.array(vect.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])
    sorted_coef_index = clf.coef_[0].argsort()
    small = list(feature_names[sorted_coef_index[:10]])
    large = list(feature_names[sorted_coef_index[:-11:-1]])

    a11 = (auc, small, large)


    return a11

answer_eleven()

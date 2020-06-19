import numpy as np
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import pandas as pd


def convert_tag(tag):
    """Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets"""

    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return None


def doc_to_synsets(doc):
    """
    Returns a list of synsets in document.

    Tokenizes and tags the words in the document doc.
    Then finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        doc: string to be converted

    Returns:
        list of synsets

    Example:
        doc_to_synsets('Fish are nvqjp friends.')
        Out: [Synset('fish.n.01'), Synset('be.v.01'), Synset('friend.n.01')]
    """

    doc_tokens = nltk.word_tokenize(doc)
    pos_tags = nltk.pos_tag(doc_tokens)
    wn_tags = [convert_tag(x[1]) for x in pos_tags]
    synsets = [wn.synsets(x,y) for x,y in zip(doc_tokens, wn_tags)]
    synset_list = [x[0] for x in synsets if len(x) > 0]

    return synset_list

def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        synsets1 = doc_to_synsets('I like cats')
        synsets2 = doc_to_synsets('I like dogs')
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """

    max_value = []
    for synset in s1:
        value = [synset.path_similarity(x) for x in s2 if synset.path_similarity(x) is not None]
        if len(value) > 0:
            max_value.append(max(value))

    return np.mean(max_value)

def document_path_similarity(doc1, doc2):
    """Finds the symmetrical similarity between doc1 and doc2"""

    synsets1 = doc_to_synsets(doc1)
    synsets2 = doc_to_synsets(doc2)

    return (similarity_score(synsets1, synsets2) + similarity_score(synsets2, synsets1)) / 2

def test_document_path_similarity():
    doc1 = 'This is a function to test document_path_similarity.'
    doc2 = 'Use this function to see if your code in doc_to_synsets \
    and similarity_score is correct!'
    return document_path_similarity(doc1, doc2)
test_document_path_similarity()

# Use this dataframe for questions most_similar_docs and label_accuracy
paraphrases = pd.read_csv('paraphrases.csv')
paraphrases.head()

def most_similar_docs():

    paraphrases['scores'] = [document_path_similarity(x,y) for x,y in zip(paraphrases['D1'], paraphrases['D2'])]
    D1 = paraphrases.loc[np.argmax(paraphrases['scores']),'D1']
    D2 = paraphrases.loc[np.argmax(paraphrases['scores']),'D2']
    similarity_score = max(paraphrases['scores'])

    return (D1, D2, similarity_score)

most_similar_docs()

def label_accuracy():
    from sklearn.metrics import accuracy_score

    paraphrases['labels'] = 0
    for x in range(len(paraphrases['scores'])) :
        if paraphrases['scores'][x] > 0.75:
            paraphrases['labels'][x] = 1

    return accuracy_score(paraphrases['Quality'], paraphrases['labels'])

label_accuracy()

#PART2
import pickle
import gensim
from sklearn.feature_extraction.text import CountVectorizer

# Load the list of documents
with open('newsgroups', 'rb') as f:
    newsgroup_data = pickle.load(f)

# Use CountVectorizor to find three letter tokens, remove stop_words,
# remove tokens that don't appear in at least 20 documents,
# remove tokens that appear in more than 20% of the documents
vect = CountVectorizer(min_df=20, max_df=0.2, stop_words='english',
                       token_pattern='(?u)\\b\\w\\w\\w+\\b')
# Fit and transform
X = vect.fit_transform(newsgroup_data)

# Convert sparse matrix to gensim corpus.
corpus = gensim.matutils.Sparse2Corpus(X, documents_columns=False)

# Mapping from word IDs to words (To be used in LdaModel's id2word parameter)
id_map = dict((v, k) for k, v in vect.vocabulary_.items())

# Use the gensim.models.ldamodel.LdaModel constructor to estimate
# LDA model parameters on the corpus, and save to the variable `ldamodel`

# Your code here:
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10,id2word=id_map, passes=25, random_state=34)

def lda_topics():

    return ldamodel.print_topics(num_topics=10, num_words=10)

lda_topics()

new_doc = ["\n\nIt's my understanding that the freezing will start to occur because \
of the\ngrowing distance of Pluto and Charon from the Sun, due to it's\nelliptical orbit. \
It is not due to shadowing effects. \n\n\nPluto can shadow Charon, and vice-versa.\n\nGeorge \
Krumins\n-- "]

def topic_distribution():

    Sparse = vect.transform(new_doc)
    corpus = gensim.matutils.Sparse2Corpus(Sparse, documents_columns=False)
    topics = ldamodel.get_document_topics(corpus)

    return list(topics)[0]

topic_distribution()

def topic_names():

    topic_names= ['Sports','Society & Lifestyle','Society & Lifestyle','Health', 'Science',  'Computers & IT','Automobiles',
                  'Government', 'Business', 'Religion']

    return topic_names
topic_names()

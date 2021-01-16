import nltk
import numpy as np
from tqdm import tqdm
from langdetect import detect
from collections import defaultdict
from string import ascii_uppercase
import pdb

import unicodedata as ud
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

letters = (x for x in ascii_uppercase if x not in ('Q', 'X'))
mapping = {ord(ud.lookup('LATIN LETTER SMALL CAPITAL ' + x)): x for x in letters}
tt = str.maketrans(mapping)


from html.parser import HTMLParser
class HTMLTextExtractor(HTMLParser):
    def __init__(self):
        super(HTMLTextExtractor, self).__init__()
        self.result = [ ]

    def handle_data(self, d):
        self.result.append(d)

    def get_text(self):
        return ''.join(self.result)

def html_to_text(html):
    s = HTMLTextExtractor()
    s.feed(html)
    return s.get_text()


def create_vocab(sentences, bar=5):
    vocab = defaultdict(int)
    vocab['18+'] = 0
    exception = []
    for id, sentence in enumerate(tqdm(sentences)):
        sentence = html_to_text(sentence.translate(tt).lower())
        try:
            if detect(sentence) == 'en':
                tokens = nltk.word_tokenize(sentence)
                for token in tokens:
                    token = token.lower()
                    try:
                        if len(token) > 2 and detect(token) == 'en':
                            vocab[token] += 1
                    except:
                        exception.append(token)
        except:
            exception.append(sentence)

    output = defaultdict(int)
    for voc, times in vocab.items():
        if times >= bar:
            output[voc] = times

    return output, exception


def filter_posts(corpus_file, posts_file):
    count = 0
    with open(posts_file, 'w') as fp:
        with open(corpus_file, 'r') as lines:
            for line in lines:
                count += 1
                if count % 100000 == 0:
                    print(count)
                try:
                    if detect(line) == 'en':
                        fp.write(line)
                except:
                    continue


def create_token_map(id2blog, vocab):
    id2token = defaultdict(list)
    for key, sentence in tqdm(id2blog.items()):
        sentence = html_to_text(sentence.translate(tt).lower())
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            if token in vocab:
                id2token[key].append(token)
    return id2token


def group(model, word):
    return model.get_term_topics(word)[0][0]


def TSNE_graph(vocab, w2v, image_name, color=None, annotate=True):
    if color == None:
        color = ['black' for _ in vocab]

    words, embeddings, colors = [], [], []
    for i, word in enumerate(vocab):
        if word in w2v:
            words.append(word)
            embeddings.append(w2v[word])
            colors.append(color[i])

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    embeddings = tsne_model.fit_transform(embeddings)

    x, y = [], []
    for embedding in embeddings:
        x.append(embedding[0])
        y.append(embedding[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i],y[i], c=colors[i])
        if annotate:
            plt.annotate(words[i],
                        xy=(x[i], y[i]),
                        xytext=(5, 2),
                        textcoords='offset points',
                        ha='right',
                        va='bottom')
    plt.savefig(image_name)
    plt.show()


def get_embeddings(vocab, w2v):
    words, embeddings = [], []
    for word in vocab:
        if word in w2v:
            words.append(word)
            embeddings.append(w2v[word])
    return words, embeddings


def words2embedding(tokens, w2v):
    embeddings = []
    for word in tokens:
        if word in w2v:
            embeddings.append(w2v[word])
    if len(embeddings) == 0:
        pdb.set_trace()
    return np.mean(embeddings, axis=0)

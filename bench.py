
import time

from itertools import chain, izip

from mltk.aptagger import FastPerceptronTagger
from mltk.np_chunker import NPChunker

tagger = FastPerceptronTagger()
chunker = NPChunker()

def benchmark_aptagger():
    '''
    Benchmark the aptagger vs the Penn Treebank sample in nltk
    '''
    from nltk.corpus import treebank

    # we want to remove "-NONE-" tags since these appear to be garbage
    text = []
    tags = []
    k = 0
    for sentence in treebank.tagged_sents():
        text.append([ele[0] for ele in sentence if ele[1] != '-NONE-'])
        tags.extend([ele[1] for ele in sentence if ele[1] != '-NONE-'])
        k += 1

    t1 = time.time()
    predicted = tagger.tag_sents(text)
    t2 = time.time()

    ncorrect = sum(bool(t == p[1])
        for t, p in izip(tags, chain.from_iterable(predicted)))

    print("Took %s seconds to tag %s sentences (%s tokens)" % (
        t2 - t1, k, len(tags)))
    print("Accuracy: %s" % (float(ncorrect) / len(tags)))


def benchmark_np_chunker():
    '''
    Benchmark the NP chunker vs the Conll 2000 test set in NLTK
    '''
    from collections import defaultdict
    from nltk.corpus import conll2000

    def read_data():
        from collections import defaultdict
        # read in the data
        # need the raw tokens and the corresponding noun phrases
        tokens = []
        actual_noun_phrases = []
        for sentence in conll2000.iob_sents('test.txt'):
            # get the IOB tags for NP only
            # each element in sentence is (token, Brill POS tag, IOB)
            tokens.append([ele[0] for ele in sentence])
            phrase = []
            phrases = []
            for ele in sentence:
                es = ele[2].split('-')
                if len(es) == 2 and es[1] == 'NP':
                    # it's a part of a noun phrase
                    if es[0] == 'I':
                        phrase.append(ele[0])
                    else:
                        if len(phrase) > 0:
                            phrases.append(' '.join(phrase))
                            phrase = []
                        if es[0] == 'B':
                            phrase.append(ele[0])
            if len(phrase) > 0:
                phrases.append(' '.join(phrase))
            actual_noun_phrases.append(phrases)
        return tokens, actual_noun_phrases

    def prec_rec_f1(predicted, actual):
        act = defaultdict(lambda: 0)
        for phrase in actual:
            act[phrase] += 1

        true_positive = 0
        for phrase in predicted:
            if act[phrase] > 0:
                true_positive += 1
                act[phrase] -= 1

        if len(predicted) == 0:
            precision = 0.0
        else:
            precision = true_positive / float(len(predicted))

        if len(actual) == 0:
            recall = 0.0
        else:
            recall = true_positive / float(len(actual))

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)

        return precision, recall, f1


    # load in the test set
    all_tokens, actual_noun_phrases = read_data()

    # tag tokens with POS
    t1 = time.time()
    tags = tagger.tag_sents(all_tokens)
    pos_time = time.time() - t1

    # now get the noun phrases
    t2 = time.time()
    noun_phrases = chunker.chunk_sents(tags)
    np_time = time.time() - t2

    # finally compute precision, recall, f1
    npredicted_chunks = 0
    npredicted_correct_chunks = 0
    nactual_chunks = 0
    nactual_predicted_chunks = 0
    for actual_chunks, predicted in izip(actual_noun_phrases, noun_phrases):
        predicted_chunks = []
        for chunk in predicted:
            tokens = [token for token, tags in chunk]
            predicted_chunks.append(' '.join(tokens))

        npredicted_chunks += len(predicted_chunks)
        nactual_chunks += len(actual_chunks)
        p, r, f = prec_rec_f1(predicted_chunks, actual_chunks)
        npredicted_correct_chunks += len(predicted_chunks) * p
        nactual_predicted_chunks += len(actual_chunks) * r

    prec = float(npredicted_correct_chunks) / npredicted_chunks
    recall = float(nactual_predicted_chunks) / nactual_chunks
    f1 = 2 * prec * recall / (prec + recall)
    print("Precision, Recall, F1 on CoNLL-2000 test set")
    print((prec, recall, f1))

    print("Time for POS tagging: %s" % pos_time)
    print("Time for NP chunking after POS tagging: %s" % np_time)
    print("(for %s sentences, %s tokens)" % (
        len(all_tokens), sum(len(sent) for sent in all_tokens)))



def benchmark_aptagger():
    '''
    Benchmark the aptagger vs the Penn Treebank sample in nltk
    '''
    import time
    from itertools import chain, izip

    from nltk.corpus import treebank
    from mltk.aptagger import FastPerceptronTagger

    # we want to remove "-NONE-" tags since these appear to be garbage
    text = []
    tags = []
    k = 0
    for sentence in treebank.tagged_sents():
        text.append([ele[0] for ele in sentence if ele[1] != '-NONE-'])
        tags.extend([ele[1] for ele in sentence if ele[1] != '-NONE-'])
        k += 1

    tagger = FastPerceptronTagger()
    t1 = time.time()
    predicted = tagger.tag_sents(text)
    t2 = time.time()

    ncorrect = sum(bool(t == p[1])
        for t, p in izip(tags, chain.from_iterable(predicted)))

    print("Took %s seconds to tag %s sentences (%s tokens)" % (
        t2 - t1, k, len(tags)))
    print("Accuracy: %s" % (float(ncorrect) / len(tags)))


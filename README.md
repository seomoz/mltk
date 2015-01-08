mltk
====

[![Build Status](https://api.travis-ci.org/seomoz/mltk.png)](https://api.travis-ci.org/seomoz/mltk.png)

mltk - Moz Language Tool Kit.  Like `nltk` but faster.

The overall design goal for `mltk` is fast components that are "good enough",
unlike most NLP libraries that trade speed for accuracy.
"Good enough" in this context is typically a 0.5 - 1% decrease
in accuracy or F score on standard benchmarks vs published state
of the art results in order to gain an order of magnitude or more throughput.

**Note:** this is currently under development, expect undocumented API
changes until things stabilize.

POS Tagger
----------

There is a C++ implementation of the part of speech tagger in
[textblob-aptagger](https://github.com/sloria/textblob-aptagger).
It implements the NLTK POS tagger interface (`tag` and `tag_sents` methods)
that assign POS tags to word and sentence tokenized text.  Example usage,
using NLTK to do the tokenization:

```python
from nltk import word_tokenize, sent_tokenize
from mltk.aptagger import FastPerceptronTagger

tagger = FastPerceptronTagger()

doc = "This is a document as a utf-8 encoded string.  It has two sentences."
tokens = [word_tokenize(sent) for sent in sent_tokenize(doc)]
tags = tagger.tag_sents(tokens)
```

NP Chunker
----------

Our noun phrase chunker generally follows the averaged perceptron approach
in [Collins (2002)](http://scholar.google.com/scholar?hl=en&q=Discriminative+training+methods+for+hidden+Markov+models%3A+theory+and+experiments+with+perceptron+algorithms), using most of the features in
[Sha and Pereira (2003)](http://scholar.google.com/scholar?q=Shallow+Parsing+with+Conditional+Random+Fields).
It achieves a F score of 93.1% on the CoNLL-2000 test set (compared
to published state of the art values of 94.39%).

We made a few modifications to these algorithms to improve the speed
at the expense of a little accuracy:

* We use feature hashing with a hash size 2^17 which we found to be a good tradeoff between accuracy and model size.
* We use a greedy approach instead of beam or other search.

Usage usage:

```python
from nltk import word_tokenize, sent_tokenize
from mltk.aptagger import FastPerceptronTagger
from mltk.np_chunker import NPChunker

tagger = FastPerceptronTagger()
chunker = NPChunker()

doc = "This is a document as a utf-8 encoded string.  It has two sentences."
tokens = [word_tokenize(sent) for sent in sent_tokenize(doc)]
tags = tagger.tag_sents(tokens)
chunks = chunker.chunk_sents(tags)
```

Benchmarks
----------

The script `bench.py` provides benchmarks for both the POS tagger and
NP chunker.  The times below were run on a mid 2014 MacBook Pro
(2.5 GHz Intel Core i7).

Our implementation of the POS tagger achieves 98.8% accuracy on the small
Penn Treebank sample in NLTK and can tag an average webpage in 1-2
milliseconds (6-700,000 tokens per second) after tokenization.

The NP chunker achieves 93.1% F score on the CoNLL-2000 test set
and can chunk an average web page in about 3 milliseconds (4-500,000 tokens
per second).



mltk
====

mltk - Moz Language Tool Kit.  Like `nltk` but faster.

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


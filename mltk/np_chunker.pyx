
# c imports
cimport cython
from np_chunker cimport *

# python imports
import os
import pkgutil
try:
    # if faster option is available use it
    import simplejson as json
except ImportError:
    import json

from gzip import GzipFile
from StringIO import StringIO

cdef class NPChunker:
    def __cinit__(self, rules={}):
        '''
        Initialize the chunker.
        Load the model and construct the C++ class

        rules: an optional list of rules to apply after chunking.
            Currenly supports:
                combine_np: If True, then combines frequent chunks
                    that are separated by preposition phrases, e.g.
                    "Statue of Liberty" is returned as a single
                    chunk instead of "Statue" and "Liberty" as two separate
                    ones.  Uses a local bi-gram count over the document to
                    determine which NPs to combine.  Default is False
        '''
        with GzipFile(
                fileobj=StringIO(pkgutil.get_data(
                'mltk', os.path.join('models', 'np_chunker.json.gz'))),
                mode='r') as fin:
            model_weights = json.load(fin)
        # in C, labelmap is string -> char
        # model_weights stores label as a string, e.g. 'I'
        # to get conversion from python to C, need to convert the string
        # to int value with ord
        labelmap = {k: ord(v)
            for k, v in model_weights['labelmap'].iteritems()}
        self._chunkerptr = new FastNPChunker(
            model_weights['weights'], labelmap, rules)

    def __dealloc__(self):
        del self._chunkerptr

    def chunk_sents(self, sentences, iob=False):
        '''
        Sentences = a list of tokenized and POS tagged sentences, e.g.
            [[('The', 'DT'), ('sentence', 'NN'), ('.', '.')], ...]

        If IOB is true then returns the NP-IOB tags for each tokens, e.g.
            [[('The', 'DT', 'B'), ('sentence', 'NN', 'I'), ('.', '.', 'O')],
                 ...]
        If IOB is false then returns just the noun phrases as (token, tag)
            tuples
        '''
        # we'll call the C method and let Cython do the automatic conversion
        cdef vector[iob_label_t] iob_labels
        cdef vector[vector[np_t] ] noun_phrases

        if not iob:
            self._chunk_sentences(sentences, noun_phrases)
            return noun_phrases
        else:
            self._tag_sentences(sentences, iob_labels)
            return self._unpack_struct(iob_labels)

    def chunk(self, sentence, iob=False):
        '''
        Sentence = a list of tokens and POS tags
        '''
        return self.chunk_sents([sentence], iob)[0]

    def _unpack_struct(self, iob_labels):
        ret = []
        for sentence in iob_labels:
            ret.append(
                [(label['token'], label['tag'], chr(label['label']))
                for label in sentence])
        return ret

    cdef void _tag_sentences(self,
        vector[vector[tag_t] ]& document, vector[iob_label_t]& iob):
        '''forwarding method'''
        self._chunkerptr.tag_sentences(document, iob)

    cdef void _chunk_sentences(self,
        vector[vector[tag_t] ]& document, vector[vector[np_t] ]& noun_phrases):
        '''forwarding method'''
        self._chunkerptr.chunk_sentences(document, noun_phrases)


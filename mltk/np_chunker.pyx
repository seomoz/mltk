
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
    def __cinit__(self):
        '''
        Initialize the chunker.
        Load the model and construct the C++ class
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
            model_weights['weights'], labelmap)

    def __dealloc__(self):
        del self._chunkerptr

    def chunk_sents(self, sentences, iob=False):
        '''
        Sentences = a list of tokenized and POS tagged sentences, e.g.
            [[('The', 'DT'), ('sentence', 'NN'), ('.', '.')], ...]

        If IOB is true then returns the NP-IOB tags for each tokens, e.g.
            [[('The', 'DT', 'B'), ('sentence', 'NN', 'I'), ('.', '.', 'O')],
                 ...]
        '''
        # we'll call the C method and let Cython do the automatic conversion
        cdef vector[iob_label_t] iob_labels

        if not iob:
            # we'll implement this shortly..
            raise ValueError
        else:
            self._tag_sentences(sentences, iob_labels)
            return self._unpack_struct(iob_labels)

    def _unpack_struct(self, iob_labels):
        ret = []
        for sentence in iob_labels:
            ret.append(
                [(label['token'], label['tag'], chr(label['label']))
                for label in sentence])
        return ret

    cdef void _tag_sentences(
        self, vector[vector[tag_t] ]& document, vector[iob_label_t]& iob):
        '''forwarding method'''
        self._chunkerptr.tag_sentences(document, iob)


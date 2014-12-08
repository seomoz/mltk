# c imports
cimport cython
from aptagger cimport *

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

cdef class FastPerceptronTagger:
    def __cinit__(self):
        '''
        Initialize the tagger.
        Load the model and construct the C++ class
        '''
#        weights, tagdict, classes = cPickle.loads(pkgutil.get_data(
#                'textblob_aptagger',
#                'trontagger-0.1.0.pickle'))

        # need to convert weights, tagdict to the format needed
        # in C code
        # cython makes the conversion between python string/float
        # to C++ string/float for us
#        weights_vec, bias_weights = weights_to_vector(weights)
#
#        cdef weights_in_t cweights
#        cdef map[string, class_weights_in_t] cone_feature_weight
#        cdef class_weights_in_t one_weight
#        cweights.clear()
#        for i, one_feature_weights in enumerate(weights_vec):
#            cone_feature_weight.clear()
#            for word, word_weights in one_feature_weights.iteritems():
#                one_weight.clear()
#                for t, w in word_weights.iteritems():
#                    one_weight.push_back((t, w))
#                cone_feature_weight[word] = one_weight
#            cweights.push_back(cone_feature_weight)
#
#        cdef class_weights_in_t cbias_weights
#        cbias_weights.clear()
#        for t, v in bias_weights.iteritems():
#            cbias_weights.push_back((t, v))
#
#        cdef tagmap_in_t ctagdict
#        ctagdict.clear()
#        for word, class_tag in tagdict.iteritems():
#            ctagdict[word] = class_tag
#
#        self._taggerptr = new PerceptronTagger(
#            cweights, cbias_weights,  ctagdict)

        with GzipFile(
                fileobj=StringIO(pkgutil.get_data(
                'mltk', os.path.join('models', 'aptagger-0.1.0.json.gz'))),
                mode='r') as fin:
            model_weights = json.load(fin)
        self._taggerptr = new PerceptronTagger(
            model_weights['weights'], model_weights['bias_weights'],
            model_weights['specified_tags'])

    def __dealloc__(self):
        del self._taggerptr

    def tag_sents(self, sentences):
        '''
        Sentences = a list of tokenized sentences, e.g.
            [['The', 'first', '.'], ['The', 'second', '!']]
        Returns a list of tagged token tuples:
            [[('The', 'DT'), ('first', 'JJ'), ('.', '.')], ...]
        '''
        # we'll call the C method and let Cython do the automatic conversion
        cdef vector[tag_t] tags
        self._tag_sentences(sentences, tags)
        return tags

    cdef void _tag_sentences(self, vector[vector[string] ]& document,
                       vector[tag_t]& tags):
        '''forwarding method'''
        self._taggerptr.tag_sentences(document, tags)


def _weights_to_vector(weights):
    '''
    A utility method to convert the original weights
        from textblob_aptagger to the form used in the C++ tagger

    Weights are stored as a dict: feature -> {'class': weight, ...}
    However, all the features have common prefixes so we can
        remove the the prefix and group the same prefix values
        into a single dict

    The vector form with out prefix looks like:
        [
            # for the first feature
            {word1: {'class1': weight1, ...}, word2: {}, ...},
            # second feature
            {word1: {'class1': weight1, ...}, word2: {}, ...},
            ...
    '''
    import re
    feature_names = ["i suffix",
        "i pref1",
        "i-1 tag",
        "i-2 tag",
        "i tag\+i-2 tag",
        "i word",
        "i-1 tag\+i word",
        "i-1 word",
        "i-1 suffix",
        "i-2 word",
        "i\+1 word",
        "i\+1 suffix",
        "i\+2 word",
    ]

    regex = [re.compile(ele + " ") for ele in feature_names]

    ret = []
    for k in xrange(len(feature_names)):
        ret.append({})

    for k, v in weights.iteritems():
        if k != 'bias':
            found = False
            i = 0
            while not found:
                if i == len(feature_names):
                    raise KeyError
                if regex[i].match(k):
                    found = True
                    suffix = regex[i].sub('', k)
                    ret[i][suffix] = [(c, w) for c, w in v.iteritems()]
                else:
                    i += 1
        else:
            # the bias weights
            bias_weights = [(c, w) for c, w in v.iteritems()]

    return ret, bias_weights


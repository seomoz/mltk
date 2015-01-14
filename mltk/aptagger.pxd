# c imports
cimport cython

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.map cimport map

ctypedef vector[pair[string, float] ] class_weights_in_t
ctypedef vector[map[string, class_weights_in_t] ] weights_in_t
ctypedef map[string, string] tagmap_in_t
ctypedef pair[string, string] tag_t

# wrappers for the C++ classes we'll use
cdef extern from "_ctagger.cc":
    cdef cppclass PerceptronTagger:
        PerceptronTagger(
            weights_in_t weights,
            class_weights_in_t bias_weights,
            tagmap_in_t specified_tags)
        void tag_sentences(
            vector[vector[string] ]& document,
            vector[vector[tag_t] ]& tags
        )

# only need to define C attributes and methods here
cdef class FastPerceptronTagger:
    cdef PerceptronTagger *_taggerptr
    cdef void _tag_sentences(
        self, vector[vector[string] ]& document, vector[vector[tag_t] ]& tags)


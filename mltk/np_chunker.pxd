# c imports
cimport cython

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp.string cimport string
from libcpp.map cimport map

# wrappers for the C++ classes we'll use
cdef extern from "_np_chunker.cc":
    ctypedef vector[float] np_weights_t
    ctypedef map[string, char] np_labelmap_in_t
    ctypedef pair[string, string] tag_t
    ctypedef struct iob_t:
        string token
        string tag
        char label
    ctypedef vector[iob_t] iob_label_t
    ctypedef vector[tag_t] np_t;

    cdef cppclass FastNPChunker:
        FastNPChunker(
            np_weights_t weights,
            np_labelmap_in_t labelmap_in)
        void tag_sentences(
            vector[vector[tag_t] ]& document, vector[iob_label_t]& iob)
        void chunk_sentences(
            vector[vector[tag_t] ]& document,
            vector[vector[np_t] ] & noun_phrases)

# only need to define C attributes and methods here
cdef class NPChunker:
    cdef FastNPChunker *_chunkerptr
    cdef void _tag_sentences(
        self, vector[vector[tag_t] ]& document, vector[iob_label_t]& iob)
    cdef void _chunk_sentences(
        self, vector[vector[tag_t] ]& document,
        vector[vector[np_t] ] & noun_phrases)


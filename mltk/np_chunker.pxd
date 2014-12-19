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
    ctypedef vector[pair[string, string] ] tag_t
    ctypedef struct iob_t:
        string token
        string tag
        char label
    ctypedef vector[iob_t] iob_label_t

    cdef cppclass FastNPChunker:
        FastNPChunker(
            np_weights_t weights,
            np_labelmap_in_t labelmap_in)
        void iob_sentences(
            vector[tag_t]& document, vector[iob_label_t]& iob)

# only need to define C attributes and methods here
cdef class NPChunker:
    cdef FastNPChunker *_chunkerptr
    cdef void _iob_sentences(
        self, vector[tag_t]& document, vector[iob_label_t]& iob)


# distutils: language = c++
# cython: language_level=3
"""Exact WL signature interning using C++ integer vectors and ordered maps.
No finite-width signature hashes. Internal API; Python graph inputs are validated
by GraKeL before construction. Keeps the GIL; instances are not thread-safe.
"""
from libcpp.vector cimport vector
from libcpp.map cimport map as cppmap
from libcpp.algorithm cimport sort
from cython.operator cimport dereference as deref, preincrement
ctypedef long long Label
ctypedef vector[Label] Signature
ctypedef cppmap[Signature, Label] Vocabulary

cdef class Batch:
    cdef vector[vector[Py_ssize_t]] neighbors
    cdef vector[Label] labels
    cdef vector[Py_ssize_t] offsets
    cdef list keys

    def __init__(self, dict adjacency, dict labels_by_graph):
        cdef Py_ssize_t j, offset, i, k
        cdef object node, neighbor, keylist, index, graph, labs
        cdef vector[Py_ssize_t] ns
        self.keys = []
        self.offsets.push_back(0)
        offset = 0
        for j in range(len(labels_by_graph)):
            labs = labels_by_graph[j]
            graph = adjacency[j]
            keylist = list(labs)
            self.keys.append(keylist)
            index = {node: offset + i for i, node in enumerate(keylist)}
            for node in keylist:
                self.labels.push_back(labs[node])
                ns.clear()
                for neighbor in graph.get(node, ()):
                    ns.push_back(index[neighbor])
                self.neighbors.push_back(ns)
            offset += len(keylist)
            self.offsets.push_back(offset)

    def graphs(self, dict adjacency, dict extras):
        cdef Py_ssize_t j, k, offset
        cdef dict lab
        out = []
        for j in range(len(self.keys)):
            offset = self.offsets[j]
            lab = {}
            for k in range(len(self.keys[j])):
                lab[self.keys[j][k]] = self.labels[offset+k]
            out.append((adjacency[j], lab) + extras[j])
        return out

cdef class Relabeler:
    cdef vector[Vocabulary] levels

    def step(self, Batch batch, Py_ssize_t level, bint training):
        cdef Vocabulary temporary
        cdef Vocabulary* vocab
        cdef Vocabulary.iterator found
        cdef Signature signature
        cdef vector[Label] output
        cdef Py_ssize_t i, k
        cdef Label next_label
        if level < 0:
            raise ValueError('level must be nonnegative')
        if training:
            if level != self.levels.size():
                raise ValueError('fit levels must be sequential')
            self.levels.push_back(temporary)
            vocab = &self.levels[level]
        else:
            if level >= self.levels.size():
                raise ValueError('unfitted level')
            # Unseen test signatures get temporary IDs above all fitted IDs.
            # Never mutate the training vocabulary across transform calls.
            temporary = self.levels[level]
            vocab = &temporary
        next_label = deref(vocab).size()
        output.resize(batch.labels.size())
        for i in range(batch.labels.size()):
            signature.resize(batch.neighbors[i].size() + 1)
            signature[0] = batch.labels[i]
            for k in range(batch.neighbors[i].size()):
                signature[k+1] = batch.labels[batch.neighbors[i][k]]
            sort(signature.begin()+1, signature.end())
            found = deref(vocab).find(signature)
            if found == deref(vocab).end():
                deref(vocab)[signature] = next_label
                output[i] = next_label
                next_label += 1
            else:
                output[i] = deref(found).second
        batch.labels.swap(output)

    def __reduce__(self):
        cdef Py_ssize_t i
        cdef Vocabulary.iterator it
        state = []
        for i in range(self.levels.size()):
            entries = []
            it = self.levels[i].begin()
            while it != self.levels[i].end():
                entries.append((list(deref(it).first), deref(it).second))
                preincrement(it)
            state.append(entries)
        return (_restore_relabeler, (state,))


def _restore_relabeler(state):
    cdef Relabeler obj = Relabeler()
    cdef Vocabulary vocab
    cdef Signature signature
    for entries in state:
        vocab.clear()
        for key, value in entries:
            signature = key
            vocab[signature] = value
        obj.levels.push_back(vocab)
    return obj

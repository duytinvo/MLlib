# -*- coding: utf-8 -*-
"""
Created on 2020-03-10
@author: duytinvo
"""
import numpy as np
from mlmodels.utils.special_tokens import PAD, SOT, EOT, UNK


class Embeddings:
    def __init__(self, fname, limit=False):
        #     pass
        # @staticmethod
        # def load_embs(fname, use_small=False):
        self.word_vecs = dict()
        self.wsize = 0
        self.vlen = 0
        c = 0
        print("Read the embedding file")
        with open(fname, 'r') as f:
            for line in f:
                p = line.strip().split()
                if len(p) == 2:
                    self.vlen = int(p[0])  # Vocabulary
                    self.wsize = int(p[1])  # embeddings size
                else:
                    try:
                        w = "".join(p[0])
                        e = [float(i) for i in p[1:]]
                        self.word_vecs[w] = np.array(e, dtype="float32")
                        c += 1
                    except:
                        continue
                if c >= limit > 0:
                    break
        #        assert len(embs) == V
        # return embs

    # @staticmethod
    def get_embmtx(self, wsize=300, vocab=set(), scale=0.25):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        if self.wsize > 0 and self.wsize != wsize:
            print("Unmatching emb size")
            return None, None
        else:
            # print("Extracting pretrained embeddings:")
            # word_vecs = Embeddings.load_embs(emb_file, use_small)
            word_len = len(vocab.union(set(self.word_vecs.keys()))) + 4
            print('\t%d pre-trained word embeddings' % (len(self.word_vecs)))
            W = np.zeros(shape=(word_len, wsize), dtype="float32")
            w2i = {}
            for k in [PAD, SOT, EOT, UNK]:
                w2i[k] = len(w2i)
            for word, emb in self.word_vecs.items():
                w2i[word] = len(w2i)
                W[w2i[word]] = emb
            if len(vocab) != 0:
                for word in vocab.difference(set(self.word_vecs.keys())):
                    w2i[word] = len(w2i)
                    W[w2i[word]] = np.asarray(np.random.uniform(-scale, scale, (1, wsize)), dtype="float32")
            return W, w2i

    # @staticmethod
    def get_W(self, wsize, vocabx, scale=0.25):
        """
        Get word matrix. W[i] is the vector for word indexed by i
        """
        if self.wsize > 0 and self.wsize != wsize:
            print("Unmatching emb size")
            return None
        else:
            # print("\t- Extracting pretrained embeddings:")
            # word_vecs = Embeddings.load_embs(emb_file, use_small)
            print('\t\t- %d pre-trained word embeddings' % (len(self.word_vecs)))
            print('\t- Mapping to vocabulary:')
            unk = 0
            part = 0
            W = np.zeros(shape=(len(vocabx), wsize), dtype="float32")
            for word, idx in vocabx.items():
                if idx == 0:
                    continue
                if self.word_vecs.get(word) is not None:
                    W[idx] = self.word_vecs.get(word)
                else:
                    if self.word_vecs.get(word.lower()) is not None:
                        W[idx] = self.word_vecs.get(word.lower())
                        part += 1
                    else:
                        unk += 1
                        rvector = np.asarray(np.random.uniform(-scale, scale, (1, wsize)), dtype="float32")
                        W[idx] = rvector
            print('\t\t- %d randomly word vectors;' % unk)
            print('\t\t- %d partially word vectors;' % part)
            print('\t\t- %d pre-trained embeddings.' % (len(vocabx) - unk - part))
            return W

    @staticmethod
    def init_W(wsize, vocabx, scale=0.25):
        """
        Randomly initial word vectors between [-scale, scale]
        """
        W = np.zeros(shape=(len(vocabx), wsize), dtype="float32")
        for word, idx in vocabx.iteritems():
            if idx == 0:
                continue
            rvector = np.asarray(np.random.uniform(-scale, scale, (1, wsize)), dtype="float32")
            W[idx] = rvector
        return W
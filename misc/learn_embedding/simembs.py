import argparse
import numpy as np


def generate(emb_file):
    with open(emb_file, 'r') as f:
        vectors = {}
        words = []
        for line in f:
            vals = line.rstrip().split(' ')
            if len(vals) == 2:
                continue
            vectors[vals[0]] = [float(x) for x in vals[1:]]
            words.append(vals[0])
            
    vocab_size = len(words)
    vocab = {w: idx for idx, w in enumerate(words)}
    ivocab = {idx: w for idx, w in enumerate(words)}

    vector_dim = len(vectors[ivocab[0]])
    W = np.zeros((vocab_size, vector_dim))
    for word, v in vectors.items():
        if word == '<unk>':
            continue
        W[vocab[word], :] = v

    # normalize each word vector to unit variance
    W_norm = np.zeros(W.shape)
    d = (np.sum(W ** 2, 1) ** (0.5))
    W_norm = (W.T / d).T
    return (W_norm, vocab, ivocab)


def distance(W, vocab, ivocab, input_term, N=100):
    for idx, term in enumerate(input_term.split(' ')):
        if term in vocab:
            print('Word: %s  Position in vocabulary: %i' % (term, vocab[term]))
            if idx == 0:
                vec_result = np.copy(W[vocab[term], :])
            else:
                vec_result += W[vocab[term], :] 
        else:
            print('Word: %s  Out of dictionary!\n' % term)
            return

    vec_norm = np.zeros(vec_result.shape)
    d = (np.sum(vec_result ** 2,) ** (0.5))
    vec_norm = (vec_result.T / d).T

    dist = np.dot(W, vec_norm.T)

    for term in input_term.split(' '):
        if term in vocab:
            index = vocab[term]
            dist[index] = -np.Inf
        else:
            continue

    a = np.argsort(-dist)[:N]

    print("\n                               Word       Cosine distance\n")
    print("---------------------------------------------------------\n")
    for x in a:
        print("%35s\t\t%f\n" % (ivocab[x], dist[x]))


if __name__ == "__main__":
    """
    python simembs.py --emb_file ./results/w2v_yelp100.vec
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_file', default='results/w2v_yelp100.vec.txt', type=str)
    parser.add_argument('--N', default=10, type=int)
    args = parser.parse_args()
    
    W, vocab, ivocab = generate(args.emb_file)
    while True:
        input_term = input("\nEnter word or sentence (EXIT to break): ")
        if input_term == 'EXIT':
            break
        else:
            distance(W, vocab, ivocab, input_term, args.N)


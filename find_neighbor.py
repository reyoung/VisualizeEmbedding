# coding=utf-8
"""
find_neighbor.py

Usage:
  find_neighbor.py (-h | --help)
  find_neighbor.py [--word=<word>] [--dict=<dict_fn>] [--limit=<l>] [--emb=<emb_fn>] [--distance=<d>]

Options:
  -h --help         Show this screen.
  --word=<word>     Words to look up [default: 美国].
  --dict=<dict_fn>  Dictionary file [default: word_dict.txt].
  --limit=<l>       Limit value [default: 10].
  --emb=<emb_fn>    Embedding file [default: emb.npy].
  --distance=<d>    Distance method, cos or eucl [default: cos].
"""
import numpy
import math
import cPickle
import docopt


def cosine_similarity(v1, v2):
    """compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"""
    upper = numpy.dot(v1, v2)
    return 1 - upper / (numpy.linalg.norm(v1) * numpy.linalg.norm(v2))


def euclidean_similarity(v1, v2):
    """compute euclidean similarity of v1 to v2: \sqrt{\sum{(v1-v2)^2}}"""
    return math.sqrt(numpy.sum((v1 - v2) ** 2))


def main(word, dict_filename, distance, limit, embedding_filename):
    word_dict = dict()
    lookup_id = None

    with open(dict_filename, 'r') as f:
        word_dict_reversed = cPickle.load(f)
        for k in word_dict_reversed:
            cur_id = word_dict_reversed[k]
            word_dict[cur_id] = k
            if k == word:
                lookup_id = cur_id

    print 'Look up word is %s\t%d\n' % (word_dict[lookup_id].encode('utf-8'), lookup_id)

    emb = numpy.load(embedding_filename)['emb']

    row = emb[lookup_id]

    scores = []

    for i in xrange(len(emb)):
        if i == lookup_id: continue
        target_row = emb[i]
        score = distance(row, target_row)
        scores.append((score, i))

    scores.sort(key=lambda x: x[0])

    print 'Similar words are:'
    for i in xrange(limit):
        print '\t', scores[i][0], '\t', word_dict[scores[i][1]].encode('utf-8'), '\t', scores[i][1]


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    main(word=args['--word'].decode('utf-8'),
         distance=cosine_similarity if args[
                                           '--distance'] == 'cos' else euclidean_similarity,
         embedding_filename=args['--emb'],
         limit=int(args['--limit']),
         dict_filename=args['--dict'])

# coding:utf-8
import sys
import numpy
import cPickle
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

WORDS = [
    u"律师", u"证人", u"大律师", u"银行家", u"合伙人", u"当事人", u"会计师", u"妓女", u"雇主",
    u"法学家", u"球员", u"代理人", u"警官", u"商人", u"球星", u"检察官", u"情妇", u"陪审团", u"魔术师",
    u"经纪人", u"例子",
]

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

model = numpy.load(open(sys.argv[1]))
emb = model['emb']

dic = cPickle.load(open(sys.argv[2]))

emb_out = numpy.ones(shape=(len(WORDS), emb.shape[1]))

for i, w in enumerate(WORDS):
    emb_out[i] = emb[dic[w]]

emb = emb_out

dic_array = WORDS

tsne = TSNE(n_components=2, verbose=True)
pca = PCA(n_components=2)
low_dim_emb = pca.fit_transform(emb)

print 'Transform complete'
plt.scatter(low_dim_emb[:, 0], low_dim_emb[:, 1])
for label, x, y in zip(dic_array, low_dim_emb[:, 0], low_dim_emb[:, 1]):
    plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
plt.show()

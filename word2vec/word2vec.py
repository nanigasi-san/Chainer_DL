import numpy as np
import chainer
from chainer import cuda,Function,gradient_check,Variable,optimizers,serializers,utils
from chainer import Link,Chain,ChainList
import chainer.functions as F
import chainer.links as L
import collections

#コーパスを読み込んで単語にidをつける
index2word = {} #idから単語を取り出す
word2index = {} #単語からidを取り出す
counts = collections.Counter()
dataset = [] #idのリスト インデックスがコーパス内の単語の位置に対応している

#データセット製作
with open("data/ptb.train.txt") as f:
    for line in f:
        for word in line.split():
            if word not in word2index:
                ind = len(word2index)
                word2index[word] = ind
                index2word[ind] = word
            counts[word2index[word]] += 1
            dataset.append(word2index[word])

n_vocab = len(word2index)
datasize = len(dataset)

"""
print(dataset[100]) -> 78
print(index2word[dataset[100]]) -> more
"""

#サンプル生成器
cs = [counts[w] for w in range(len(counts))] #Counterは辞書の拡張型
power = np.float32(0.75)
p = np.array(cs,power.dtype)
sampler = utils.walker_alias.WalkerAlias(p)

#print(sampler.sample(5)) -> 五つのサンプルidが配列で帰る

#核となるクラス
class MyW2V(Chain):
    def __init__(self,n_vocab,n_units):
        super(MyW2V,self).__init__(
        embed = L.EmbedID(n_vocab,n_units),
        )
    
    def __call__(self,xb,yb,tb):
        xc = Variable(np.array(xb,dtype=np.int32))
        yc = Variable(np.array(yb,dtype=np.int32))
        tc = Variable(np.array(tb,dtype=np.int32))#tb=教師信号
        fv = self.fwd(xc,yc)
        return F.sigmoid_cross_entropy(fv,tc)
    
    def fwd(self,x,y):
        xv = self.embed(x)
        yv = self.embed(y)
        return F.sum(xv*yv,axis=1)#内積

#モデルと最適化アルゴリズムの設定
demb = 100 #分散表現の次元数
model = MyW2V(n_vocab,demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

#モデルに与える単語idのペアのバッチを作る
ws = 3 #Window Size
ngs = 5 #Negative Sample Size

def make_batch_set(dataset,ids):
    xb,yb,tb = [],[],[]
    for pos in ids:
        xid = dataset[pos]
        for i in range(1,ws):

            p = pos-i
            if p >= 0:
                xb.append(xid)
                yid = dataset[p]
                yb.append(yid)
                tb.append(1)
                for nid in sampler.sample(ngs):
                    xb.append(yid)
                    yb.append(nid)
                    tb.append(0)

            p = pos+i
            if p < datasize:
                xb.append(xid)
                yid = dataset[p]
                yb.append(yid)
                tb.append(1)
                for nid in sampler.sample(ngs):
                    xb.append(yid)
                    yb.append(nid)
                    tb.append(0)
    return [xb,yb,tb]

#パラメータの更新
bs = 100 #Batch Size
for epoch in range(15):
    print("epoch: {0}".format(epoch))
    indexes = np.random.permutation(datasize)
    for pos in range(0,datasize,bs):
        print(epoch,pos/datasize)
        ids = indexes[pos:(pos+bs) if (pos+bs)<datasize else datasize]
        xb,yb,tb = make_batch_set(dataset,ids)
        model.zerograds()
        loss = model(xb,yb,tb)
        loss.backward()
        optimizer.update()

#データの保存
with open("word2vec/model/myw2v.model","w") as f:
    f.write("{0} {1}\n".format(len(index2word),100))
    w = model.embed.W.data
    for i in range(w.shape[0]):
        v = " ".join([str(v) for v in w[i]])
        f.write("{0} {1}\n".format(index2word[i],v))

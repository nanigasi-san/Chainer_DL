import numpy as np
from chainer import Function,Variable,optimizers,serializers,utils,Link,Chain,ChainList
import chainer.functions as F
import chainer.links as L
import math

vocab = {}
def load_data(filename):
    global vocab
    words = open(filename).read().replace("\n","<eos>").strip().split()
    dataset = np.ndarray((len(words),),dtype=np.int32)
    for i,word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset
#
class MyRNN(Chain):
    def __init__(self,v,k):
        super(MyRNN,self).__init__(
            embed = L.EmbedID(v,k),
            H = L.Linear(k,k),
            W = L.Linear(k,v),
        )
    
    def __call__(self,s):# s=idのリスト
        accum_loss = None
        v,k = self.embed.W.data.shape
        h = Variable(np.zeros((1,k),dtype=np.float32))
        for i in range(len(s)):
            next_word_id = eos_id if (i == len(s)-1) else s[i+1]
            tx = Variable(np.array([next_word_id],dtype=np.int32))
            x_k = self.embed(Variable(np.array([s[i]],dtype=np.int32)))
            h = F.tanh(x_k + self.H(h))
            loss = F.softmax_cross_entropy(self.W(h),tx)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss

demb = 100
def cal_ps(model,s):
    h = Variable(np.zeros((1,demb),dtype=np.float32))
    sum = 0.0
    for i in range(1,len(s)):
        w1,w2 = s[i-1],s[i]
        x_k = model.embed(Variable(np.array([w1],dtype=np.int32)))
        h = F.tanh(x_k+model.H(h))
        yv = F.softmax(model.W(h))
        pi = yv.data[0][w2]
        sum -= math.log(pi,2)
    return sum

test_data = load_data("data/ptb.test.txt")[:1000]
max_id = 10000 #len(vocab)と同義
eos_id = 24 #調べた
model = MyRNN(max_id,demb)#len(vocab)は10000
model_path = "Language_model/model/myrnn-0.model"
serializers.load_npz(model_path,model)

sum = 0.0
wnum = 0
s = []
unknown_word = 0

for pos in range(len(test_data)):
    id = test_data[pos]
    s.append(id)
    if (id>max_id):
        unknown_word = 1

    if (id==eos_id):
        if (unknown_word!=1):
            ps = cal_ps(model,s)
            sum += ps
            wnum += len(s)-1
        else:
            unknown_word = 0
        s = []
print(math.pow(2,sum/wnum))

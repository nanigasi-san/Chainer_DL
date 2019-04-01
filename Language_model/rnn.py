# LSTMが入っている
import numpy as np
from chainer import Variable,optimizers,serializers,Chain
import chainer.functions as F
import chainer.links as L
from time import time
vocab = {}

# 単語にidをつける
def load_data(filename):
    global vocab
    words = open(filename).read().replace("\n","<eos>").strip().split()
    dataset = np.ndarray((len(words),),dtype=np.int32)
    for i,word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset

train_data = load_data("data/ptb.train.txt")
eos_id = vocab["<eos>"]

# モデルの定義
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


# モデルと最適化の設定
demb = 100
model = MyRNN(len(vocab),demb)
optimizer = optimizers.Adam()
optimizer.setup(model)

with open("Language_model/time.text","w"):
    pass
for epoch in range(5):
    start = time()
    s = []
    for pos in range(len(train_data)):
        id = train_data[pos]
        s.append(id)
        if id==eos_id:
            model.cleargrads()
            loss = model(s)
            loss.backward()
            optimizer.update()
            s = []
        if pos%1000==0:
            print("{0}% time:{1}".format(round(pos*1000/len(train_data),2),round(time()-start,3)))
    serializers.save_npz("model/myrnn-{0}.model".format(epoch),model)
    with open("Language_model/time.text","a") as f:
        f.write("epoch: {0} ,time: {1}s\n".format(epoch,round(time()-start),3))

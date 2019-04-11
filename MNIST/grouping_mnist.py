import numpy as np
import chainer
from chainer import cuda,Function,report,training,utils,Variable
from chainer import datasets,iterators,optimizers,serializers
from chainer import Link,Chain,ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
# import多くね

train,test = datasets.get_mnist(ndim=3) #ネットワーク接続が必要

class MnistModel(Chain):
    def __init__(self):
        super(MnistModel,self).__init__(
            l1 = L.Linear(784,100),
            l2 = L.Linear(100,100),
            l3 = L.Linear(100,10)
        )
        
    def __call__(self,x,t):
        return F.softmax_cross_entropy(self.fwd(x),t)
    
    def fwd(self,x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        return self.l3(h2)

model = MnistModel()
optimizer = optimizers.Adam()
optimizer.setup(model)

iterator = iterators.SerialIterator(train,1000)
updater = training.StandardUpdater(iterator,optimizer)
trainer = training.Trainer(updater,(10,"epoch"))
trainer.extend(extensions.ProgressBar())

trainer.run() #学習開始

serializers.save_npz("MNIST/model/mnist_model.model",model)
serializers.load_npz("MNIST/model/mnist_model.model",model)

#評価部分
ok = 0
for i in range(len(test)):
    x = Variable(np.array([test[i][0]],dtype=np.float32))
    t = test[i][1]
    out = model.fwd(x)
    ans = np.argmax(out.data)
    if (ans==t):
        ok += 1

print((ok/len(test))*100,"%")

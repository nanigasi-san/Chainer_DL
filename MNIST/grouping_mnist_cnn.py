import numpy as np
from chainer import training,Variable
from chainer import datasets,iterators,optimizers,serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions


train,test = datasets.get_mnist(ndim=3)

class MnistCNNModel(Chain):
    def __init__(self):
        super(MnistCNNModel,self).__init__(
            cn1 = L.Convolution2D(1,20,5),
            cn2 = L.Convolution2D(20,50,5),
            l1 = L.Linear(800,500),
            l2 = L.Linear(500,10),
        )

    def __call__(self,x,t):
        return F.softmax_cross_entropy(self.fwd(x),t)

    def fwd(self,x):
        h1 = F.max_pooling_2d(F.relu(self.cn1(x)),2)
        h2 = F.max_pooling_2d(F.relu(self.cn2(h1)),2)
        h3 = F.dropout(F.relu(self.l1(h2)))
        return self.l2(h3)

model = MnistCNNModel()
optimizer = optimizers.Adam()
optimizer.setup(model)

iterator = iterators.SerialIterator(train,1000)
updater = training.StandardUpdater(iterator,optimizer)
trainer = training.Trainer(updater,(10,"epoch"))
trainer.extend(extensions.ProgressBar())

trainer.run() #学習開始

serializers.save_npz("MNIST/model/mnist_cnn.model",model)#ここをコメントアウトすると前回の続きからになる
serializers.load_npz("MNIST/model/mnist_cnn.model",model)

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

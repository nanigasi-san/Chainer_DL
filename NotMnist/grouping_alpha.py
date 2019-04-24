import numpy as np
from chainer import training,Variable
from chainer import datasets,iterators,optimizers,serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from make_datasets import make_tupledata_set_train,make_tupledata_set_test
from time import time

def grouping_notmnist(size):
    start = time()
    train = make_tupledata_set_train(size=size)
    test = make_tupledata_set_test(size=1000)

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

    trainer.run() #学習開始

    serializers.save_npz("notMNIST/model/notmnist_nn.model",model)#ここをコメントアウトすると前回の続きからになる
    serializers.load_npz("notMNIST/model/notmnist_nn.model",model)

    #評価部分
    ok = 0
    for i in range(len(test)):
        x = Variable(np.array([test[i][0]],dtype=np.float32))
        t = test[i][1]
        out = model.fwd(x)
        ans = np.argmax(out.data)
        if (ans==t):
            ok += 1
    finish = time()
    print("train:",len(train))
    print("test: ",len(test))
    print((ok/len(test))*100,"%")
    print("time: ",int(finish-start),"s","\n")

grouping_notmnist(10)
grouping_notmnist(100)
grouping_notmnist(1000)
grouping_notmnist(10000)

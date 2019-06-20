import numpy as np
from chainer import training,Variable
from chainer import datasets,iterators,optimizers,serializers
from chainer import Chain
import chainer.functions as F
import chainer.links as L
from chainer.datasets import TupleDataset
from chainer.training import extensions
from PIL import Image
# import多くね

test = np.array(Image.open("MNIST/image/14x5.png"))
test = test.reshape(1,28,28)
test.shape
x = Variable(np.array(test,dtype=np.float32))
x /= 255
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


# serializers.save_npz("MNIST/model/mnist_nn.model",model)#ここをコメントアウトすると前回の続きからになる
serializers.load_npz("MNIST/model/mnist_nn.model",model)

out = model.fwd(x)
ans = np.argmax(out.data)

print(ans)
import numpy as np
from chainer import Variable,optimizers,Chain
import chainer.functions as F
import chainer.links as L

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data.astype(np.float32)
Y = iris.target
N = Y.size

Y2 = np.zeros(3*N).reshape(N,3).astype(np.float32)
for i in range(N):
    Y2[i,Y[i]] = 1.0
index = np.arange(N)
xtrain = X[index[index%2 != 0],:]
ytrain = Y2[index[index%2 != 0],:]
xtest = X[index[index%2 == 0],:]
yans = Y[index[index%2 == 0]]

class IrisLogi(Chain):
    def __init__(self):
        super(IrisLogi,self).__init__(
        l1 = L.Linear(4,3),
        )
    def __call__(self,x,y):
        return F.mean_squared_error(self.fwd(x),y)
    
    def fwd(self,x):
        return F.softmax(self.l1(x))

model = IrisLogi()
optimizer = optimizers.Adam()
optimizer.setup(model)

for i in range(10000):
    x = Variable(xtrain,)
    y = Variable(ytrain)
    model.zerograds()
    loss = model(x,y)
    loss.backward()
    optimizer.update()

xt = Variable(xtest)
yt = model.fwd(xt)
ans = yt.data
nrow,ncol = ans.shape
ok = 0

for i in range(nrow):
    cls = np.argmax(ans[i,:])
    if cls == yans[i]:
        ok += 1
print("optimizer:Adam")
print("{0}/{1} = {2}%".format(ok,nrow,(ok*1.0)/nrow))

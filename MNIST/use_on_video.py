import cv2
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
serializers.load_npz("MNIST/model/mnist_nn.model",model)

def aaa(img):
    test = np.array(img)
    test = test.reshape(1,28,28)
    x = Variable(np.array(test,dtype=np.float32))/255
    out = model.fwd(x)
    ans = np.argmax(out.data)
    return ans

import cv2
def capture_camera(mirror=True, size=None):
    """Capture video from camera"""
    # カメラをキャプチャする
    cap = cv2.VideoCapture(1) # 0はカメラのデバイス番号
    cascade = cv2.CascadeClassifier("face.xml")
    while True:
        # retは画像を取得成功フラグ
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray,scaleFactor=1.5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame, (x,y), (x+w,y+h),(0,0,255))

        # 鏡のように映るか否か
        if mirror is True:
            frame = frame[:,::-1]

        # フレームをリサイズ
        # sizeは例えば(800, 600)
        if size is not None and len(size) == 2:
            frame = cv2.resize(frame, size)

        # フレームを表示する
        cv2.imshow('camera capture', frame)

        k = cv2.waitKey(1) # 1msec待つ
        if k == 27: # ESCキーで終了
            break

    # キャプチャを解放する
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_camera(False)
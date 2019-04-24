from skimage import io
import glob
import numpy as np
from chainer.datasets import TupleDataset
from random import randint

#Tuple_Datasetを作る
def make_tupledata_set_train(size=100):
    alphabet_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    image_list = []
    answer_list = []
    def make_image_set():
        image_path_list = glob.glob("F://notMnist_large/{0}/*".format(alphabet))
        count = 0
        _dataset = []
        for image_path in image_path_list[:size+100]:
            try:
                _dataset.append(io.imread(image_path)/255)
                count += 1
            except:
                continue
            if count == size:
                break
        return _dataset

    def make_answer_set():
        return np.array( [alphabet_list.index(alphabet)] * size)

    for alphabet in alphabet_list[:10]:
        image_list.extend(make_image_set())
        answer_list.extend(make_answer_set())

    return TupleDataset(np.array(image_list,dtype=np.float32),np.array(answer_list))

def make_tupledata_set_test(size=10):
    alphabet_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    image_list = []
    answer_list = []
    def make_image_set():
        image_path_list = glob.glob("F://notMnist_large/{0}/*".format(alphabet))
        count = 0
        _dataset = []
        for i in range(size+50):
            try:
                _dataset.append(io.imread(image_path_list[randint(0,30000)])/255)
                count += 1
            except:
                continue
            if count == size:
                break
        return _dataset

    def make_answer_set():
        return np.array( [alphabet_list.index(alphabet)] * size)

    for alphabet in alphabet_list[:10]:
        image_list.extend(make_image_set())
        answer_list.extend(make_answer_set())

    return TupleDataset(np.array(image_list,dtype=np.float32),np.array(answer_list))

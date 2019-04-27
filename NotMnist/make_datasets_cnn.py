from skimage import io
import glob
import numpy as np
from chainer.datasets import TupleDataset
from random import randint

#Tuple_Datasetを作る
def make_image_set(size):
    image_set = []
    alphabet_list = list("ABCDEFGHIJ")
    for alpha in alphabet_list:
        image_path_list = glob.glob("F://notMnist_large/{0}/*".format(alpha))
        counter = 0
        for image_path in image_path_list[:size+100]:
            try:
                image = io.imread(image_path).reshape((1,28,28))
                image_for_0to1 = (image/255).astype(np.float32)
                image_set.append(image_for_0to1)
                ounter += 1
            except:
                continue
            if counter == size:
                break
    return np.array(image_set)

def make_answer_set(size):
    alphabet_list = list("ABCDEFGHIJ")
    answer_list = []
    for alpha in alphabet_list:
        for _ in range(size):
            answer_list.append(alphabet_list.index(alpha))
    return np.array(answer_list)

def make_tupledata_set(size):
    return TupleDataset(make_image_set(size),make_answer_set(size))

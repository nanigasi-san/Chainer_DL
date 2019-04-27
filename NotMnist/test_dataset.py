# In[]:
import sys
sys.path.append("NotMnist/")
from make_datasets_cnn import make_tupledata_set,make_image_set,make_answer_set
import numpy as np


# In[]:
data = make_image_set(5)
data.shape
data
answer = make_answer_set(5)
answer.shape
answer
# In[]:
train = make_tupledata_set(5)
train[45]

import tensorflow as tf
import numpy as np
import json

from input_new import make_input_fn
from base_model import base_model

MAX_HIST_LEN = 500

def train(filename,batch_size,num_epochs,model_dir):
    with open('dictionary.json') as f:
        dictionary=json.load(f)
        category=np.load('category_list.npy').tolist()
        item_len=len(dictionary['product_id'])
        assert item_len == len(category)

    train_input_fn=make_input_fn('train.tfrecord',MAX_HIST_LEN,batch_size,num_epochs)

    estimator=tf.estimator.Estimator(base_model,model_dir=model_dir,
    params={
        'item_len':item_len,
        'categories':category,
        'num_hidden':128,
        'learning_rate':0.001
    })

    estimator.train(train_input_fn)

if __name__=='__main__':
    train('train.tfrecord',32,3,'./model_log')  
import random
import numpy as np
import pandas as pd
#from sklearn.utils import shuffle
from collections import Counter
import pickle
import json
import tensorflow as tf

random.seed(1234)

with open('./reviews_Electronics_5.json') as f:
  a={}
  i=0
  for line in f:
    a[i]=eval(line)
    i+=1
  
  df=pd.DataFrame.from_dict(a,orient='index')

data_dict={}

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _int64_list_feature(elems):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=elems))

reviews_df=df[['reviewerID', 'asin','unixReviewTime']]
data_dict['reviewer_id']=reviews_df['reviewerID'].unique().tolist()
user_map=dict(zip(data_dict['reviewer_id'],range(len(data_dict['reviewer_id']))))
reviews_df['reviewerID']=reviews_df['reviewerID'].map(lambda x: user_map[x])
reviews_df = reviews_df.sort_values(['reviewerID','unixReviewTime'])



# maybe this is unnecessary
reviews_df=reviews_df.sort_values(['reviewerID'])

product_ids=reviews_df['asin']

product_id_list=sorted(product_ids.unique().tolist())

data_dict['product_id']=product_id_list

product_id_map=dict(zip(product_id_list,range(len(product_id_list))))

reviews_df['asin']=reviews_df['asin'].map(lambda x: product_id_map[x] )

with open('./meta_Electronics.json') as f:
  a={}
  i=0
  for line in f:
    a[i]=eval(line)
    i+=1
  meta_df=pd.DataFrame.from_dict(a,orient='index')
meta_df=meta_df[meta_df['asin'].isin(product_ids.unique())]
assert meta_df.shape[0] == product_ids.unique().shape[0]
meta_df=meta_df[['asin', 'categories']]
meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])
meta_df['asin']=meta_df['asin'].map(lambda x: product_id_map[x])
meta_df = meta_df.sort_values(['asin'])
data_dict['category']=meta_df['categories'].unique().tolist()
category_map=dict(zip(data_dict['category'],range(len(data_dict['category']))))
with open('dictionary.json', 'w') as f:
  json.dump(data_dict,f)

category_list=meta_df['categories'].map(lambda x: category_map[x]).as_matrix()
np.save('category_list.npy',category_list)

user_train=[]
product_train=[]
retrieved_train=[]
user_test=[]
product_test=[]
retrieved_test=[]
labels_train=[]
labels_test=[]

history=np.zeros([len(data_dict['reviewer_id']),500],np.int64)

for reviewerID, hist in reviews_df.groupby('reviewerID'):
  pos_list = hist['asin'].tolist()
  for index,elem in enumerate(pos_list):
    history[reviewerID][index]=elem
  list_len=len(pos_list)
  def gen_neg():
    neg = pos_list[0]
    while neg in pos_list:
      neg = random.randint(0, len(data_dict['product_id'])-1)
    return neg
  neg_list = [gen_neg() for i in range(list_len)]
  user_train+=[reviewerID]*((list_len-2)*2)
  user_test+=[reviewerID]*2
  product_train+=[i for i in range(1,list_len-1)]
  product_train+=[i for i in range(1,list_len-1)]
  retrieved_train+=[pos_list[i] for i in range(1,list_len-1)]
  retrieved_train+=[neg_list[i] for i in range(1,list_len-1)]
  product_test.append(list_len-1)
  product_test.append(list_len-1)
  retrieved_test+=[pos_list[-1],neg_list[-1]]

  labels_train+=[1]*(list_len-2)+[0]*(list_len-2)
  labels_test+=[1,0]

# product_train,retrieved_train,user_train,labels_train=shuffle(
#   product_train,retrieved_train,user_train,labels_train,random_state=2000)
# product_test,retrieved_test,user_test,labels_test=shuffle(
#   product_test,retrieved_test,user_test,labels_test,random_state=2000)

np.save('history.npy',history)

with tf.python_io.TFRecordWriter('train.tfrecord') as writer:
  for i in range(len(product_train)):
    temp_history=history[user_train[i]]
    for j in range(product_train[i],500):
      temp_history[j]=0
    example=tf.train.Example(
      features=tf.train.Features(
        feature={
          'user_id':_int64_feature(user_train[i]),
          'hist_len':_int64_feature(product_train[i]),
          'retrieved':_int64_feature(retrieved_train[i]),
          'label':_int64_feature(labels_train[i]),
          'hist':_int64_list_feature(temp_history)
        }
      )
    )
    writer.write(example.SerializeToString())

  with tf.python_io.TFRecordWriter('test.tfrecord') as writer:
    for i in range(len(product_test)):
      temp_history=history[user_test[i]]
      for j in range(product_test[i],500):
        temp_history[j]=0
      example=tf.train.Example(
        features=tf.train.Features(
          feature={
            'user_id':_int64_feature(user_test[i]),
            'hist_len':_int64_feature(product_test[i]),
            'retrieved':_int64_feature(retrieved_test[i]),
            'label':_int64_feature(labels_test[i]),
            'hist':_int64_list_feature(temp_history)
          }
        )
      )
      writer.write(example.SerializeToString())

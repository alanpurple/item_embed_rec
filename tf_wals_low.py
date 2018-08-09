import tensorflow as tf
import numpy as np
from mongoengine import connect
from scipy.sparse import coo_matrix
from tensorflow.contrib.factorization import WALSMatrixFactorization as wmf
from tensorflow.contrib.factorization import WALSModel
import json

from models import PosData

HISTORY_FROM='03-21'
HISTORY_TO='04-10'

data_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_sparse.json'

deal_dict=np.load('dict_'+HISTORY_FROM+'_'+HISTORY_TO+'_for_sparse.npy')
user_dict=np.load('user_'+HISTORY_FROM+'_'+HISTORY_TO+'.npy')

num_rows=len(user_dict)
num_cols=len(deal_dict)
print('number of users: ',num_rows)
print('number of items: ',num_cols)
dimension=30
n_iter=30

connect('wprec',host='mongodb://10.102.61.251:27017')

deal_finder=dict(zip(deal_dict,range(num_cols)))
goal_data=PosData.objects(TransDate='2018-04-11 21',WepickRank__gte=20).aggregate(
        *[{'$group':{'_id':'$DealId'}}],allowDiskUse=True)
goal_list=[elem['_id'] for elem in goal_data]
target=[]
for elem in goal_list:
    if elem in deal_dict:
        target.append(elem)

if __name__=='__main__':
    with open(data_path,'r') as f:
        data=json.load(f)

    indices=[]
    values=[]

    for idx,elem in enumerate(data):
        indices+=zip([idx]*len(elem),elem)
        values+=[1.0]*len(elem)
    with tf.Graph().as_default() as graph1:
        sp_mat=tf.SparseTensor(indices,values,[num_rows,num_cols])

        model=WALSModel(num_rows,num_cols,dimension,0.5,2.0,row_weights=None,col_weights=None)

        row_factors=model.row_factors[0]
        col_factors=model.col_factors[0]

        sess=tf.Session(graph=graph1)

        writer=tf.summary.FileWriter('walsmodels',graph1)

        row_update_op=model.update_row_factors(sp_mat)[1]
        col_update_op=model.update_col_factors(sp_mat)[1]
        
        sess.run(model.initialize_op)
        for _ in range(n_iter):
            sess.run(model.row_update_prep_gramian_op)
            sess.run(model.initialize_row_update_op)
            sess.run(row_update_op)
            sess.run(model.col_update_prep_gramian_op)
            sess.run(model.initialize_col_update_op)
            sess.run(col_update_op)

    output_row=row_factors.eval(sess)
    output_col=col_factors.eval(sess)

    sess.close()

    writer.close()

    print(output_row[16000])
    print(output_col[700])

    temp_users=user_dict[100000:100010]

    for i in range(100000,100010):
        print('===========================')
        print('user id: ',user_dict[i])
        for j in range(len(target)):
            print('***')
            print('item id: ',target[j])
            target_index=deal_finder[target[j]]
            print('score: ',sum(output_row[i][:]*output_col[target_index]))


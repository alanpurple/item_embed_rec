import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix
from tensorflow.contrib.factorization import WALSMatrixFactorization as wmf
from tensorflow.contrib.factorization import WALSModel
import json

HISTORY_FROM='04-01'
HISTORY_TO='04-10'
DEAL_TO='04-11'

data_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_sparse.json'

deal_dict=np.load('dict_'+HISTORY_FROM+'_'+DEAL_TO+'_for_sparse.npy')
user_dict=np.load('user_'+HISTORY_FROM+'_'+DEAL_TO+'.npy')

num_rows=len(user_dict)
num_cols=len(deal_dict)
dimension=30
n_iter=30

if __name__=='__main__':
    with open(data_path,'r') as f:
        data=json.load(f)

    #row_idx=[]
    #column_idx=[]
    #values=[]

    #for idx,elem in enumerate(data):
    #    row_idx+=[idx]*len(elem)
    #    column_idx+=elem
    #    values+=[1.0]*len(elem)

    #data_sparse=coo_matrix((values,(row_idx,column_idx)),(num_rows,num_cols))

    indices=[]
    values=[]

    for idx,elem in enumerate(data):
        indices+=zip([idx]*len(elem),elem)
        values+=[1.0]*len(elem)
    with tf.Graph().as_default() as graph1:
        sp_mat=tf.SparseTensor(indices,values,[num_rows,num_cols])

        model=WALSModel(num_rows,num_cols,dimension,0.1,2.0,row_weights=None,col_weights=None)

        row_factors=model.row_factors[0]
        col_factors=model.col_factors[0]

        sess=tf.Session(graph=graph1)

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

    print(output_row[16000])
    print(output_col[700])
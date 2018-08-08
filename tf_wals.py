import tensorflow as tf
import numpy as np
from scipy.sparse import coo_matrix
from tensorflow.contrib.factorization import WALSMatrixFactorization as wmf
import json

HISTORY_FROM='04-01'
HISTORY_TO='04-10'
DEAL_TO='04-11'

data_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_sparse.json'

deal_dict=np.load('dict_'+HISTORY_FROM+'_'+DEAL_TO+'_for_sparse.npy')
user_dict=np.load('user_'+HISTORY_FROM+'_'+DEAL_TO+'.npy')

num_rows=len(user_dict)
num_cols=len(deal_dict)
dimension=300

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
    indices_t=[]
    values=[]

    for idx,elem in enumerate(data):
        indices+=zip([idx]*len(elem),elem)
        indices_t+=zip(elem,[idx]*len(elem))
        values+=[1.0]*len(elem)

    def sparse_input():
        sp_mat=tf.SparseTensor(indices,values,[num_rows,num_cols])
        sp_mat_t=tf.SparseTensor(indices_t,values,[num_rows,num_cols])
        return {'input_rows':sp_mat,'input_cols':sp_mat_t},None


    estimator=wmf(num_rows,num_cols,dimension,regularization_coeff=9.8,model_dir='./walsmodels',max_sweeps=30)

    estimator.fit(input_fn=sparse_input)
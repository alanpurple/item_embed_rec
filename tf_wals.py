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
        mat_slices=tf.data.Dataset.from_tensor_slices(sp_mat)
        mat_t_slices=tf.data.Dataset.from_tensor_slices(sp_mat_t)
        iter1=mat_slices.make_one_shot_iterator()
        iter2=mat_t_slices.make_one_shot_iterator()
        return {'input_rows':iter1.next(),'input_cols':iter2.next()},None


    estimator=wmf(num_rows,num_cols,dimension,0.01,9.8,model_dir='./walsmodels')

    estimator.fit(input_fn=sparse_input)
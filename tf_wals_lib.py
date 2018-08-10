import tensorflow as tf
import numpy as np
from mongoengine import connect
from scipy.sparse import coo_matrix
from tensorflow.contrib.factorization import WALSMatrixFactorization as wmf
from tensorflow.contrib.factorization import WALSModel
import json

from models import PosData
from models import WepickDeal
from models import DealW2v


def wals(id,from_date,to_date,predict_moment,dimension=30,weight=0.5,coef=2.0,n_iter=30):

    data_path='wp_'+from_date+'_'+to_date+'_sparse.json'

    deal_dict=np.load('dict_'+from_date+'_'+to_date+'_for_sparse.npy')
    user_dict=np.load('user_'+from_date+'_'+to_date+'.npy')

    if id not in user_dict:
        return -1
    else:
        user_index=np.where(user_dict==id)[0][0]

    num_rows=len(user_dict)
    num_cols=len(deal_dict)

    connect('wprec',host='mongodb://10.102.61.251:27017')

    deals=WepickDeal.objects(pk__gte=predict_moment+' 20',pk__lte=predict_moment+' 99')
    deal_slots=[]
    deal_ids=[]
    predict_input=[]
    for elem in deals:
        dealid=elem['deal'].id
        if dealid in deal_dict:
            deal_slots.append(int(elem.id[-2:]))
            deal_ids.append(elem['deal'].id)

    deal_finder=dict(zip(deal_dict,range(num_cols)))

    with open(data_path,'r') as f:
        data=json.load(f)

    indices=[]
    values=[]

    for idx,elem in enumerate(data):
        indices+=zip([idx]*len(elem),elem)
        values+=[1.0]*len(elem)
    with tf.Graph().as_default() as graph1:
        sp_mat=tf.SparseTensor(indices,values,[num_rows,num_cols])

        model=WALSModel(num_rows,num_cols,dimension,weight,coef,row_weights=None,col_weights=None)

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

    results=[]

    for i in range(len(deal_ids)):
        deal_index=deal_finder[deal_ids[i]]
        results.append({'id':deal_ids[i],'slot':deal_slots[i],'score':sum(output_row[user_index][:]*output_col[deal_index])})
    return results


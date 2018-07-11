import tensorflow as tf
import numpy as np
import pickle

from mongoengine import connect
from models import DealW2v

HISTORY_FROM='04-01'
HISTORY_TO='04-10'

train_data_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_seq.pkl'

deal_list=np.load('dict_'+HISTORY_FROM+'_'+HISTORY_TO+'.npy')

connect('wprecdb',host='mongodb://10.102.50.46:27017')

deal_dict=np.array([[0.0]*100]+[DealW2v.objects(pk=elem).first().vectorizedWords for elem in deal_list[1:]])

def make_input_nda(data):
    train_data=[]
    train_lens=[]
    train_labels=[]
    test_data=[]
    test_lens=[]
    test_labels=[]
    # per user
    for pair in data:
        hist_len=len(pair[0])
        assert hist_len<40
        hist=pair[0]
        neg=pair[1]
        # make 10 data per user( 5 pos, 5 neg )
        for i in range(hist_len-6,hist_len-1):
            # max length of 38 ( 37 history and 1 output )
            temp=hist[:i]+[0]*(38-i)
            train_data.append(temp)
            temp_neg=hist[:i-1]+[neg[i-1]]+[0]*(38-i)
            train_data.append(temp_neg)
            train_labels+=[1,0]
            train_lens+=[i,i]
        # history can have length 39
        if hist_len>38:
            temp=hist[-38:]
            temp_neg=temp[:37]+[neg[-1]]
            test_lens+=[38,38]
        else:
            temp=hist+[0]*(38-hist_len)
            temp_neg=hist[:-1]+[neg[-1]]+[0]*(38-hist_len)
            test_lens+=[hist_len,hist_len]
        test_data.append(temp)
        test_data.append(temp_neg)
        test_labels+=[1,0]
    return {
        'seq':np.array(train_data),'seq_len':np.array(train_lens)
        },np.array(train_labels),{
            'seq':np.array(test_data),'seq_len':np.array(test_lens)
            },np.array(test_labels)

def wp_rnn_classifier_fn(features,labels,mode,params):
    seq_len=features['seq_len']
    input_seq=features['seq']
    if mode!=tf.estimator.ModeKeys.PREDICT:
        deal_emb=params['dict']
        input_emb=tf.nn.embedding_lookup(deal_emb,input_seq)
    else:
        input_emb=input_seq
    rnn_depth=params['rnn_depth']
    if rnn_depth==1:
        cell=tf.nn.rnn_cell.GRUCell(100)
        if params['use_dropout']:
            cell=tf.nn.rnn_cell.DropoutWrapper(cell,params['dropout_input_keep'],params['dropout_output_keep'])
    else:
        cell=[tf.nn.rnn_cell.GRUCell(100) for _ in range(rnn_depth)]
        if params['use_dropout']:
            cell=[tf.nn.rnn_cell.DropoutWrapper(elem,params['dropout_input_keep'],params['dropout_output_keep']) for elem in cell]
            cell=tf.nn.rnn_cell.MultiRNNCell(cell)
    _,state=tf.nn.dynamic_rnn(cell,input_emb,seq_len,dtype=tf.float64)
    if rnn_depth!=1:
        state=state[-1]

    logits=tf.layers.dense(state,2)
    predicted=tf.argmax(logits,1)
    prob=tf.sigmoid(logits[:,1])
    if mode==tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            prediction={
                'class':predicted,
                # probability for 1
                'prob':prob
                })
    loss=tf.losses.softmax_cross_entropy(tf.one_hot(labels,depth=2),logits)
    if mode==tf.estimator.ModeKeys.TRAIN:
        optimizer=tf.train.AdamOptimizer()
        train_op=optimizer.minimize(loss,tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)
    eval_metrics={
        'accuracy':tf.metrics.accuracy(labels,predicted),
        'auc':tf.metrics.auc(tf.ones_like(prob),prob)
        }
    return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=eval_metrics)


if __name__ == '__main__':
    with open(train_data_path,'rb') as f:
        data=pickle.load(f)
    train_x,train_y,test_x,test_y=make_input_nda(data)

    train_input_fn=tf.estimator.inputs.numpy_input_fn(train_x,train_y,32,5,True,10000,4)
    test_input_fn=tf.estimator.inputs.numpy_input_fn(test_x,test_y,4,1,False)

    wp_rnn_classifier=tf.estimator.Estimator(wp_rnn_classifier_fn,'./seq_models',
                                             params={
                                                 'dict':deal_dict,
                                                 'rnn_depth':3,
                                                 'use_dropout':True,
                                                 'dropout_input_keep':0.9,
                                                 'dropout_output_keep':0.9
                                                 })

    train_spec=tf.estimator.TrainSpec(input_fn=train_input_fn,max_steps=10000)
    eval_spec=tf.estimator.EvalSpec(input_fn=test_input_fn)

    tf.estimator.train_and_evaluate(wp_rnn_classifier,train_spec,eval_spec)
import tensorflow as tf
import pickle
import json
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from mongoengine import connect

from models import DealW2v
from models import PosData
from models import WepickDeal

from wp_rnn_37 import wp_rnn_classifier_fn

import wprecservice_pb2
import wprecservice_pb2_grpc

connect('wprec',host='mongodb://10.102.61.251:27017')
# this should be set to user input, this values are temporary
HISTORY_FROM='04-01'
HISTORY_TO='04-10'
PREDICT_DATE='04-11'

class WpRecService(wprecservice_pb2_grpc.WpRecServiceServicer):
    def GetRecommend(self, request, context):
        if request.methodName=='' or request.dayFrom=='' or request.dayTo=='' or request.predictMoment=='':
            return wprecservice_pb2.RecommendResponse(error=0)

        profile_data_path='profile_'+request.dayFrom+'_'+request.dayTo+'.csv'
        data_seq_path='wp_'+request.dayFrom+'_'+request.dayTo+'_seq.json'
        with open(data_seq_path,'r') as f:
            seq_data=json.load(f)
        for elem in seq_data:
            if elem['id']==request.user:
                if len(elem['pos'])>36:
                    user_seq=elem['pos'][-37:]
                else:
                    user_seq=elem['pos']
                break

        profile_df=pd.read_csv(profile_data_path,index_col=0)
        user_profile=profile_df.loc[request.user].tolist()
        deal_list=np.load('dict_'+request.dayFrom+'_'+request.predictMoment[5:-3]+'.npy')
        deals=WepickDeal.objects(pk__gte=request.predictMoment+' 20',pk__lte=request.predictMoment+' 99')
        scaler=joblib.load('scaler.pkl')
        deal_slots=[]
        deal_ids=[]
        predict_input=[]
        predict_seq_input=[]
        for elem in deals:
            deal=DealW2v.objects(pk=elem['deal'].id).first()
            if deal!=None:
                deal_vec=deal.vectorizedWords
                predict_input.append(user_profile+deal_vec)
                deal_slots.append(int(elem.id[-2:]))
                deal_ids.append(elem['deal'].id)
                predict_seq_input.append(user_seq+[elem['deal'].id]+[0]*(37-len(user_seq)))
        predict_input=scaler.transform(predict_input)
        predict_seq_lens=[len(user_seq)+1]*len(predict_seq_input)

        if request.methodName=='dnn_tf':
            pass
        elif request.methodName=='alibaba_din':
            pass
        elif request.methodName=='gbc':
            gbc=joblib.load('wpgbc.pkl')
            probs=gbc.predict_proba(predict_input)[:,1]

        elif request.methodName=='logistic':
            lr=joblib.load('wplr.pkl')
            probs=lr.predict_proba(predict_input)[:,1]

        elif request.methodName=='rnn':
            deal_dict=np.array([[0.0]*100]+[DealW2v.objects(pk=elem).first().vectorizedWords for elem in deal_list[1:]])
            predict_input_fn=tf.estimator.inputs.numpy_input_fn({'seq':np.array(predict_seq_input),'seq_len':np.array(predict_seq_lens)},shuffle=False)
            rnn_predictor=tf.estimator.Estimator(wp_rnn_classifier_fn,'./seq_models',
                                             params={
                                                 'dict':deal_dict,
                                                 'rnn_depth':3,
                                                 'use_dropout':True,
                                                 'dropout_input_keep':0.9,
                                                 'dropout_output_keep':0.9
                                                 })
            result=rnn_predictor.predict(predict_input_fn)
            probs=[elem['prob'] for elem in result]

        elif request.methodName=='logistic_tf':
            pass
        elif request.methodName=='boosted_tree_tf':
            pass

        results=[{'id':deal_ids[i],'slot':deal_slots[i],'score':probs[i]} for i in range(len(predict_input))]
        return wprecservice_pb2.RecommendResponse(error=-1,result=results)

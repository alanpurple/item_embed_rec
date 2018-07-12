import tensorflow as tf
import pickle
import pandas as pd
from sklearn.externals import joblib
from mongoengine import connect

from models import DealW2v
from models import PosData
from models import WepickDeal

import wprecservice_pb2
import wprecservice_pb2_grpc

connect('wepickw2v',host='mongodb://localhost')
# this should be set to user input, this values are temporary
HISTORY_FROM='04-01'
HISTORY_TO='04-10'

class WpRecService(wprecservice_pb2_grpc.WpRecServiceServicer):
    def GetRecommend(self, request, context):
        if request.methodName=='' or request.dayFrom=='' or request.dayTo=='' or request.predictMoment=='':
            return wprecservice_pb2.RecommendResponse(error=0)

        profile_data_path='profile_'+request.dayFrom+'_'+request.dayTo+'.csv'

        profile_df=pd.read_csv(profile_data_path,index_col=0)
        user_profile=profile_df.loc[request.user].tolist()
        deals=WepickDeal.objects(pk__gte=request.predictMoment+' 20',pk__lte=request.predictMoment+' 99')
        scaler=joblib.load('scaler.pkl')
        deal_slots=[]
        deal_ids=[]
        predict_input=[]
        for elem in deals:
            deal=DealW2v.objects(pk=elem['deal'].id).first()
            if deal!=None:
                deal_vec=deal.vectorizedWords
                predict_input.append(user_profile+deal_vec)
                deal_slots.append(int(elem.id[-2:]))
                deal_ids.append(elem['deal'].id)
        predict_input=scaler.transform(predict_input)

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

        #elif request.methodName=='rnn':
        #    rnn_predictor=tf.contrib.predictor.from_saved_model('./seq_models')
        #    rnn_predictor.predict()

        elif request.methodName=='logistic_tf':
            pass
        elif request.methodName=='boosted_tree_tf':
            pass

        results=[{'id':deal_ids[i],'slot':deal_slots[i],'score':probs[i]} for i in range(len(predict_input))]
        return wprecservice_pb2.RecommendResponse(error=-1,result=results)

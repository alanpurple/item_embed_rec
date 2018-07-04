import pickle
import pandas as pd
from sklearn.externals import joblib
from mongoengine import connect

from models import DealW2v
from models import PosData

HISTORY_FROM='04-01'
HISTORY_TO='04-10'

connect('wepickw2v',host='mongodb://localhost')

profile_data_path='profile_'+HISTORY_FROM+'_'+HISTORY_TO+'.csv'

profile_df=pd.read_csv(profile_data_path,index_col=0)

scaler=joblib.load('scaler.pkl')

goal_data=PosData.objects(TransDate='2018-04-11 21',WepickRank__gte=20).aggregate(
        *[{'$group':{'_id':'$DealId'}}],allowDiskUse=True)
goal_list=[elem['_id'] for elem in goal_data]

goal_list_final=[]
goal_vec=[]

for id in goal_list:
    deal=DealW2v.objects(pk=id).first()
    if deal != None:
        deal_vec=deal.vectorizedWords
        if len(deal_vec)==100:
            if deal_vec[0]!=0 and deal_vec[1]!=0 and deal_vec[2]!=0:
                goal_list_final.append(id)
                goal_vec.append(deal_vec)

scaler=joblib.load('scaler.pkl')
wplr=joblib.load('wplr.pkl')
gbc=joblib.load('wpgbc.pkl')

sample_user=profile_df.iloc[8001:8030]

user_ids=sample_user.index.tolist()

userdata=PosData.objects(UserId__in=user_ids,TransDate__gte='2018-'+HISTORY_FROM+' 00',TransDate__lte='2018-'+HISTORY_TO+' 23',WepickRank__gte=20,WepickRank__lte=55).aggregate(
        *[{'$group':{'_id':'$UserId','docs':{'$push':'$DealId'}}}],allowDiskUse=True)

histdata=[]
hist_columns=['hist_{}'.format(i+1) for i in range(40)]
for user in userdata:
    temp=[0]*40
    i=0
    for elem in user['docs']:
        deal=DealW2v.objects(pk=elem).first()
        if deal != None:
            deal_vec=deal.vectorizedWords
            if len(deal_vec)==100:
                if deal_vec[0]!=0 and deal_vec[1]!=0 and deal_vec[2]!=0:
                    temp[i]=elem
                    i+=1
    histdata.append(temp)

hist_df=pd.DataFrame(histdata,index=user_ids,columns=hist_columns)

hist_df.to_csv('random_user_history.csv')

# user profiles X  goal_vec s

lr_results=[]
gbc_results=[]
for user,profile in sample_user.iterrows():
    input=[]
    for vec in goal_vec:
        input.append(profile.tolist()+vec)
    scaled=scaler.transform(input)
    lr_results.append(wplr.predict_proba(scaled)[:,1])
    gbc_results.append(gbc.predict_proba(scaled)[:,1])

lr_df=pd.DataFrame(lr_results,user_ids,goal_list_final)
gbc_df=pd.DataFrame(gbc_results,user_ids,goal_list_final)

lr_df.to_csv('lr_review_0411_21.csv')
gbc_df.to_csv('gbc_review_0411_21.csv')
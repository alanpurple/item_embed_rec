from sklearn.externals import joblib
from mongoengine import connect
from mongoengine.errors import DoesNotExist
import pandas as pd
from os import path

from models import DealW2v
from models import PosData
from models import WepickDeal

HISTORY_FROM='04-01'
HISTORY_TO='04-10'

connect('wepickw2v',host='mongodb://localhost')

profile_data_path='profile_'+HISTORY_FROM+'_'+HISTORY_TO+'.csv'
if path.exists(profile_data_path):
    profile_df=pd.read_csv(profile_data_path,index_col=0)
else:
    wepickdata=PosData.objects(TransDate__gte='2018-'+HISTORY_FROM+' 00',TransDate__lte='2018-'+HISTORY_TO+' 23',WepickRank__gte=20,WepickRank__lte=55).aggregate(
        *[{'$group':{'_id':'$UserId','docs':{'$push':'$$ROOT'}}}],allowDiskUse=True)
    # in case cursornotfounderrror caused by very long sampling times
    wepickdata=list(wepickdata)
    feature_columns=['feature_'+str(i+1) for i in range(100)]
    data=[]
    user_ids=[]

    print('Number of Users: ',len(wepickdata))
    
    for elem in wepickdata:
        if len(elem['docs'])>20 and len(elem['docs'])<40:
            for doc in elem['docs']:
                hist_sum=[0]*100
                result=DealW2v.objects(pk=doc['DealId']).first()
                if result != None:
                    temp=result.vectorizedWords['values']
                    if len(temp)==100:
                        if temp[0]!=0 and temp[1]!=0 and temp[2]!=0:
                            hist_sum=[sum(x) for x in zip(hist_sum,temp)]
            if hist_sum[0]!=0 and hist_sum[1]!=0 and hist_sum[2]!=0:
                user_ids.append(elem['_id'])
                data.append(hist_sum)

    assert len(user_ids)==len(data)

    print('resulting number of users: ',len(data)-1)

    profile_df=pd.DataFrame(data,columns=feature_columns,index=user_ids)
    profile_df.to_csv(profile_data_path)

scaler=joblib.load('scaler.pkl')
wplr=joblib.load('wplr.pkl')
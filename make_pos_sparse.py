import numpy as np
from mongoengine import connect
import json

from models import DealW2v
from models import PosData
from models import WepickDeal

HISTORY_FROM='04-01'
HISTORY_TO='04-10'
DEAL_TO='04-11'

data_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_sparse.json'

connect('wprec',host='mongodb://10.102.61.251:27017')

wepickdata=PosData.objects(TransDate__gte='2018-'+HISTORY_FROM+' 00',TransDate__lte='2018-'+HISTORY_TO+' 23',WepickRank__gte=20,WepickRank__lte=65).aggregate(
    *[{'$group':{'_id':'$UserId','docs':{'$push':'$$ROOT'}}}],allowDiskUse=True)
# in case cursornotfounderrror caused by very long sampling times
wepickdata=list(wepickdata)
data=[]

user_dict=[]
deal_dict=[]


for elem in wepickdata:
    if len(elem['docs'])>20 and len(elem['docs'])<60:
        user_dict.append(elem['_id'])
        hist=[]
        for doc in elem['docs']:
            pos_id=doc['DealId']
            result=DealW2v.objects(pk=pos_id).first()
            if result != None:
                if pos_id in deal_dict:
                    hist.append(deal_dict.index(pos_id))
                else:
                    hist.append(len(deal_dict))
                    deal_dict.append(pos_id)
                
        data.append(hist)

goal_data=PosData.objects(TransDate='2018-04-11 21',WepickRank__gte=20).aggregate(
        *[{'$group':{'_id':'$DealId'}}],allowDiskUse=True)
goal_list=[elem['_id'] for elem in goal_data]

for id in goal_list:
    deal=DealW2v.objects(pk=id).first()
    if deal != None:
        if id not in deal_dict:
            deal_dict.append(id)

print('Number of Actual Users: ',len(data))

np.save('dict_'+HISTORY_FROM+'_'+DEAL_TO+'_for_sparse.npy',deal_dict)

np.save('user_'+HISTORY_FROM+'_'+DEAL_TO+'.npy',user_dict)

with open(data_path,'w') as f:
    json.dump(data,f)
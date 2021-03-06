from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.externals import joblib
from mongoengine import connect
from mongoengine.errors import DoesNotExist
import pickle
from os import path

from models import DealW2v
from models import PosData
from models import WepickDeal

HISTORY_FROM='04-01'
HISTORY_TO='04-10'

train_data_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'.pkl'

connect('wepickw2v',host='mongodb://localhost')

wepickdata=PosData.objects(TransDate__gte='2018-'+HISTORY_FROM+' 00',TransDate__lte='2018-'+HISTORY_TO+' 23',WepickRank__gte=20,WepickRank__lte=55).aggregate(
    *[{'$group':{'_id':'$UserId','docs':{'$push':'$$ROOT'}}}],allowDiskUse=True)
# in case cursornotfounderrror caused by very long sampling times
wepickdata=list(wepickdata)
data=[]

print('Number of Users: ',len(wepickdata))

# fake randomness variable
interrupter=0

for elem in wepickdata:
    if len(elem['docs'])>20 and len(elem['docs'])<40:
        hist=[]
        neg_samples=[]
        for doc in elem['docs']:
            result=DealW2v.objects(pk=doc['DealId']).first()
            if result != None:
                temp=result.vectorizedWords
                hist.append(temp)
                while True:
                    interrupter+=1
                    interrupter%=11
                    slot_num=doc['WepickRank']+interrupter-5
                    if slot_num<10:
                        slot_str='0'+str(slot_num)
                    else:
                        slot_str=str(slot_num)
                    search_str=doc['TransDate']+' '+slot_str

                    neg_result=neg_sample=WepickDeal.objects(pk=search_str).first()
                    try:
                        musttrue=hasattr(neg_result,'deal')
                    except DoesNotExist:
                        continue
                    if neg_result!=None:
                        if neg_result.deal.id != doc['DealId']:
                            neg_sample=neg_result.deal.vectorizedWords
                            neg_samples.append(neg_sample)
                            break

        data.append([hist,neg_samples])

print('Number of Actual Users: ',len(data))

with open('wp_pos_04-01_04-10.pkl','wb') as f:
    pickle.dump(data,f,pickle.HIGHEST_PROTOCOL)
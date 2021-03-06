﻿import numpy as np
from mongoengine import connect
import json

from models import DealW2v
from models import PosData
from models import WepickDeal
from models import Category2

HISTORY_FROM='03-11'
HISTORY_TO='04-10'

data_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_cate.json'

connect('wprec',host='mongodb://10.102.61.251:27017')

categories=Category2.objects()
cate_dict=[elem.id for elem in categories]
cate_finder=dict(zip(cate_dict,range(len(cate_dict))))

# cursor=PosData.objects(TransDate__gte='2018-'+HISTORY_FROM+' 00',TransDate__lte='2018-'+HISTORY_TO+' 23',WepickRank__gte=20,WepickRank__lte=85)
cursor=PosData.objects(TransDate__gte='2018-'+HISTORY_FROM+' 00',TransDate__lte='2018-'+'04-11'+' 20',WepickRank__gte=20,WepickRank__lte=90)

wepickdata=cursor.aggregate(
    *[{'$lookup':{'from':'dealw2v','localField':'DealId','foreignField':'_id','as':'deal'}},
        {'$group':{'_id':'$UserId','docs':{'$push':'$deal.category2'}}}],allowDiskUse=True)
# in case cursornotfounderrror caused by very long sampling times
wepickdata=list(wepickdata)
print('number of users: ',len(wepickdata))
data=[]

user_dict=[]


for elem in wepickdata:
    if len(elem['docs'])>30 and len(elem['docs'])<200:
        user_dict.append(elem['_id'])
        hist=[cate_finder[elem2[0]] for elem2 in elem['docs']]
        hist=list(set(hist))
        data.append(hist)


print('Number of Actual Users: ',len(data))

np.save('cate_dict.npy',cate_dict)

np.save('user_'+HISTORY_FROM+'_'+HISTORY_TO+'_for_cate.npy',user_dict)

with open(data_path,'w') as f:
    json.dump(data,f)
from sklearn.linear_model import LogisticRegression
import pandas as pd
from mongoengine import connect

from models import DealW2v
from models import PosData

connect('wepickw2v',host='mongodb://localhost')

wepickdata=PosData.objects(TransDate__gte='2018-04-01 00',TransDate__lte='2018-04-11 23',WepickRank__gte=20).aggregate(
    *[{'$group':{'_id':'$UserId','docs':{'$push':'$$ROOT'}}}],allowDiskUse=True)

data=[]

for elem in wepickdata:
    if len(elem['docs'])>20:
        hist=[DealW2v.objects(v=doc['DealId'])['vectorizedWords'] for doc in elem['docs']]
        data.append(hist)

train_data=[]
train_label=[]
test_data=[]
test_label=[]
for hist in data:
    hist_sum=[0]*100
    for i in range(len(hist)-2):
        hist_sum=[sum(x) for x in zip(hist_sum,hist[i])]
        train_data.append(hist_sum)
        train_label.append(hist[i+1])
    hist_sum=[sum(x) for x in zip(hist_sum,hist[-2])]
    test_data.append(hist_sum)
    test_label.append(hist[-1])


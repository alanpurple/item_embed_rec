from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from mongoengine import connect

from models import DealW2v
from models import PosData

connect('wepickw2v',host='mongodb://localhost')

wepickdata=PosData.objects(TransDate__gte='2018-04-10 00',TransDate__lte='2018-04-11 23',WepickRank__gte=20).aggregate(
    *[{'$group':{'_id':'$UserId','docs':{'$push':'$$ROOT'}}}],allowDiskUse=True)
# in case cursornotfounderrror caused by very long sampling times
wepickdata=list(wepickdata)
data=[]

for elem in wepickdata:
    if len(elem['docs'])>5:
        hist=[]
        for doc in elem['docs']:
            result=DealW2v.objects(v=doc['DealId']).first()
            if result != None:
                temp=result.vectorizedWords['values']
                if len(temp)==100:
                    if temp[0]!=0 and temp[1]!=0 and temp[2]!=0:
                        hist.append(temp)
        sampled=DealW2v.objects().aggregate(*[{'$sample':{'size':len(hist)*5}}])
        sampled_v=[]
        for sample in sampled:
            if sample['v'] not in elem['docs']:
                t= sample['vectorizedWords']['values']
                if len(t)==100:
                    if t[0]!=0 and t[1]!=0 and t[2]!=0:
                        sampled_v.append(t)
        neg_samples=[]
        for e in hist:
            # choose negative samples from farthest
            dist=[pdist([e,a])[0] for a in sampled_v]
            max_index=np.argmax(dist)
            neg_samples.append(sampled_v[max_index])
            del sampled_v[max_index]
        data.append([hist,neg_samples])
        

train_data=[]
train_label=[]
test_data=[]
test_label=[]
for pair in data:
    hist=pair[0]
    neg_samples=pair[1]
    hist_sum=[0]*100
    for i in range(len(hist)-2):
        hist_sum=[sum(x) for x in zip(hist_sum,hist[i])]
        train_data.append([hist_sum,hist[i+1]])
        train_label.append(1)
        train_data.append([hist_sum,neg_samples[i+1]])
        train_label.append(0)
    hist_sum=[sum(x) for x in zip(hist_sum,hist[-2])]
    test_data.append([hist_sum,hist[-1]])
    test_label.append(1)
    test_data.append([hist_sum,neg_samples[-1]])
    test_label.append(0)

assert len(train_data)==len(train_label)
assert len(test_data)==len(test_label)

lr=LogisticRegression()
lr.fit(train_data,train_label)

score=lr.score(test_data,test_label)

print(score)
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
from mongoengine import connect

from models import DealW2v
from models import PosData
from models import WepickDeal

connect('wepickw2v',host='mongodb://localhost')

wepickdata=PosData.objects(TransDate__gte='2018-04-08 00',TransDate__lte='2018-04-11 23',WepickRank__gte=20,WepickRank_lte=55).aggregate(
    *[{'$group':{'_id':'$UserId','docs':{'$push':'$$ROOT'}}}],allowDiskUse=True)
# in case cursornotfounderrror caused by very long sampling times
wepickdata=list(wepickdata)
data=[]

# fake randomness variable
interrupter=0

for i,elem in enumerate(wepickdata):
    if len(elem['docs'])>40:
        hist=[]
        neg_samples=[]
        for doc in elem['docs']:
            result=DealW2v.objects(v=doc['DealId']).first()
            if result != None:
                temp=result.vectorizedWords['values']
                if len(temp)==100:
                    if temp[0]!=0 and temp[1]!=0 and temp[2]!=0:
                        hist.append(temp)
                        while True:
                            interrupter+=1
                            interrupter%=11
                            # 0 insertion is not needed since we queried only transactions with wepickrank more than 19
                            search_str=doc['TransDate']+' '+str(doc['WepickRank']+interrupter-5)

                            neg_result=neg_sample=WepickDeal.objects(_id=search_str).first()
                            if neg_result!=None:
                                if neg_result.deal.v != doc['DealId']:
                                    neg_sample=neg_result.deal.vectorizedWords['values']
                                    neg_samples.append(neg_sample)
                                    break

        #sampled=DealW2v.objects().aggregate(*[{'$sample':{'size':len(hist)*3}}])
        #sampled_v=[]
        #for sample in sampled:
        #    if sample['v'] not in elem['docs']:
        #        t= sample['vectorizedWords']['values']
        #        if len(t)==100:
        #            if t[0]!=0 and t[1]!=0 and t[2]!=0:
        #                sampled_v.append(t)
        
        #for e in hist:
        #    # choose negative samples from farthest
        #    dist=[pdist([e,a])[0] for a in sampled_v]
        #    max_index=np.argmax(dist)
        #    neg_samples.append(sampled_v[max_index])
        #    del sampled_v[max_index]
        data.append([hist,neg_samples])
    if len(data)>300:
        break
        

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
        train_data.append(hist_sum+hist[i+1])
        train_label.append(1)
        train_data.append(hist_sum+neg_samples[i+1])
        train_label.append(0)
    hist_sum=[sum(x) for x in zip(hist_sum,hist[-2])]
    test_data.append(hist_sum+hist[-1])
    test_label.append(1)
    test_data.append(hist_sum+neg_samples[-1])
    test_label.append(0)

assert len(train_data)==len(train_label)
assert len(test_data)==len(test_label)

lr=LogisticRegression()
lr.fit(train_data,train_label)

score=lr.score(test_data,test_label)

print(score)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.utils import shuffle
import pickle
import numpy as np

with open('train.pkl','rb') as f:
    product_train=pickle.load(f)
    retrieved_train=pickle.load(f)
    user_train=pickle.load(f)
    labels_train=pickle.load(f)

with open('test.pkl','rb') as f:
    product_test=pickle.load(f)
    retrieved_test=pickle.load(f)
    user_test=pickle.load(f)
    labels_test=pickle.load(f)

with open('history.pkl','rb') as f:
    history=pickle.load(f)

train_data=[]
test_data=[]

for i,prod_len in enumerate(product_train):
    if prod_len>49:
        prod_len=50
        zero_padding_len=0
    else:
        zero_padding_len=50-prod_len
    temp=[user_train[i],retrieved_train[i]]
    temp+=[0]*zero_padding_len
    temp_history=history[user_train[i]][-prod_len:]
    temp+=temp_history
    assert len(temp)==52
    
    train_data.append(temp)

for i,prod_len in enumerate(product_test):
    if prod_len>49:
        prod_len=50
        zero_padding_len=0
    else:
        zero_padding_len=50-prod_len
    temp=[user_test[i],retrieved_test[i]]
    temp+=[0]*zero_padding_len
    temp_history=history[user_test[i]][-prod_len:]
    temp+=temp_history

    assert len(temp)==52
    
    test_data.append(temp)

train_data,labels_train=shuffle(train_data,labels_train,random_state=2000)

gbc=GradientBoostingClassifier()

gbc.fit(train_data,labels_train)

print(gbc.feature_importances_)

score=gbc.score(test_data,labels_test)

print('The test score is : {:.4f}'.format(score))
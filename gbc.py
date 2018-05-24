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

# each of size 50, filled with 0 if history has length less than 50
history=np.load('history.npy')

train_data=[]
test_data=[]

for i,prod_len in enumerate(product_train):
    assert len(history[user_train[i]])<51
    zero_padding_len=50-prod_len
    temp=[user_train[i],retrieved_train[i]]
    temp_history=history[user_train[i]][:prod_len].tolist()
    temp_history.reverse()
    temp+=temp_history
    temp+=[0]*zero_padding_len
    train_data.append(temp)

for i,prod_len in enumerate(product_test):
    assert len(history[user_test[i]])<51
    zero_padding_len=50-prod_len
    temp=[user_test[i],retrieved_test[i]]
    temp_history=history[user_test[i]][:prod_len].tolist()
    temp_history.reverse()
    temp+=temp_history
    temp+=[0]*zero_padding_len
    test_data.append(temp)

train_data,labels_train=shuffle(train_data,labels_train,random_state=2000)

gbc=GradientBoostingClassifier()

gbc.fit(train_data,labels_train)

print(gbc.feature_importances_)

score=gbc.score(test_data,labels_test)

print('The test score is : {:.4f}'.format(score))
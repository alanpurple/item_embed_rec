from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.utils import shuffle
from sklearn.externals import joblib
from matplotlib import pyplot
from mongoengine import connect
from mongoengine.errors import DoesNotExist
import pickle

from models import DealW2v
from models import PosData
from models import WepickDeal

connect('wepickw2v',host='mongodb://localhost')

HISTORY_FROM='04-01'
HISTORY_TO='04-10'

train_data_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'.pkl'

with open(train_data_path,'rb') as f:
    data=pickle.load(f)

print('Number of Actual Users: ',len(data))

train_data=[]
train_label=[]
test_data=[]
test_label=[]
# history value is just a summation, not an average
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

print('train data length: ',len(train_data))

# need scaling for use of Stochastic Average Gradient descent solver ( much faster )
scaler=StandardScaler()
scaler.fit(train_data)

joblib.dump(scaler,'scaler.pkl')

X=scaler.transform(train_data)
X_test=scaler.transform(test_data)
gbc=GradientBoostingClassifier()
gbc.fit(X,train_label)

joblib.dump(gbc,'wpgbc.pkl')

score=gbc.score(X_test,test_label)

print(score)

print('probability for a few results: \n')
print(gbc.predict_proba(test_data[:10]))
print('original class of above data: \n')
print(test_label[:10])

predicted_probs=gbc.predict_proba(X_test)[:,1]

fpr,tpr,threshold=roc_curve(test_label,predicted_probs,pos_label=1)

pyplot.plot(fpr,tpr)
pyplot.xlabel('False positive rate')
pyplot.ylabel('True positive rate')
pyplot.title('ROC curve')
pyplot.legend(loc='best')

auc_score=roc_auc_score(test_label,predicted_probs)

print('auc score: {:.4f}'.format(auc_score))

pyplot.show()
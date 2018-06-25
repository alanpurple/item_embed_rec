from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.utils import shuffle
from sklearn.externals import joblib
import pickle
import tensorflow as tf

HISTORY_FROM='04-01'
HISTORY_TO='04-10'

train_data_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'.pkl'
train_tfr_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_train.tfrecord'
test_tfr_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_test.tfrecord'

with open(train_data_path,'rb') as f:
    data=pickle.load(f)

print('Number of Actual Users: ',len(data))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_list_feature(elems):
    return tf.train.Feature(float_list=tf.train.FloatList(value=elems))

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
train_data=scaler.transform(train_data)
test_data=scaler.transform(test_data)

joblib.dump(scaler,'scaler.pkl')

with tf.python_io.TFRecordWriter(train_tfr_path) as writer:
    for i in range(len(train_data)):
        example=tf.train.Example(
            features=tf.train.Features(
                feature={
                    'profile':_float_list_feature(train_data[i]),
                    'label':_int64_feature(train_label[i])
                    }))
        writer.write(example.SerializeToString())

with tf.python_io.TFRecordWriter(test_tfr_path) as writer:
    for i in range(len(test_data)):
        example=tf.train.Example(
            features=tf.train.Features(
                feature={
                    'profile':_float_list_feature(test_data[i]),
                    'label':_int64_feature(test_label[i])
                    }))
        writer.write(example.SerializeToString())
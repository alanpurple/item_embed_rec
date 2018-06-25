import tensorflow as tf

HISTORY_FROM='04-01'
HISTORY_TO='04-10'

train_tfr_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_train.tfrecord'
test_tfr_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_test.tfrecord'

def input_fn(file,num_epochs,shuffle,batch_size):
    dataset=tf.data.TFRecordDataset([file])

    def parser(record):
        keys_to_features={
            'profile':tf.FixedLenFeature((200),tf.float32),
            'label':tf.FixedLenFeature((),tf.int64)
            }
        parsed=tf.parse_single_example(record,keys_to_features)
        label=parsed.pop('label')
        return parsed,label

    dataset=dataset.map(parser,4)
    if shuffle:
        dataset=dataset.shuffle(10000)
    dataset=dataset.batch(batch_size)
    dataset=dataset.repeat(num_epochs)

    iterator=dataset.make_one_shot_iterator()
    return iterator.get_next()

model=tf.estimator.LinearClassifier([
    tf.feature_column.numeric_column('profile',200)
    ],'lr_tf_models')
model.train(lambda: input_fn(train_tfr_path,10,True,256))

model.evaluate(lambda: input_fn(test_tfr_path,1,True,32))
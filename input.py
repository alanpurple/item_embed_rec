import tensorflow as tf

def make_input_fn(filename,max_history_len,batch_size,num_epochs):
    def input_fn():

        def parser(example_proto):
            features={
                'user_id':tf.FixedLenFeature((),tf.int64,0),
                'hist_len':tf.FixedLenFeature((),tf.int64,0),
                'retrieved':tf.FixedLenFeature((),tf.int64,0),
                'label':tf.FixedLenFeature((),tf.int64,1),
                'hist':tf.FixedLenFeature((max_history_len),tf.int64)
            }
            parsed=tf.parse_single_example(example_proto,features)
            label=parsed.pop('label')
            return parsed,label
        dataset=tf.data.TFRecordDataset(filename)
        dataset=dataset.map(parser)
        dataset=dataset.shuffle(1000)
        dataset=dataset.batch(batch_size)
        dataset=dataset.repeat(num_epochs)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn
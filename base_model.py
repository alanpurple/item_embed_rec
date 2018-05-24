import tensorflow as tf

def base_model(features, labels, mode, params):
    # user_id=features['user_id']
    hist_len=features['hist_len']
    retrieved=features['retrieved']
    user_history=features['hist']
    print(tf.shape(user_history))
    item_len=params['item_len']
    hidden_len=params['num_hidden']
    category_list=params['categories']
    learning_rate=params['learning_rate']

    item_emb_w=tf.get_variable('item_emb_w',[item_len,hidden_len//2])
    item_b=tf.get_variable('item_b',[item_len],initializer=tf.constant_initializer(0.0))
    cate_emb_w=tf.get_variable('cate_emb_w',[item_len,hidden_len//2])
    # category_list=tf.convert_to_tensor(category_list,tf.int64)

    retrieved_category=tf.gather(category_list,retrieved)
    retrieved_emb_with_cate=tf.concat([
        tf.nn.embedding_lookup(item_emb_w,retrieved),   # [B,H/2]
        tf.nn.embedding_lookup(cate_emb_w, retrieved_category) # [B,H/2]
        ],1)    # [B,H]
    retrieved_b= tf.gather(item_b,retrieved)

    history_cat=tf.gather(category_list,user_history)
    print(tf.shape(history_cat))
    history_emb=tf.concat([
        tf.nn.embedding_lookup(item_emb_w,user_history), # [B,T, H/2]
        tf.nn.embedding_lookup(cate_emb_w,history_cat)   # [B,T, H/2]
    ],2) # [B,T,H]

    hist_sum=tf.reduce_sum(history_emb,1)   # [B,H]
    hist_mean=tf.div(hist_sum,tf.cast(tf.tile(tf.expand_dims(hist_len,1),[1,hidden_len]),tf.float32))

    training = mode==tf.estimator.ModeKeys.TRAIN
    hist_mean=tf.layers.batch_normalization(hist_mean,training=training)
    hist_mean=tf.layers.dense(hist_mean,hidden_len)
    retrieved_plus_hist=tf.concat([hist_mean,retrieved_emb_with_cate],1,name='concat_retrieved_with_history')
    retrieved_plus_hist=tf.layers.batch_normalization(retrieved_plus_hist,training=training,name='fc')
    fc_retrieved1 = tf.layers.dense(retrieved_plus_hist,80,tf.nn.sigmoid,name='fc1')
    fc_retrieved2 = tf.layers.dense(fc_retrieved1,40,tf.nn.sigmoid,name='fc2')
    fc_retrieved3 = tf.layers.dense(fc_retrieved2,1,name='fc3')
    

    logits = retrieved_b + fc_retrieved3[0]
    loss=tf.losses.sigmoid_cross_entropy(labels,logits)

    ######################
    # training graph ends here
    if training:
        update_ops=tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer=tf.train.AdamOptimizer(learning_rate)
        with tf.control_dependencies(update_ops):
            train_op=optimizer.minimize(loss,tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode,loss=loss,train_op=train_op)
    ######################

    prediction=tf.sigmoid(logits)
    if mode==tf.estimator.ModeKeys.EVAL:
        eval_metric_ops={
            'auc':tf.metrics.auc(labels,prediction)
        }
        return tf.estimator.EstimatorSpec(mode,loss=loss,eval_metric_ops=eval_metric_ops)

    hist_emb_all = tf.tile(tf.expand_dims(hist_mean,1),[-1,item_len,1]) # [B,I,H]

    all_product_emb=tf.concat([
        item_emb_w, # [I,H/2]
        tf.nn.embedding_lookup(cate_emb_w,category_list)    #[I,H/2]
        ],1) # [I,H]
    all_product_emb=tf.tile(all_product_emb,[tf.shape(hist_emb_all)[0],1,1])    #[B,I,H]
    hist_emb_all=tf.concat([hist_emb_all,all_product_emb],2)    # [B,I,2H]
    hist_emb_all=tf.layers.batch_normalization(hist_emb_all,name='fc',reuse=True)
    fc_all1=tf.layers.dense(hist_emb_all,80,tf.nn.sigmoid,name='fc1',reuse=True)
    fc_all2=tf.layers.dense(fc_all1,40,tf.nn.sigmoid,name='fc2',reuse=True)
    fc_all3=tf.layers.dense(fc_all2,1,name='fc3',reuse=True) # [B,I,1]
    fc_all3=tf.reshape(fc_all3,[-1,item_len])    #[B,I]

    logits_all=tf.sigmoid(item_b+fc_all3)

    return tf.estimator.EstimatorSpec(mode,logits_all)

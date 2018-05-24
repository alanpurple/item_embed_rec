import tensorflow as tf

def din_model(features,labels,mode,params):
    hist_len=features['hist_len']
    retrieved=features['retrieved']
    user_history=features['hist']
    item_len=params['item_len']
    hidden_len=params['num_hidden']
    category_list=params['categories']
    learning_rate=params['learning_rate']

    item_emb_w=tf.get_variable('item_emb_w',[item_len,hidden_len//2])
    item_b=tf.get_variable('item_b',[item_len],initializer=tf.constant_initializer(0.0))
    cate_emb_w=tf.get_variable('cate_emb_w',[item_len,hidden_len//2])

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

    ######################
    # Attention mechanism
    ######################

    queries_hidden_units = retrieved_emb_with_cate.get_shape().as_list()[-1]
    queries = tf.tile(retrieved_emb_with_cate, [1, tf.shape(history_emb)[1]])
    queries = tf.reshape(queries, [-1, tf.shape(history_emb)[1], queries_hidden_units])
    din_all = tf.concat([queries, history_emb, queries-history_emb, queries*history_emb], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(history_emb)[1]])
    outputs = d_layer_3_all 
    # Mask
    key_masks = tf.sequence_mask(hist_len, tf.shape(history_emb)[1])   # [B, T]
    key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
    paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
    outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

    # Scale
    outputs = outputs / (history_emb.get_shape().as_list()[-1] ** 0.5)
        
    # Activation
    outputs = tf.nn.softmax(outputs)  # [B, 1, T]

    # Weighted sum
    outputs = tf.matmul(outputs, history_emb)  # [B, 1, H]

    ######################
    # Attention ends
    ######################

    training = mode==tf.estimator.ModeKeys.TRAIN
    outputs=tf.layers.batch_normalization(outputs,training=training)
    outputs=tf.layers.dense(outputs,hidden_len)
    retrieved_plus_hist=tf.concat([outputs,retrieved_emb_with_cate],1,name='concat_retrieved_with_history')
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

    hist_emb_all = tf.tile(tf.expand_dims(outputs,1),[-1,item_len,1]) # [B,I,H]

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
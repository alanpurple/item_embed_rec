import tensorflow as tf

def din_model(features,labels,mode,params):
    user_id=features['user_id']
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
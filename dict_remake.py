import numpy as np
import json


HISTORY_FROM='03-11'
HISTORY_TO='04-10'

data_path='wp_'+HISTORY_FROM+'_'+HISTORY_TO+'_cate.json'

with open('temp.json','r') as f:
    prev_data=json.load(f)

data=[list(set(elem)) for elem in prev_data]

with open(data_path,'w') as f:
    json.dump(data,f)
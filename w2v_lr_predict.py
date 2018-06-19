from sklearn.externals import joblib
from mongoengine import connect
from mongoengine.errors import DoesNotExist
import statistics

from models import DealW2v
from models import PosData
from models import WepickDeal

connect('wepickw2v',host='mongodb://localhost')

#wepickdata=PosData.objects(TransDate__gte='2018-04-01 00',TransDate__lte='2018-04-10 23',WepickRank__gte=20,WepickRank__lte=55).aggregate(
#    *[{'$group':{'_id':'$UserId','docs':{'$push':'$$ROOT'}}}],allowDiskUse=True)
# in case cursornotfounderrror caused by very long sampling times
#wepickdata=list(wepickdata)
#data=[]

userstat=PosData.objects(TransDate__gte='2018-04-01 00',TransDate__lte='2018-04-10 23',WepickRank__gte=20,WepickRank__lte=55).aggregate(
    *[{'$group':{'_id':'$UserId','count':{'$sum':1}}}],allowDiskUse=True)

len_list=[elem['count'] for elem in userstat]

print('mean: ',statistics.mean(len_list))
print('harmonic mean: ',statistics.harmonic_mean(len_list))
print('min: ',min(len_list))
print('max: ',max(len_list))

long_len_list=[elem for elem in len_list if elem>20]

print(len(long_len_list))
print(long_len_list[:10])

print('mean: ',statistics.mean(long_len_list))
print('harmonic mean: ',statistics.harmonic_mean(long_len_list))
print('min: ',min(long_len_list))
print('max: ',max(long_len_list))

#print('Number of Users: ')
#print(len(wepickdata))